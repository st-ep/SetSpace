from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from case_studies.point_cloud_consistency.common import build_model_from_config, load_json, save_json
from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset
from case_studies.point_cloud_consistency.models import PointCloudMeanRegressor, PointCloudSetClassifier
from case_studies.shared import make_view_seeds as _make_view_seeds_generic
from case_studies.shared import save_training_artifacts as _save_artifacts
from case_studies.shared import train_loop

_SPHERE_MODE_OFFSETS = {"uniform": 11, "polar": 23, "equatorial": 31, "clustered": 47, "hemisphere": 59}


def _make_view_seeds(
    split: str,
    local_indices: torch.Tensor,
    *,
    n_points: int,
    sampling_mode: str,
    replica_idx: int,
) -> torch.Tensor:
    return _make_view_seeds_generic(
        split, local_indices, n_points=n_points, sampling_mode=sampling_mode,
        replica_idx=replica_idx, mode_offsets=_SPHERE_MODE_OFFSETS,
    )


def _is_regressor(model: torch.nn.Module) -> bool:
    return isinstance(model, PointCloudMeanRegressor)


def train_classifier(
    model: PointCloudSetClassifier,
    dataset: SyntheticSurfaceSignalDataset,
    *,
    run_name: str | None = None,
    device: torch.device,
    train_points: int,
    train_sampling_mode: str,
    batch_size: int,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    eval_every: int,
    val_sampling_modes: list[str],
    val_objects: int,
    seed: int,
) -> dict:
    def train_step(m, _step):
        coords, values, labels = dataset.sample_batch(
            "train", batch_size=batch_size, n_points=train_points,
            sampling_mode=train_sampling_mode, device=device,
        )
        logits = m(coords, values)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        return loss, {"acc": acc}

    def eval_fn(m):
        summary = evaluate_classifier(
            m, dataset, split="val", device=device, point_counts=[train_points],
            sampling_modes=val_sampling_modes, n_resamples=1,
            reference_points=max(train_points * 4, 1024),
            batch_size=batch_size, max_objects=val_objects,
        )
        if any(mode != "uniform" for mode in val_sampling_modes):
            return summary["aggregate"]["avg_nonuniform_accuracy"][str(train_points)]
        return summary["aggregate"]["accuracy_by_count"][str(train_points)]["uniform"]

    return train_loop(
        model, run_name=run_name, device=device, steps=steps, lr=lr,
        weight_decay=weight_decay, grad_clip=grad_clip, eval_every=eval_every,
        seed=seed, train_step_fn=train_step, eval_fn=eval_fn, higher_is_better=True,
    )


def train_regressor(
    model: PointCloudMeanRegressor,
    dataset: SyntheticSurfaceSignalDataset,
    *,
    run_name: str | None = None,
    device: torch.device,
    train_points: int,
    train_sampling_mode: str,
    batch_size: int,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    eval_every: int,
    val_sampling_modes: list[str],
    val_objects: int,
    seed: int,
) -> dict:
    def train_step(m, _step):
        coords, values, _, local_indices = dataset.sample_batch_with_indices(
            "train", batch_size=batch_size, n_points=train_points,
            sampling_mode=train_sampling_mode, device=device,
        )
        targets = dataset.get_integral_targets("train", local_indices.detach().cpu(), device=device)
        preds = m(coords, values)
        loss = F.mse_loss(preds, targets)
        with torch.no_grad():
            mae = (preds - targets).abs().mean().item()
        return loss, {"mae": mae}

    def eval_fn(m):
        summary = evaluate_regressor(
            m, dataset, split="val", device=device, point_counts=[train_points],
            sampling_modes=val_sampling_modes, n_resamples=1,
            reference_points=max(train_points * 4, 1024),
            batch_size=batch_size, max_objects=val_objects,
        )
        return summary["aggregate"]["avg_nonuniform_rmse"][str(train_points)]

    return train_loop(
        model, run_name=run_name, device=device, steps=steps, lr=lr,
        weight_decay=weight_decay, grad_clip=grad_clip, eval_every=eval_every,
        seed=seed, train_step_fn=train_step, eval_fn=eval_fn, higher_is_better=False,
    )


def _evaluate_consistency(
    model: torch.nn.Module,
    dataset: SyntheticSurfaceSignalDataset,
    *,
    split: str,
    device: torch.device,
    point_counts: list[int],
    sampling_modes: list[str],
    n_resamples: int,
    reference_points: int,
    batch_size: int,
    max_objects: int | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Shared evaluation core for both classifier and regressor."""
    model.eval()
    is_regression = _is_regressor(model)
    n_objects = dataset.split_size(split)
    if max_objects is not None:
        n_objects = min(n_objects, int(max_objects))

    local_indices = torch.arange(n_objects, dtype=torch.long)

    # Collect reference outputs
    ref_outputs_list = []
    ref_embeddings_list = []
    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            idx = local_indices[start : start + batch_size]
            ref_seeds = _make_view_seeds(split, idx, n_points=reference_points, sampling_mode="uniform", replica_idx=0)
            coords, values, _ = dataset.collate_views(
                split, idx, n_points=reference_points, sampling_mode="uniform",
                view_seeds=ref_seeds, device=device,
            )
            if is_regression:
                ref_outputs_list.append(model(coords, values).cpu())
            else:
                logits = model(coords, values)
                ref_outputs_list.append(logits.cpu())
                ref_embeddings_list.append(model.embed(coords, values).cpu())

    ref_outputs = torch.cat(ref_outputs_list, dim=0)
    ref_embeddings = torch.cat(ref_embeddings_list, dim=0) if ref_embeddings_list else None

    # Ground truth for regression
    targets = dataset.get_integral_targets(split, local_indices).cpu() if is_regression else None

    # Evaluate per (mode, n_points)
    per_setting: dict[str, dict[str, dict[str, float]]] = {}
    for mode in sampling_modes:
        per_setting[mode] = {}
        for n_points in point_counts:
            totals: dict[str, float] = {}
            count = 0
            with torch.no_grad():
                for replica_idx in range(n_resamples):
                    for start in range(0, n_objects, batch_size):
                        idx = local_indices[start : start + batch_size]
                        view_seeds = _make_view_seeds(
                            split, idx, n_points=n_points, sampling_mode=mode,
                            replica_idx=replica_idx + 1,
                        )
                        coords, values, labels = dataset.collate_views(
                            split, idx, n_points=n_points, sampling_mode=mode,
                            view_seeds=view_seeds, device=device,
                        )
                        ref_out = ref_outputs[start : start + len(idx)]

                        if is_regression:
                            preds = model(coords, values).cpu()
                            tgt = targets[start : start + len(idx)]
                            error = preds - tgt
                            totals["squared_error"] = totals.get("squared_error", 0.0) + float(error.square().sum().item())
                            totals["absolute_error"] = totals.get("absolute_error", 0.0) + float(error.abs().sum().item())
                            totals["signed_error"] = totals.get("signed_error", 0.0) + float(error.sum().item())
                            totals["prediction_drift"] = totals.get("prediction_drift", 0.0) + float((preds - ref_out).abs().sum().item())
                        else:
                            logits = model(coords, values).cpu()
                            embeddings = model.embed(coords, values).cpu()
                            preds = logits.argmax(dim=-1)
                            ref_emb = ref_embeddings[start : start + len(idx)]
                            ref_preds = ref_out.argmax(dim=-1)

                            totals["accuracy"] = totals.get("accuracy", 0.0) + float((preds == labels.cpu()).float().sum().item())
                            totals["prediction_consistency"] = totals.get("prediction_consistency", 0.0) + float((preds == ref_preds).float().sum().item())
                            totals["embedding_drift"] = totals.get("embedding_drift", 0.0) + float(
                                ((embeddings - ref_emb).norm(dim=1) / ref_emb.norm(dim=1).clamp_min(1e-8)).sum().item()
                            )
                            totals["logit_drift"] = totals.get("logit_drift", 0.0) + float(
                                ((logits - ref_out).norm(dim=1) / ref_out.norm(dim=1).clamp_min(1e-8)).sum().item()
                            )
                        count += len(idx)

            denom = float(count)
            if is_regression:
                per_setting[mode][str(n_points)] = {
                    "rmse": (totals["squared_error"] / denom) ** 0.5,
                    "mae": totals["absolute_error"] / denom,
                    "bias": totals["signed_error"] / denom,
                    "prediction_drift": totals["prediction_drift"] / denom,
                }
            else:
                per_setting[mode][str(n_points)] = {
                    "accuracy": totals["accuracy"] / denom,
                    "embedding_drift": totals["embedding_drift"] / denom,
                    "logit_drift": totals["logit_drift"] / denom,
                    "prediction_consistency": totals["prediction_consistency"] / denom,
                }

    return per_setting, ref_outputs, ref_embeddings, targets


def _aggregate_classifier_metrics(
    per_setting: dict, point_counts: list[int], sampling_modes: list[str],
) -> dict:
    accuracy_by_count = {}
    embedding_drift_by_count = {}
    logit_drift_by_count = {}
    consistency_by_count = {}
    worst_case_accuracy = {}
    for n_points in point_counts:
        key = str(n_points)
        accuracy_by_count[key] = {mode: per_setting[mode][key]["accuracy"] for mode in sampling_modes}
        embedding_drift_by_count[key] = {mode: per_setting[mode][key]["embedding_drift"] for mode in sampling_modes}
        logit_drift_by_count[key] = {mode: per_setting[mode][key]["logit_drift"] for mode in sampling_modes}
        consistency_by_count[key] = {mode: per_setting[mode][key]["prediction_consistency"] for mode in sampling_modes}
        worst_case_accuracy[key] = min(accuracy_by_count[key].values())

    nonuniform_modes = [mode for mode in sampling_modes if mode != "uniform"]
    if nonuniform_modes:
        avg_nonuniform_accuracy = {
            str(n): sum(per_setting[m][str(n)]["accuracy"] for m in nonuniform_modes) / len(nonuniform_modes)
            for n in point_counts
        }
        avg_nonuniform_embedding_drift = {
            str(n): sum(per_setting[m][str(n)]["embedding_drift"] for m in nonuniform_modes) / len(nonuniform_modes)
            for n in point_counts
        }
    else:
        avg_nonuniform_accuracy = {str(n): per_setting["uniform"][str(n)]["accuracy"] for n in point_counts}
        avg_nonuniform_embedding_drift = {str(n): per_setting["uniform"][str(n)]["embedding_drift"] for n in point_counts}

    return {
        "accuracy_by_count": accuracy_by_count,
        "embedding_drift_by_count": embedding_drift_by_count,
        "logit_drift_by_count": logit_drift_by_count,
        "consistency_by_count": consistency_by_count,
        "worst_case_accuracy": worst_case_accuracy,
        "avg_nonuniform_accuracy": avg_nonuniform_accuracy,
        "avg_nonuniform_embedding_drift": avg_nonuniform_embedding_drift,
    }


def _aggregate_regressor_metrics(
    per_setting: dict, point_counts: list[int], sampling_modes: list[str],
) -> dict:
    rmse_by_count = {}
    mae_by_count = {}
    bias_by_count = {}
    prediction_drift_by_count = {}
    worst_case_rmse = {}
    for n_points in point_counts:
        key = str(n_points)
        rmse_by_count[key] = {mode: per_setting[mode][key]["rmse"] for mode in sampling_modes}
        mae_by_count[key] = {mode: per_setting[mode][key]["mae"] for mode in sampling_modes}
        bias_by_count[key] = {mode: per_setting[mode][key]["bias"] for mode in sampling_modes}
        prediction_drift_by_count[key] = {mode: per_setting[mode][key]["prediction_drift"] for mode in sampling_modes}
        worst_case_rmse[key] = max(rmse_by_count[key].values())

    nonuniform_modes = [mode for mode in sampling_modes if mode != "uniform"]
    if nonuniform_modes:
        avg_nonuniform_rmse = {
            str(n): sum(per_setting[m][str(n)]["rmse"] for m in nonuniform_modes) / len(nonuniform_modes)
            for n in point_counts
        }
        avg_nonuniform_prediction_drift = {
            str(n): sum(per_setting[m][str(n)]["prediction_drift"] for m in nonuniform_modes) / len(nonuniform_modes)
            for n in point_counts
        }
    else:
        avg_nonuniform_rmse = {str(n): per_setting["uniform"][str(n)]["rmse"] for n in point_counts}
        avg_nonuniform_prediction_drift = {str(n): per_setting["uniform"][str(n)]["prediction_drift"] for n in point_counts}

    return {
        "rmse_by_count": rmse_by_count,
        "mae_by_count": mae_by_count,
        "bias_by_count": bias_by_count,
        "prediction_drift_by_count": prediction_drift_by_count,
        "worst_case_rmse": worst_case_rmse,
        "avg_nonuniform_rmse": avg_nonuniform_rmse,
        "avg_nonuniform_prediction_drift": avg_nonuniform_prediction_drift,
    }


def evaluate_classifier(
    model: PointCloudSetClassifier,
    dataset: SyntheticSurfaceSignalDataset,
    *,
    split: str,
    device: torch.device,
    point_counts: list[int],
    sampling_modes: list[str],
    n_resamples: int,
    reference_points: int,
    batch_size: int,
    max_objects: int | None = None,
) -> dict:
    per_setting, _, _, _ = _evaluate_consistency(
        model, dataset, split=split, device=device, point_counts=point_counts,
        sampling_modes=sampling_modes, n_resamples=n_resamples,
        reference_points=reference_points, batch_size=batch_size, max_objects=max_objects,
    )
    return {"per_setting": per_setting, "aggregate": _aggregate_classifier_metrics(per_setting, point_counts, sampling_modes)}


def evaluate_regressor(
    model: PointCloudMeanRegressor,
    dataset: SyntheticSurfaceSignalDataset,
    *,
    split: str,
    device: torch.device,
    point_counts: list[int],
    sampling_modes: list[str],
    n_resamples: int,
    reference_points: int,
    batch_size: int,
    max_objects: int | None = None,
) -> dict:
    per_setting, _, _, _ = _evaluate_consistency(
        model, dataset, split=split, device=device, point_counts=point_counts,
        sampling_modes=sampling_modes, n_resamples=n_resamples,
        reference_points=reference_points, batch_size=batch_size, max_objects=max_objects,
    )
    return {"per_setting": per_setting, "aggregate": _aggregate_regressor_metrics(per_setting, point_counts, sampling_modes)}


def save_training_artifacts(
    output_dir: Path,
    *,
    model: PointCloudSetClassifier,
    dataset: SyntheticSurfaceSignalDataset,
    model_config: dict,
    training_config: dict,
    training_summary: dict,
) -> None:
    _save_artifacts(
        output_dir,
        model=model,
        dataset_config=dataset.get_config(),
        model_config=model_config,
        training_config=training_config,
        training_summary=training_summary,
    )


def load_model_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = load_json(checkpoint_dir / "experiment_config.json")
    model = build_model_from_config(cfg["model"]).to(device)
    state_dict = torch.load(checkpoint_dir / "model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg
