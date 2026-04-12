from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from case_studies.point_cloud_consistency.common import build_model_from_config
from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset
from case_studies.point_cloud_consistency.models import PointCloudMeanRegressor, PointCloudSetClassifier
from case_studies.shared import avg_over_nonuniform_modes, make_view_seeds
from case_studies.shared import save_training_artifacts as _save_artifacts
from case_studies.shared import train_loop

_SPHERE_MODE_OFFSETS = {"uniform": 11, "polar": 23, "equatorial": 31, "clustered": 47, "hemisphere": 59}


def _make_view_seeds(split, local_indices, *, n_points, sampling_mode, replica_idx):
    return make_view_seeds(
        split, local_indices, n_points=n_points, sampling_mode=sampling_mode,
        replica_idx=replica_idx, mode_offsets=_SPHERE_MODE_OFFSETS,
    )


def _is_regressor(model: torch.nn.Module) -> bool:
    return isinstance(model, PointCloudMeanRegressor) or str(getattr(model, "task", "")).lower() == "regression"


def _uses_oracle_density(model: torch.nn.Module) -> bool:
    return str(getattr(model, "weight_mode", "")).lower() == "oracle_density"


def _has_trainable_parameters(model: torch.nn.Module) -> bool:
    return any(param.requires_grad and param.numel() > 0 for param in model.parameters())


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
    use_oracle_weights = _uses_oracle_density(model)

    def train_step(m, _step):
        batch = dataset.sample_batch(
            "train",
            batch_size=batch_size,
            n_points=train_points,
            sampling_mode=train_sampling_mode,
            device=device,
            return_oracle_weights=use_oracle_weights,
        )
        if use_oracle_weights:
            coords, values, labels, oracle_weights = batch
            logits = m(coords, values, point_weights=oracle_weights)
        else:
            coords, values, labels = batch
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
    model: torch.nn.Module,
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
    use_oracle_weights = _uses_oracle_density(model)

    def train_step(m, _step):
        batch = dataset.sample_batch_with_indices(
            "train",
            batch_size=batch_size,
            n_points=train_points,
            sampling_mode=train_sampling_mode,
            device=device,
            return_oracle_weights=use_oracle_weights,
        )
        if use_oracle_weights:
            coords, values, _, local_indices, oracle_weights = batch
            preds = m(coords, values, point_weights=oracle_weights)
        else:
            coords, values, _, local_indices = batch
            preds = m(coords, values)
        targets = dataset.get_integral_targets("train", local_indices.detach().cpu(), device=device)
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

    if not _has_trainable_parameters(model):
        val_score = float(eval_fn(model))
        return {
            "seed": seed,
            "best_val_score": val_score,
            "history_tail": [{"step": 0, "val_score": val_score}],
            "optimization_skipped": True,
        }

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
    use_oracle_weights = _uses_oracle_density(model)
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
            ref_batch = dataset.collate_views(
                split,
                idx,
                n_points=reference_points,
                sampling_mode="uniform",
                view_seeds=ref_seeds,
                device=device,
                return_oracle_weights=use_oracle_weights,
            )
            if use_oracle_weights:
                coords, values, _, oracle_weights = ref_batch
            else:
                coords, values, _ = ref_batch
            if is_regression:
                ref_outputs_list.append(model(coords, values, point_weights=oracle_weights).cpu() if use_oracle_weights else model(coords, values).cpu())
            else:
                logits = model(coords, values, point_weights=oracle_weights) if use_oracle_weights else model(coords, values)
                ref_outputs_list.append(logits.cpu())
                ref_embeddings_list.append(
                    model.embed(coords, values, point_weights=oracle_weights).cpu()
                    if use_oracle_weights
                    else model.embed(coords, values).cpu()
                )

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
                        batch = dataset.collate_views(
                            split,
                            idx,
                            n_points=n_points,
                            sampling_mode=mode,
                            view_seeds=view_seeds,
                            device=device,
                            return_oracle_weights=use_oracle_weights,
                        )
                        if use_oracle_weights:
                            coords, values, labels, oracle_weights = batch
                        else:
                            coords, values, labels = batch
                        ref_out = ref_outputs[start : start + len(idx)]

                        if is_regression:
                            preds = (
                                model(coords, values, point_weights=oracle_weights).cpu()
                                if use_oracle_weights
                                else model(coords, values).cpu()
                            )
                            tgt = targets[start : start + len(idx)]
                            error = preds - tgt
                            totals["squared_error"] = totals.get("squared_error", 0.0) + float(error.square().sum().item())
                            totals["absolute_error"] = totals.get("absolute_error", 0.0) + float(error.abs().sum().item())
                            totals["signed_error"] = totals.get("signed_error", 0.0) + float(error.sum().item())
                            totals["prediction_drift"] = totals.get("prediction_drift", 0.0) + float((preds - ref_out).abs().sum().item())
                        else:
                            if use_oracle_weights:
                                logits = model(coords, values, point_weights=oracle_weights).cpu()
                                embeddings = model.embed(coords, values, point_weights=oracle_weights).cpu()
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
    def _by_count(metric: str) -> dict:
        return {str(n): {m: per_setting[m][str(n)][metric] for m in sampling_modes} for n in point_counts}

    accuracy_by_count = _by_count("accuracy")
    _avg = lambda metric: avg_over_nonuniform_modes(per_setting, metric, point_counts, sampling_modes)

    return {
        "accuracy_by_count": accuracy_by_count,
        "embedding_drift_by_count": _by_count("embedding_drift"),
        "logit_drift_by_count": _by_count("logit_drift"),
        "consistency_by_count": _by_count("prediction_consistency"),
        "worst_case_accuracy": {str(n): min(accuracy_by_count[str(n)].values()) for n in point_counts},
        "avg_nonuniform_accuracy": _avg("accuracy"),
        "avg_nonuniform_embedding_drift": _avg("embedding_drift"),
    }


def _aggregate_regressor_metrics(
    per_setting: dict, point_counts: list[int], sampling_modes: list[str],
) -> dict:
    def _by_count(metric: str) -> dict:
        return {str(n): {m: per_setting[m][str(n)][metric] for m in sampling_modes} for n in point_counts}

    rmse_by_count = _by_count("rmse")
    _avg = lambda metric: avg_over_nonuniform_modes(per_setting, metric, point_counts, sampling_modes)

    return {
        "rmse_by_count": rmse_by_count,
        "mae_by_count": _by_count("mae"),
        "bias_by_count": _by_count("bias"),
        "prediction_drift_by_count": _by_count("prediction_drift"),
        "worst_case_rmse": {str(n): max(rmse_by_count[str(n)].values()) for n in point_counts},
        "avg_nonuniform_rmse": _avg("rmse"),
        "avg_nonuniform_prediction_drift": _avg("prediction_drift"),
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
    from case_studies.shared import load_model_checkpoint as _load
    return _load(checkpoint_dir, device, build_model_from_config)
