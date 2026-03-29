from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from set_encoders import calculate_l2_relative_error

from case_studies.shared import make_view_seeds as _make_view_seeds_generic
from case_studies.shared import save_training_artifacts as _save_artifacts
from case_studies.shared import train_loop
from case_studies.sphere_signal_reconstruction.common import build_model_from_config, load_json, save_json
from case_studies.sphere_signal_reconstruction.dataset import SphereSignalDataset
from case_studies.sphere_signal_reconstruction.models import SphereSignalReconstructor

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


def train_reconstructor(
    model: SphereSignalReconstructor,
    dataset: SphereSignalDataset,
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
    reference_points: int,
    seed: int,
) -> dict:
    def train_step(m, _step):
        obs_coords, obs_values, query_coords, query_targets, _ = dataset.sample_batch(
            "train", batch_size=batch_size, n_points=train_points,
            sampling_mode=train_sampling_mode, device=device,
        )
        preds = m(obs_coords, obs_values, query_coords).unsqueeze(-1)
        loss = F.mse_loss(preds, query_targets)
        with torch.no_grad():
            rel = calculate_l2_relative_error(preds.squeeze(-1), query_targets.squeeze(-1)).item()
        return loss, {"rel_l2": rel}

    def eval_fn(m):
        summary = evaluate_reconstructor(
            m, dataset, split="val", device=device, point_counts=[train_points],
            sampling_modes=val_sampling_modes, n_resamples=1,
            reference_points=reference_points, batch_size=batch_size, max_objects=val_objects,
        )
        return summary["aggregate"]["avg_nonuniform_rmse"][str(train_points)]

    return train_loop(
        model, run_name=run_name, device=device, steps=steps, lr=lr,
        weight_decay=weight_decay, grad_clip=grad_clip, eval_every=eval_every,
        seed=seed, train_step_fn=train_step, eval_fn=eval_fn, higher_is_better=False,
    )


def _evaluate_batch_metrics(
    dataset: SphereSignalDataset,
    preds_std: torch.Tensor,
    targets_std: torch.Tensor,
    ref_preds_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds_raw = dataset.destandardize_values(preds_std).cpu()
    targets_raw = dataset.destandardize_values(targets_std).cpu()
    error = preds_raw - targets_raw
    rmse = error.square().mean(dim=1).sqrt()
    rel_l2 = torch.norm(error, dim=1) / torch.norm(targets_raw, dim=1).clamp_min(1e-8)
    drift = torch.norm(preds_raw - ref_preds_raw, dim=1) / torch.norm(ref_preds_raw, dim=1).clamp_min(1e-8)
    pred_coeffs = dataset.project_spectral_coeffs(preds_raw)
    target_coeffs = dataset.project_spectral_coeffs(targets_raw)
    spectral = torch.norm(pred_coeffs - target_coeffs, dim=1) / torch.norm(target_coeffs, dim=1).clamp_min(1e-8)
    return rmse, rel_l2, drift, spectral


def _aggregate_metrics(
    per_setting: dict[str, dict[str, dict[str, float]]],
    point_counts: list[int],
    sampling_modes: list[str],
) -> dict:
    rmse_by_count = {}
    relative_l2_by_count = {}
    prediction_drift_by_count = {}
    spectral_error_by_count = {}
    worst_case_rmse = {}
    for n_points in point_counts:
        key = str(n_points)
        rmse_by_count[key] = {mode: per_setting[mode][key]["rmse"] for mode in sampling_modes}
        relative_l2_by_count[key] = {mode: per_setting[mode][key]["relative_l2"] for mode in sampling_modes}
        prediction_drift_by_count[key] = {mode: per_setting[mode][key]["prediction_drift"] for mode in sampling_modes}
        spectral_error_by_count[key] = {mode: per_setting[mode][key]["spectral_error"] for mode in sampling_modes}
        worst_case_rmse[key] = max(rmse_by_count[key].values())

    nonuniform_modes = [mode for mode in sampling_modes if mode != "uniform"]
    if nonuniform_modes:
        avg_nonuniform_rmse = {
            str(n_points): sum(per_setting[mode][str(n_points)]["rmse"] for mode in nonuniform_modes) / len(nonuniform_modes)
            for n_points in point_counts
        }
        avg_nonuniform_prediction_drift = {
            str(n_points): sum(per_setting[mode][str(n_points)]["prediction_drift"] for mode in nonuniform_modes)
            / len(nonuniform_modes)
            for n_points in point_counts
        }
        avg_nonuniform_spectral_error = {
            str(n_points): sum(per_setting[mode][str(n_points)]["spectral_error"] for mode in nonuniform_modes)
            / len(nonuniform_modes)
            for n_points in point_counts
        }
    else:
        avg_nonuniform_rmse = {str(n_points): per_setting["uniform"][str(n_points)]["rmse"] for n_points in point_counts}
        avg_nonuniform_prediction_drift = {
            str(n_points): per_setting["uniform"][str(n_points)]["prediction_drift"] for n_points in point_counts
        }
        avg_nonuniform_spectral_error = {
            str(n_points): per_setting["uniform"][str(n_points)]["spectral_error"] for n_points in point_counts
        }

    return {
        "rmse_by_count": rmse_by_count,
        "relative_l2_by_count": relative_l2_by_count,
        "prediction_drift_by_count": prediction_drift_by_count,
        "spectral_error_by_count": spectral_error_by_count,
        "worst_case_rmse": worst_case_rmse,
        "avg_nonuniform_rmse": avg_nonuniform_rmse,
        "avg_nonuniform_prediction_drift": avg_nonuniform_prediction_drift,
        "avg_nonuniform_spectral_error": avg_nonuniform_spectral_error,
    }


def evaluate_reconstructor(
    model: SphereSignalReconstructor,
    dataset: SphereSignalDataset,
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
    model.eval()
    n_objects = dataset.split_size(split)
    if max_objects is not None:
        n_objects = min(n_objects, int(max_objects))

    reference_points = min(int(reference_points), dataset.query_points)
    local_indices = torch.arange(n_objects, dtype=torch.long)
    query_coords_batch = dataset.get_query_coords_batch(batch_size, device=device)
    targets_std_all = dataset.get_query_targets(split, local_indices, standardized=True).cpu()
    reference_predictions = []

    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            idx = local_indices[start : start + batch_size]
            ref_seeds = _make_view_seeds(split, idx, n_points=reference_points, sampling_mode="uniform", replica_idx=0)
            obs_coords, obs_values = dataset.collate_observations(
                split,
                idx,
                n_points=reference_points,
                sampling_mode="uniform",
                view_seeds=ref_seeds,
                deterministic_uniform=True,
                standardized=True,
                device=device,
            )
            preds_std = model(obs_coords, obs_values, query_coords_batch[: len(idx)])
            reference_predictions.append(dataset.destandardize_values(preds_std).cpu())

    ref_preds_all = torch.cat(reference_predictions, dim=0)
    per_setting: dict[str, dict[str, dict[str, float]]] = {}

    for mode in sampling_modes:
        per_setting[mode] = {}
        for n_points in point_counts:
            totals = {"rmse": 0.0, "relative_l2": 0.0, "prediction_drift": 0.0, "spectral_error": 0.0, "count": 0}
            with torch.no_grad():
                for replica_idx in range(n_resamples):
                    for start in range(0, n_objects, batch_size):
                        idx = local_indices[start : start + batch_size]
                        view_seeds = _make_view_seeds(
                            split,
                            idx,
                            n_points=n_points,
                            sampling_mode=mode,
                            replica_idx=replica_idx + 1,
                        )
                        obs_coords, obs_values = dataset.collate_observations(
                            split,
                            idx,
                            n_points=n_points,
                            sampling_mode=mode,
                            view_seeds=view_seeds,
                            deterministic_uniform=False,
                            standardized=True,
                            device=device,
                        )
                        preds_std = model(obs_coords, obs_values, query_coords_batch[: len(idx)])
                        rmse, rel_l2, drift, spectral = _evaluate_batch_metrics(
                            dataset,
                            preds_std.cpu(),
                            targets_std_all[start : start + len(idx)],
                            ref_preds_all[start : start + len(idx)],
                        )
                        totals["rmse"] += float(rmse.sum().item())
                        totals["relative_l2"] += float(rel_l2.sum().item())
                        totals["prediction_drift"] += float(drift.sum().item())
                        totals["spectral_error"] += float(spectral.sum().item())
                        totals["count"] += len(idx)

            denom = float(totals["count"])
            per_setting[mode][str(n_points)] = {
                "rmse": totals["rmse"] / denom,
                "relative_l2": totals["relative_l2"] / denom,
                "prediction_drift": totals["prediction_drift"] / denom,
                "spectral_error": totals["spectral_error"] / denom,
            }

    return {"per_setting": per_setting, "aggregate": _aggregate_metrics(per_setting, point_counts, sampling_modes)}


def evaluate_deterministic_convergence(
    model: SphereSignalReconstructor,
    dataset: SphereSignalDataset,
    *,
    split: str,
    device: torch.device,
    point_counts: list[int],
    reference_points: int,
    batch_size: int,
    max_objects: int | None = None,
) -> dict:
    model.eval()
    n_objects = dataset.split_size(split)
    if max_objects is not None:
        n_objects = min(n_objects, int(max_objects))

    reference_points = min(int(reference_points), dataset.query_points)
    local_indices = torch.arange(n_objects, dtype=torch.long)
    query_coords_batch = dataset.get_query_coords_batch(batch_size, device=device)
    targets_std_all = dataset.get_query_targets(split, local_indices, standardized=True).cpu()
    ref_predictions = []

    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            idx = local_indices[start : start + batch_size]
            ref_seeds = _make_view_seeds(split, idx, n_points=reference_points, sampling_mode="uniform", replica_idx=0)
            obs_coords, obs_values = dataset.collate_observations(
                split,
                idx,
                n_points=reference_points,
                sampling_mode="uniform",
                view_seeds=ref_seeds,
                deterministic_uniform=True,
                standardized=True,
                device=device,
            )
            preds_std = model(obs_coords, obs_values, query_coords_batch[: len(idx)])
            ref_predictions.append(dataset.destandardize_values(preds_std).cpu())

    ref_preds_all = torch.cat(ref_predictions, dim=0)
    per_count = {}

    with torch.no_grad():
        for n_points in point_counts:
            totals = {"rmse": 0.0, "relative_l2": 0.0, "prediction_drift": 0.0, "spectral_error": 0.0, "count": 0}
            for start in range(0, n_objects, batch_size):
                idx = local_indices[start : start + batch_size]
                seeds = _make_view_seeds(split, idx, n_points=n_points, sampling_mode="uniform", replica_idx=1)
                obs_coords, obs_values = dataset.collate_observations(
                    split,
                    idx,
                    n_points=n_points,
                    sampling_mode="uniform",
                    view_seeds=seeds,
                    deterministic_uniform=True,
                    standardized=True,
                    device=device,
                )
                preds_std = model(obs_coords, obs_values, query_coords_batch[: len(idx)])
                rmse, rel_l2, drift, spectral = _evaluate_batch_metrics(
                    dataset,
                    preds_std.cpu(),
                    targets_std_all[start : start + len(idx)],
                    ref_preds_all[start : start + len(idx)],
                )
                totals["rmse"] += float(rmse.sum().item())
                totals["relative_l2"] += float(rel_l2.sum().item())
                totals["prediction_drift"] += float(drift.sum().item())
                totals["spectral_error"] += float(spectral.sum().item())
                totals["count"] += len(idx)

            denom = float(totals["count"])
            per_count[str(n_points)] = {
                "rmse": totals["rmse"] / denom,
                "relative_l2": totals["relative_l2"] / denom,
                "prediction_drift": totals["prediction_drift"] / denom,
                "spectral_error": totals["spectral_error"] / denom,
            }

    return {"per_count": per_count}


def save_training_artifacts(
    output_dir: Path,
    *,
    model: SphereSignalReconstructor,
    dataset: SphereSignalDataset,
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
        normalization=dataset.get_normalization_stats(),
    )


def load_model_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[SphereSignalReconstructor, dict]:
    cfg = load_json(checkpoint_dir / "experiment_config.json")
    model = build_model_from_config(cfg["model"]).to(device)
    state_dict = torch.load(checkpoint_dir / "model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg
