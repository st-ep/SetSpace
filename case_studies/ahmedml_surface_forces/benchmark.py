from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from case_studies.ahmedml_surface_forces.common import build_model_from_config
from case_studies.ahmedml_surface_forces.dataset import AhmedMLSurfaceForceDataset
from case_studies.shared import avg_over_nonuniform_modes, make_view_seeds
from case_studies.shared import load_model_checkpoint as _load_checkpoint
from case_studies.shared import save_training_artifacts as _save_artifacts
from case_studies.shared import train_loop

_AHMED_MODE_OFFSETS = {
    "uniform": 11,
    "front": 23,
    "rear": 31,
    "roof": 47,
    "clustered": 59,
    "occluded": 71,
}


def _make_view_seeds(split, local_indices, *, n_points, sampling_mode, replica_idx):
    return make_view_seeds(
        split,
        local_indices,
        n_points=n_points,
        sampling_mode=sampling_mode,
        replica_idx=replica_idx,
        mode_offsets=_AHMED_MODE_OFFSETS,
    )


def _accumulate_vector_metrics(
    totals: dict[str, torch.Tensor | float],
    preds: torch.Tensor,
    targets: torch.Tensor,
    ref_preds: torch.Tensor | None = None,
) -> None:
    error = preds - targets
    sq = error.square().sum(dim=0)
    abs_err = error.abs().sum(dim=0)
    totals["squared_error"] = totals.get("squared_error", torch.zeros_like(sq)) + sq
    totals["absolute_error"] = totals.get("absolute_error", torch.zeros_like(abs_err)) + abs_err
    if ref_preds is not None:
        drift = (preds - ref_preds).abs().sum(dim=0)
        totals["prediction_drift"] = totals.get("prediction_drift", torch.zeros_like(drift)) + drift


def train_regressor(
    model: torch.nn.Module,
    dataset: AhmedMLSurfaceForceDataset,
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
        coords, values, targets = dataset.sample_batch(
            "train",
            batch_size=batch_size,
            n_points=train_points,
            sampling_mode=train_sampling_mode,
            device=device,
            normalized_targets=True,
        )
        preds = m(coords, values)
        loss = F.mse_loss(preds, targets)
        with torch.no_grad():
            mae = (preds - targets).abs().mean().item()
        return loss, {"mae": mae}

    def eval_fn(m):
        summary = evaluate_regressor(
            m,
            dataset,
            split="val",
            device=device,
            point_counts=[train_points],
            sampling_modes=val_sampling_modes,
            n_resamples=1,
            reference_points=max(train_points * 4, 2048),
            batch_size=batch_size,
            max_objects=val_objects,
        )
        if any(mode != "uniform" for mode in val_sampling_modes):
            return summary["aggregate"]["avg_nonuniform_rmse"][str(train_points)]
        return summary["aggregate"]["rmse_by_count"][str(train_points)]["uniform"]

    return train_loop(
        model,
        run_name=run_name,
        device=device,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        eval_every=eval_every,
        seed=seed,
        train_step_fn=train_step,
        eval_fn=eval_fn,
        higher_is_better=False,
    )


def evaluate_regressor(
    model: torch.nn.Module,
    dataset: AhmedMLSurfaceForceDataset,
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
    local_indices = torch.arange(n_objects, dtype=torch.long)
    target_names = list(dataset.target_names)

    ref_outputs_list = []
    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            idx = local_indices[start : start + batch_size]
            ref_seeds = _make_view_seeds(
                split,
                idx,
                n_points=reference_points,
                sampling_mode="uniform",
                replica_idx=0,
            )
            coords, values = dataset.collate_views(
                split,
                idx,
                n_points=reference_points,
                sampling_mode="uniform",
                view_seeds=ref_seeds,
                device=device,
            )
            ref_outputs_list.append(dataset.denormalize_targets(model(coords, values)).cpu())
    ref_outputs = torch.cat(ref_outputs_list, dim=0)
    targets = dataset.get_targets(split, local_indices, normalized=False).cpu()

    per_setting: dict[str, dict[str, dict[str, float | dict[str, float]]]] = {}
    for mode in sampling_modes:
        per_setting[mode] = {}
        for n_points in point_counts:
            totals: dict[str, torch.Tensor | float] = {}
            count = 0
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
                        coords, values = dataset.collate_views(
                            split,
                            idx,
                            n_points=n_points,
                            sampling_mode=mode,
                            view_seeds=view_seeds,
                            device=device,
                        )
                        preds = dataset.denormalize_targets(model(coords, values)).cpu()
                        tgt = targets[start : start + len(idx)]
                        ref = ref_outputs[start : start + len(idx)]
                        _accumulate_vector_metrics(totals, preds, tgt, ref)
                        count += len(idx)

            denom = float(max(count, 1))
            target_dim = float(len(target_names))
            squared_error = totals["squared_error"] / denom
            absolute_error = totals["absolute_error"] / denom
            prediction_drift = totals["prediction_drift"] / denom
            per_setting[mode][str(n_points)] = {
                "rmse": float(torch.sqrt(squared_error.mean()).item()),
                "mae": float(absolute_error.mean().item()),
                "prediction_drift": float(prediction_drift.mean().item()),
                "per_target_rmse": {
                    name: float(torch.sqrt(squared_error[target_idx]).item())
                    for target_idx, name in enumerate(target_names)
                },
                "per_target_mae": {
                    name: float(absolute_error[target_idx].item())
                    for target_idx, name in enumerate(target_names)
                },
            }

    def _by_count(metric: str) -> dict:
        return {str(n): {m: per_setting[m][str(n)][metric] for m in sampling_modes} for n in point_counts}

    aggregate = {
        "rmse_by_count": _by_count("rmse"),
        "mae_by_count": _by_count("mae"),
        "prediction_drift_by_count": _by_count("prediction_drift"),
        "worst_case_rmse": {
            str(n): max(per_setting[m][str(n)]["rmse"] for m in sampling_modes)
            for n in point_counts
        },
        "avg_nonuniform_rmse": avg_over_nonuniform_modes(per_setting, "rmse", point_counts, sampling_modes),
        "avg_nonuniform_prediction_drift": avg_over_nonuniform_modes(
            per_setting, "prediction_drift", point_counts, sampling_modes
        ),
    }
    return {
        "per_setting": per_setting,
        "aggregate": aggregate,
        "target_names": target_names,
    }


def save_training_artifacts(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    dataset: AhmedMLSurfaceForceDataset,
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


def load_model_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    return _load_checkpoint(checkpoint_dir, device, build_model_from_config)

