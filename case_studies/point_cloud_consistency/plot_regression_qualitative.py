#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from case_studies.point_cloud_consistency.benchmark import _make_view_seeds, load_model_checkpoint
from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset

MODEL_STYLES = {
    "uniform": {"label": "Uniform", "color": "#d95f02"},
    "geometry_aware": {"label": "kNN", "color": "#1b9e77"},
    "pointnext": {"label": "PointNeXt", "color": "#7570b3"},
}


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _default_sampling_mode(metrics: dict, point_key: str) -> str:
    sampling_modes = [mode for mode in metrics["sampling_modes"] if mode != "uniform"]
    if not sampling_modes:
        return "uniform"
    uniform_by_mode = metrics["models"]["uniform"]["metrics"]["aggregate"]["rmse_by_count"][point_key]
    geometry_by_mode = metrics["models"]["geometry_aware"]["metrics"]["aggregate"]["rmse_by_count"][point_key]
    return max(sampling_modes, key=lambda mode: uniform_by_mode[mode] - geometry_by_mode[mode])


def _load_optional_pointnext_bundle(
    *,
    output_dir: Path,
    device: torch.device,
    pointnext_metrics_path: str | Path | None,
) -> tuple[dict | None, SyntheticSurfaceSignalDataset | None, torch.nn.Module | None]:
    candidates: list[Path] = []
    if pointnext_metrics_path is not None:
        candidates.append(Path(pointnext_metrics_path))
    else:
        candidates.append(output_dir.parent / "point_cloud_mean_regression_pointnext_run" / "metrics.json")

    for path in candidates:
        if not path.exists():
            continue
        metrics = load_metrics(path)
        if "pointnext" not in metrics.get("models", {}):
            continue
        dataset = SyntheticSurfaceSignalDataset(**metrics["dataset"])
        model, _ = load_model_checkpoint(Path(metrics["models"]["pointnext"]["checkpoint_dir"]), device)
        return metrics, dataset, model
    return None, None, None


def _sample_view_seed(split: str, local_index: int, *, n_points: int, sampling_mode: str, replica_idx: int) -> int:
    seeds = _make_view_seeds(
        split,
        torch.tensor([int(local_index)], dtype=torch.long),
        n_points=int(n_points),
        sampling_mode=sampling_mode,
        replica_idx=int(replica_idx),
    )
    return int(seeds[0].item())


def _predict_split_targets(
    model: torch.nn.Module,
    dataset: SyntheticSurfaceSignalDataset,
    *,
    split: str,
    device: torch.device,
    n_points: int,
    sampling_mode: str,
    batch_size: int = 64,
    replica_idx: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n_objects = dataset.split_size(split)
    local_indices = torch.arange(n_objects, dtype=torch.long)
    targets = dataset.get_integral_targets(split, local_indices).cpu().numpy()
    preds_list: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            idx = local_indices[start : start + batch_size]
            view_seeds = _make_view_seeds(
                split,
                idx,
                n_points=n_points,
                sampling_mode=sampling_mode,
                replica_idx=replica_idx,
            )
            coords, values, _ = dataset.collate_views(
                split,
                idx,
                n_points=n_points,
                sampling_mode=sampling_mode,
                view_seeds=view_seeds,
                device=device,
            )
            preds_list.append(model(coords, values).detach().cpu())

    preds = torch.cat(preds_list, dim=0).numpy()
    return preds, targets


def _select_example_index(
    dataset: SyntheticSurfaceSignalDataset,
    uniform_model: torch.nn.Module,
    geometry_model: torch.nn.Module,
    *,
    device: torch.device,
    n_points: int,
    sampling_mode: str,
    candidate_count: int = 48,
) -> int:
    candidate_count = min(int(candidate_count), dataset.split_size("test"))
    best_index = 0
    best_score = -float("inf")

    for local_index in range(candidate_count):
        target = float(dataset.get_integral_targets("test", torch.tensor([local_index]))[0].item())
        sampled_points, sampled_values, _ = dataset.sample_view(
            "test",
            local_index,
            n_points=n_points,
            sampling_mode=sampling_mode,
            view_seed=_sample_view_seed("test", local_index, n_points=n_points, sampling_mode=sampling_mode, replica_idx=1),
        )
        coords = sampled_points.unsqueeze(0).to(device)
        values = sampled_values.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_uniform = float(uniform_model(coords, values).cpu().item())
            pred_geometry = float(geometry_model(coords, values).cpu().item())
        improvement = abs(pred_uniform - target) - abs(pred_geometry - target)
        field_std = float(sampled_values.squeeze(-1).std().item())
        score = improvement + 0.05 * field_std
        if score > best_score:
            best_score = score
            best_index = local_index

    return best_index


def _render_surface_with_samples(
    ax,
    dense_points: np.ndarray,
    dense_values: np.ndarray,
    observed_points: np.ndarray,
    *,
    cmap,
    norm,
    title: str,
) -> None:
    spans = np.ptp(dense_points, axis=0)
    spans = np.where(spans > 0.0, spans, 1.0)
    ax.scatter(
        dense_points[:, 0],
        dense_points[:, 1],
        dense_points[:, 2],
        c=dense_values,
        cmap=cmap,
        norm=norm,
        s=5.0,
        linewidths=0.0,
        alpha=0.92,
    )
    ax.scatter(
        observed_points[:, 0],
        observed_points[:, 1],
        observed_points[:, 2],
        c="black",
        s=12.0,
        linewidths=0.0,
        alpha=0.45,
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect(tuple(float(v) for v in spans))
    ax.view_init(elev=18, azim=35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()


def _render_prediction_scatter(ax, targets: np.ndarray, series: list[tuple[str, np.ndarray]]) -> None:
    all_series = [targets, *[preds for _, preds in series]]
    lo = float(min(arr.min() for arr in all_series))
    hi = float(max(arr.max() for arr in all_series))
    pad = 0.06 * (hi - lo + 1e-8)
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1.2, alpha=0.8)
    rmse_lines = []
    for model_name, preds in series:
        style = MODEL_STYLES[model_name]
        ax.scatter(
            targets,
            preds,
            s=24,
            c=style["color"],
            alpha=0.75,
            linewidths=0.0,
            label=style["label"],
        )
        rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        rmse_lines.append(f"{style['label']} RMSE: {rmse:.3f}")
    ax.text(
        0.04,
        0.95,
        "\n".join(rmse_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.0},
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("True continuum average")
    ax.set_ylabel("Predicted continuum average")
    ax.set_title("Prediction scatter")


def _render_residual_scatter(ax, targets: np.ndarray, series: list[tuple[str, np.ndarray]]) -> None:
    residuals = [preds - targets for _, preds in series]
    y_scale = float(max(np.max(np.abs(res)) for res in residuals))
    y_scale = max(y_scale, 1e-3) * 1.1

    ax.axhline(0.0, linestyle="--", color="#555555", lw=1.2, alpha=0.8)
    mae_lines = []
    for model_name, preds in series:
        style = MODEL_STYLES[model_name]
        res = preds - targets
        ax.scatter(
            targets,
            res,
            s=24,
            c=style["color"],
            alpha=0.75,
            linewidths=0.0,
            label=style["label"],
        )
        mae = float(np.mean(np.abs(res)))
        bias = float(np.mean(res))
        mae_lines.append(f"{style['label']} MAE: {mae:.3f}, bias: {bias:+.3f}")
    ax.text(
        0.04,
        0.95,
        "\n".join(mae_lines),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2.0},
    )
    ax.set_ylim(-y_scale, y_scale)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("True continuum average")
    ax.set_ylabel("Residual (prediction - truth)")
    ax.set_title("Residuals by test object")


def plot_prediction_figure(
    metrics: dict,
    output_dir: Path,
    *,
    dataset: SyntheticSurfaceSignalDataset | None = None,
    uniform_model: torch.nn.Module | None = None,
    geometry_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    sampling_mode: str | None = None,
    observation_points: int | None = None,
    dense_points: int = 4096,
    pointnext_metrics_path: str | Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(None if device is None else str(device))
    if dataset is None:
        dataset = SyntheticSurfaceSignalDataset(**metrics["dataset"])

    checkpoint_uniform = Path(metrics["models"]["uniform"]["checkpoint_dir"])
    checkpoint_geometry = Path(metrics["models"]["geometry_aware"]["checkpoint_dir"])
    if uniform_model is None:
        uniform_model, _ = load_model_checkpoint(checkpoint_uniform, device)
    if geometry_model is None:
        geometry_model, _ = load_model_checkpoint(checkpoint_geometry, device)
    pointnext_metrics, pointnext_dataset, pointnext_model = _load_optional_pointnext_bundle(
        output_dir=output_dir,
        device=device,
        pointnext_metrics_path=pointnext_metrics_path,
    )

    point_counts = [int(v) for v in metrics["point_counts"]]
    observation_points = max(point_counts) if observation_points is None else int(observation_points)
    point_key = str(observation_points)
    if sampling_mode is None:
        sampling_mode = _default_sampling_mode(metrics, point_key)

    example_index = _select_example_index(
        dataset,
        uniform_model,
        geometry_model,
        device=device,
        n_points=observation_points,
        sampling_mode=sampling_mode,
    )

    dense_view_seed = _sample_view_seed("test", example_index, n_points=dense_points, sampling_mode="uniform", replica_idx=0)
    dense_points_tensor, dense_values_tensor, _ = dataset.sample_view(
        "test",
        example_index,
        n_points=dense_points,
        sampling_mode="uniform",
        view_seed=dense_view_seed,
    )
    observed_view_seed = _sample_view_seed(
        "test",
        example_index,
        n_points=observation_points,
        sampling_mode=sampling_mode,
        replica_idx=1,
    )
    observed_points_tensor, _, _ = dataset.sample_view(
        "test",
        example_index,
        n_points=observation_points,
        sampling_mode=sampling_mode,
        view_seed=observed_view_seed,
    )

    dense_values = dense_values_tensor.squeeze(-1).cpu().numpy()
    scale = float(np.quantile(np.abs(dense_values), 0.98))
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale)
    cmap = plt.get_cmap("coolwarm")

    preds_uniform, targets = _predict_split_targets(
        uniform_model,
        dataset,
        split="test",
        device=device,
        n_points=observation_points,
        sampling_mode=sampling_mode,
    )
    preds_geometry, _ = _predict_split_targets(
        geometry_model,
        dataset,
        split="test",
        device=device,
        n_points=observation_points,
        sampling_mode=sampling_mode,
    )
    prediction_series: list[tuple[str, np.ndarray]] = [
        ("uniform", preds_uniform),
        ("geometry_aware", preds_geometry),
    ]

    if pointnext_model is not None and pointnext_dataset is not None:
        preds_pointnext, targets_pointnext = _predict_split_targets(
            pointnext_model,
            pointnext_dataset,
            split="test",
            device=device,
            n_points=observation_points,
            sampling_mode=sampling_mode,
        )
        if targets_pointnext.shape == targets.shape and np.allclose(targets_pointnext, targets):
            prediction_series.append(("pointnext", preds_pointnext))

    true_mean = float(dataset.objects[dataset._global_index("test", example_index)].integral_estimate)

    fig = plt.figure(figsize=(14.4, 4.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.32)
    ax_surface = fig.add_subplot(gs[0, 0], projection="3d")
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_residual = fig.add_subplot(gs[0, 2])

    _render_surface_with_samples(
        ax_surface,
        dense_points_tensor.cpu().numpy(),
        dense_values,
        observed_points_tensor.cpu().numpy(),
        cmap=cmap,
        norm=norm,
        title="Ground-truth surface field",
    )
    _render_prediction_scatter(ax_scatter, targets, prediction_series)
    _render_residual_scatter(ax_residual, targets, prediction_series)

    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    fig.colorbar(scalar_map, ax=ax_surface, fraction=0.046, pad=0.03, shrink=0.92)
    ax_residual.legend(frameon=False, loc="lower left")

    fig.suptitle(
        rf"Synthetic mean-regression prediction quality under {sampling_mode} sampling ($M$={observation_points}); "
        rf"surface panel shows test object {example_index} with continuum average {true_mean:+.3f}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.12, top=0.9)
    fig.savefig(output_dir / "point_cloud_mean_regression_prediction.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "point_cloud_mean_regression_prediction.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot qualitative prediction figures for the synthetic mean-regression benchmark.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sampling_mode", default=None)
    parser.add_argument("--observation_points", type=int, default=None)
    parser.add_argument("--dense_points", type=int, default=4096)
    parser.add_argument("--pointnext_metrics_path", default=None)
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_prediction_figure(
        metrics,
        Path(args.output_dir),
        device=_resolve_device(args.device),
        sampling_mode=args.sampling_mode,
        observation_points=args.observation_points,
        dense_points=args.dense_points,
        pointnext_metrics_path=args.pointnext_metrics_path,
    )


if __name__ == "__main__":
    main()
