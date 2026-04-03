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

from case_studies.ahmedml_surface_forces.benchmark import _make_view_seeds, load_model_checkpoint
from case_studies.ahmedml_surface_forces.dataset import (
    AhmedMLSurfaceForceDataset,
    _normalized_weights,
    _sample_indices,
    _sampling_bias,
)

TARGET_AXES = {"Cd": 0, "Cs": 1, "Cl": 2}
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
    rmse_by_mode = metrics["models"]["uniform"]["metrics"]["aggregate"]["rmse_by_count"][point_key]
    nonuniform_modes = [mode for mode in metrics["sampling_modes"] if mode != "uniform"]
    if not nonuniform_modes:
        return "uniform"
    return max(nonuniform_modes, key=lambda mode: rmse_by_mode[mode])


def _sample_indices_for_view(
    dataset: AhmedMLSurfaceForceDataset,
    split: str,
    local_index: int,
    *,
    n_points: int,
    sampling_mode: str,
    replica_idx: int,
) -> np.ndarray:
    probs = _sampling_probabilities_for_view(
        dataset,
        split,
        local_index,
        sampling_mode=sampling_mode,
        n_points=n_points,
        replica_idx=replica_idx,
    )
    seed = int(
        _make_view_seeds(
            split,
            torch.tensor([int(local_index)], dtype=torch.long),
            n_points=n_points,
            sampling_mode=sampling_mode,
            replica_idx=replica_idx,
        )[0].item()
    )
    rng = np.random.default_rng(seed)
    return np.asarray(_sample_indices(probs, n_points=n_points, rng=rng), dtype=np.int64)


def _sampling_probabilities_for_view(
    dataset: AhmedMLSurfaceForceDataset,
    split: str,
    local_index: int,
    *,
    sampling_mode: str,
    n_points: int,
    replica_idx: int,
) -> np.ndarray:
    sample = dataset.samples[split][int(local_index)]
    seed = int(
        _make_view_seeds(
            split,
            torch.tensor([int(local_index)], dtype=torch.long),
            n_points=n_points,
            sampling_mode=sampling_mode,
            replica_idx=replica_idx,
        )[0].item()
    )
    rng = np.random.default_rng(seed)
    return _normalized_weights(sample["base_weights"] * _sampling_bias(sample["coords_unit"], sampling_mode, rng))


def _subsample_display_indices(base_weights: np.ndarray, *, n_display: int, seed: int) -> np.ndarray:
    if base_weights.shape[0] <= n_display:
        return np.arange(base_weights.shape[0], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    probs = _normalized_weights(base_weights)
    idx = rng.choice(base_weights.shape[0], size=int(n_display), replace=False, p=probs)
    return np.sort(idx.astype(np.int64))


def _ground_truth_contribution_field(
    values: np.ndarray,
    base_weights: np.ndarray,
    *,
    target_name: str,
    target_value: float,
) -> np.ndarray:
    axis = TARGET_AXES.get(target_name)
    if axis is None:
        raise ValueError(f"Unsupported target for qualitative contribution map: {target_name}")
    pressure = values[:, 0]
    normals = values[:, 1:4]
    if values.shape[1] >= 7:
        shear = values[:, 4:7]
    else:
        shear = np.zeros((values.shape[0], 3), dtype=np.float32)
    raw = base_weights * (-pressure * normals[:, axis] + shear[:, axis])
    total = float(raw.sum())
    if abs(total) > 1e-8:
        raw = raw * (float(target_value) / total)
    return raw.astype(np.float32)


def _render_cloud(ax, coords: np.ndarray, scalars: np.ndarray, *, cmap, norm, title: str) -> None:
    spans = np.ptp(coords, axis=0)
    spans = np.where(spans > 0.0, spans, 1.0)
    point_size = 5.0 if coords.shape[0] <= 8000 else 3.0
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=scalars,
        cmap=cmap,
        norm=norm,
        s=point_size,
        linewidths=0.0,
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect(tuple(float(v) for v in spans))
    ax.view_init(elev=18, azim=-63)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()


def _render_observed_pressure(
    ax,
    background_coords: np.ndarray,
    observed_coords: np.ndarray,
    observed_pressure: np.ndarray,
    *,
    cmap,
    norm,
    title: str,
) -> None:
    spans = np.ptp(background_coords, axis=0)
    spans = np.where(spans > 0.0, spans, 1.0)
    ax.scatter(
        background_coords[:, 0],
        background_coords[:, 1],
        background_coords[:, 2],
        c="#d7d7d7",
        s=1.6,
        linewidths=0.0,
        alpha=0.2,
    )
    ax.scatter(
        observed_coords[:, 0],
        observed_coords[:, 1],
        observed_coords[:, 2],
        c=observed_pressure,
        cmap=cmap,
        norm=norm,
        s=9.0,
        linewidths=0.0,
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect(tuple(float(v) for v in spans))
    ax.view_init(elev=18, azim=-63)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()


def _render_surface_with_samples(
    ax,
    background_coords: np.ndarray,
    surface_field: np.ndarray,
    observed_coords: np.ndarray,
    *,
    cmap,
    norm,
    title: str,
) -> None:
    spans = np.ptp(background_coords, axis=0)
    spans = np.where(spans > 0.0, spans, 1.0)
    ax.scatter(
        background_coords[:, 0],
        background_coords[:, 1],
        background_coords[:, 2],
        c=surface_field,
        cmap=cmap,
        norm=norm,
        s=3.0,
        linewidths=0.0,
        alpha=0.95,
    )
    ax.scatter(
        observed_coords[:, 0],
        observed_coords[:, 1],
        observed_coords[:, 2],
        c="black",
        s=10.0,
        linewidths=0.0,
        alpha=0.45,
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect(tuple(float(v) for v in spans))
    ax.view_init(elev=18, azim=-63)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()


def _predict_split_targets(
    model: torch.nn.Module,
    dataset: AhmedMLSurfaceForceDataset,
    *,
    split: str,
    device: torch.device,
    n_points: int,
    sampling_mode: str,
    batch_size: int = 32,
    replica_idx: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n_objects = dataset.split_size(split)
    local_indices = torch.arange(n_objects, dtype=torch.long)
    targets = dataset.get_targets(split, local_indices, normalized=False).cpu()
    preds_list = []
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
            coords, values = dataset.collate_views(
                split,
                idx,
                n_points=n_points,
                sampling_mode=sampling_mode,
                view_seeds=view_seeds,
                device=device,
            )
            preds = dataset.denormalize_targets(model(coords, values)).cpu()
            preds_list.append(preds)
    preds = torch.cat(preds_list, dim=0)
    return preds.numpy(), targets.numpy()


def _render_prediction_scatter(
    ax,
    targets: np.ndarray,
    series: list[tuple[str, np.ndarray]],
    *,
    target_name: str,
) -> None:
    all_series = [targets, *[preds for _, preds in series]]
    lo = float(min(arr.min() for arr in all_series))
    hi = float(max(arr.max() for arr in all_series))
    pad = 0.05 * (hi - lo + 1e-8)
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
    ax.set_xlabel(f"True {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"{target_name} prediction")


def _load_optional_pointnext_bundle(
    *,
    output_dir: Path,
    device: torch.device,
    pointnext_metrics_path: str | Path | None,
) -> tuple[dict | None, AhmedMLSurfaceForceDataset | None, torch.nn.Module | None]:
    candidates: list[Path] = []
    if pointnext_metrics_path is not None:
        candidates.append(Path(pointnext_metrics_path))
    else:
        candidates.append(output_dir.parent / "ahmedml_surface_forces_pointnext_run" / "metrics.json")
    for path in candidates:
        if not path.exists():
            continue
        metrics = load_metrics(path)
        if "pointnext" not in metrics.get("models", {}):
            continue
        dataset = AhmedMLSurfaceForceDataset(**metrics["dataset"])
        model, _ = load_model_checkpoint(Path(metrics["models"]["pointnext"]["checkpoint_dir"]), device)
        return metrics, dataset, model
    return None, None, None


def _select_example_index(
    dataset: AhmedMLSurfaceForceDataset,
    uniform_model: torch.nn.Module,
    geometry_model: torch.nn.Module,
    *,
    device: torch.device,
    n_points: int,
    sampling_mode: str,
    target_index: int,
    candidate_count: int = 32,
) -> int:
    candidate_count = min(int(candidate_count), dataset.split_size("test"))
    targets = dataset.get_targets("test", torch.arange(candidate_count), normalized=False).cpu().numpy()
    best_index = 0
    best_gain = -float("inf")
    for local_index in range(candidate_count):
        sampled_idx = _sample_indices_for_view(
            dataset,
            "test",
            local_index,
            n_points=n_points,
            sampling_mode=sampling_mode,
            replica_idx=1,
        )
        sample = dataset.samples["test"][local_index]
        coords = torch.from_numpy(sample["coords"][sampled_idx]).unsqueeze(0).to(device)
        values = torch.from_numpy(sample["values"][sampled_idx]).unsqueeze(0).to(device)
        target_value = float(targets[local_index, target_index])
        with torch.no_grad():
            pred_uniform = float(
                dataset.denormalize_targets(uniform_model(coords, values)).cpu().numpy()[0, target_index]
            )
            pred_geometry = float(
                dataset.denormalize_targets(geometry_model(coords, values)).cpu().numpy()[0, target_index]
            )
        gain = abs(pred_uniform - target_value) - abs(pred_geometry - target_value)
        if gain > best_gain:
            best_gain = gain
            best_index = local_index
    return best_index


def plot_prediction_figure(
    metrics: dict,
    output_dir: Path,
    *,
    dataset: AhmedMLSurfaceForceDataset | None = None,
    uniform_model: torch.nn.Module | None = None,
    geometry_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    target_name: str = "Cd",
    sampling_mode: str | None = None,
    observation_points: int | None = None,
    display_points: int = 6000,
    pointnext_metrics_path: str | Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(None if device is None else str(device))
    if dataset is None:
        dataset = AhmedMLSurfaceForceDataset(**metrics["dataset"])

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

    target_names = list(dataset.target_names)
    if target_name not in target_names:
        raise ValueError(f"{target_name=} not found in dataset target names {target_names}")
    target_index = target_names.index(target_name)

    point_counts = [int(v) for v in metrics["point_counts"]]
    observation_points = max(point_counts) if observation_points is None else int(observation_points)
    point_key = str(observation_points)
    if sampling_mode is None:
        sampling_mode = "front" if "front" in metrics["sampling_modes"] else _default_sampling_mode(metrics, point_key)

    example_index = _select_example_index(
        dataset,
        uniform_model,
        geometry_model,
        device=device,
        n_points=observation_points,
        sampling_mode=sampling_mode,
        target_index=target_index,
    )

    sample = dataset.samples["test"][example_index]
    observed_idx = _sample_indices_for_view(
        dataset,
        "test",
        example_index,
        n_points=observation_points,
        sampling_mode=sampling_mode,
        replica_idx=1,
    )
    display_idx = _subsample_display_indices(
        sample["base_weights"],
        n_display=int(display_points),
        seed=10_000 + int(example_index) * 37 + int(observation_points),
    )

    obs_coords = sample["coords"][observed_idx]
    obs_values = sample["values"][observed_idx]
    display_coords = sample["coords"][display_idx]
    display_values = sample["values"][display_idx]
    display_base_weights = sample["base_weights"][display_idx]
    target_value = float(sample["targets"][target_index])
    pressure_display = display_values[:, 0].astype(np.float32)
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
    prediction_series_cd: list[tuple[str, np.ndarray]] = [
        ("uniform", preds_uniform[:, 0]),
        ("geometry_aware", preds_geometry[:, 0]),
    ]
    prediction_series_cl: list[tuple[str, np.ndarray]] = [
        ("uniform", preds_uniform[:, 1]),
        ("geometry_aware", preds_geometry[:, 1]),
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
            prediction_series_cd.append(("pointnext", preds_pointnext[:, 0]))
            prediction_series_cl.append(("pointnext", preds_pointnext[:, 1]))

    pressure_scale = float(np.quantile(np.abs(pressure_display), 0.99))
    gt_norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-pressure_scale, vmax=pressure_scale)
    gt_cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(13.8, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.32)
    ax_surface = fig.add_subplot(gs[0, 0], projection="3d")
    ax_cd = fig.add_subplot(gs[0, 1])
    ax_cl = fig.add_subplot(gs[0, 2])

    _render_surface_with_samples(
        ax_surface,
        display_coords,
        pressure_display,
        obs_coords,
        cmap=gt_cmap,
        norm=gt_norm,
        title="Ground-truth pressure field",
    )
    _render_prediction_scatter(
        ax_cd,
        targets[:, 0],
        prediction_series_cd,
        target_name="Cd",
    )
    _render_prediction_scatter(
        ax_cl,
        targets[:, 1],
        prediction_series_cl,
        target_name="Cl",
    )

    gt_map = plt.cm.ScalarMappable(norm=gt_norm, cmap=gt_cmap)
    gt_map.set_array([])
    fig.colorbar(gt_map, ax=ax_surface, fraction=0.046, pad=0.03, shrink=0.92)
    ax_cl.legend(frameon=False, loc="lower right")

    fig.suptitle(
        rf"AhmedML prediction quality under {sampling_mode} sampling ($M$={observation_points}); "
        rf"surface panel shows test object {example_index} with {target_name}={target_value:.3f}",
        y=0.98,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.12, top=0.9)
    fig.savefig(output_dir / "ahmedml_surface_forces_prediction.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ahmedml_surface_forces_prediction.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot AhmedML prediction-quality figures.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--target_name", default="Cd")
    parser.add_argument("--sampling_mode", default=None)
    parser.add_argument("--observation_points", type=int, default=None)
    parser.add_argument("--display_points", type=int, default=6000)
    parser.add_argument("--pointnext_metrics_path", default=None)
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_prediction_figure(
        metrics,
        Path(args.output_dir),
        device=_resolve_device(args.device),
        target_name=args.target_name,
        sampling_mode=args.sampling_mode,
        observation_points=args.observation_points,
        display_points=args.display_points,
        pointnext_metrics_path=args.pointnext_metrics_path,
    )


if __name__ == "__main__":
    main()
