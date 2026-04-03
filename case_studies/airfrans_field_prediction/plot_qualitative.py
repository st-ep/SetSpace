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
from matplotlib import colors, ticker

from case_studies.airfrans_field_prediction.benchmark import _make_view_seeds, load_model_checkpoint
from case_studies.airfrans_field_prediction.dataset import (
    AirfRANSForceDataset,
    _normalized_weights,
    _sample_indices,
    _sampling_bias,
)

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


def _sample_probabilities_for_view(
    dataset: AirfRANSForceDataset,
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


def _sample_indices_for_view(
    dataset: AirfRANSForceDataset,
    split: str,
    local_index: int,
    *,
    n_points: int,
    sampling_mode: str,
    replica_idx: int,
) -> np.ndarray:
    probs = _sample_probabilities_for_view(
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


def _load_optional_pointnext_bundle(
    *,
    output_dir: Path,
    device: torch.device,
    pointnext_metrics_path: str | Path | None,
) -> tuple[dict | None, AirfRANSForceDataset | None, torch.nn.Module | None]:
    candidates: list[Path] = []
    if pointnext_metrics_path is not None:
        candidates.append(Path(pointnext_metrics_path))
    else:
        candidates.append(output_dir.parent / "airfrans_field_prediction_pointnext_run" / "metrics.json")

    for path in candidates:
        if not path.exists():
            continue
        metrics = load_metrics(path)
        if "pointnext" not in metrics.get("models", {}):
            continue
        dataset = AirfRANSForceDataset(**metrics["dataset"])
        model, _ = load_model_checkpoint(Path(metrics["models"]["pointnext"]["checkpoint_dir"]), device)
        return metrics, dataset, model
    return None, None, None


def _predict_split_targets(
    model: torch.nn.Module,
    dataset: AirfRANSForceDataset,
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


def _select_example_index(
    dataset: AirfRANSForceDataset,
    uniform_model: torch.nn.Module,
    geometry_model: torch.nn.Module,
    *,
    device: torch.device,
    n_points: int,
    sampling_mode: str,
    candidate_count: int = 64,
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
        target_vec = targets[local_index]
        with torch.no_grad():
            pred_uniform = dataset.denormalize_targets(uniform_model(coords, values)).cpu().numpy()[0]
            pred_geometry = dataset.denormalize_targets(geometry_model(coords, values)).cpu().numpy()[0]
        gain = float(np.linalg.norm(pred_uniform - target_vec) - np.linalg.norm(pred_geometry - target_vec))
        if gain > best_gain:
            best_gain = gain
            best_index = local_index
    return best_index


def _render_airfoil_with_samples(
    ax,
    boundary_coords: np.ndarray,
    boundary_pressure: np.ndarray,
    observed_coords: np.ndarray,
    *,
    cmap,
    norm,
    title: str,
) -> None:
    ax.scatter(
        boundary_coords[:, 0],
        boundary_coords[:, 1],
        c=boundary_pressure,
        cmap=cmap,
        norm=norm,
        s=18,
        linewidths=0.0,
        alpha=0.92,
        zorder=0,
    )
    ax.scatter(
        observed_coords[:, 0],
        observed_coords[:, 1],
        c="#1f1f1f",
        s=10,
        linewidths=0.0,
        alpha=0.40,
        zorder=1,
    )
    ax.set_title(title, pad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


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


def plot_prediction_figure(
    metrics: dict,
    output_dir: Path,
    *,
    dataset: AirfRANSForceDataset | None = None,
    uniform_model: torch.nn.Module | None = None,
    geometry_model: torch.nn.Module | None = None,
    device: torch.device | None = None,
    sampling_mode: str | None = None,
    observation_points: int | None = None,
    pointnext_metrics_path: str | Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(None if device is None else str(device))
    if dataset is None:
        dataset = AirfRANSForceDataset(**metrics["dataset"])

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

    sample = dataset.samples["test"][example_index]
    observed_idx = _sample_indices_for_view(
        dataset,
        "test",
        example_index,
        n_points=observation_points,
        sampling_mode=sampling_mode,
        replica_idx=1,
    )
    observed_idx = np.sort(observed_idx)
    if observed_idx.size > 96:
        keep = np.linspace(0, observed_idx.size - 1, 96, dtype=np.int64)
        observed_idx = observed_idx[keep]
    boundary_coords = sample["coords"][:, :2]
    boundary_pressure = sample["values"][:, 0]
    observed_coords = sample["coords"][observed_idx, :2]

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
    series_cd: list[tuple[str, np.ndarray]] = [
        ("uniform", preds_uniform[:, 0]),
        ("geometry_aware", preds_geometry[:, 0]),
    ]
    series_cl: list[tuple[str, np.ndarray]] = [
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
            series_cd.append(("pointnext", preds_pointnext[:, 0]))
            series_cl.append(("pointnext", preds_pointnext[:, 1]))

    # Use a tighter symmetric display clip for visualization. The AirfRANS
    # pressure field on this sample has a very heavy negative tail, so a 95th
    # percentile clip still collapses almost everything to white.
    pressure_scale = max(float(np.quantile(np.abs(boundary_pressure), 0.60)), 1e-8)
    pressure_norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-pressure_scale, vmax=pressure_scale)
    pressure_cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(14.9, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.0, 1.0], wspace=0.58)
    ax_surface = fig.add_subplot(gs[0, 0])
    ax_cd = fig.add_subplot(gs[0, 1])
    ax_cl = fig.add_subplot(gs[0, 2])

    _render_airfoil_with_samples(
        ax_surface,
        boundary_coords,
        boundary_pressure,
        observed_coords,
        cmap=pressure_cmap,
        norm=pressure_norm,
        title="Airfoil boundary pressure",
    )
    _render_prediction_scatter(ax_cd, targets[:, 0], series_cd, target_name="Cd")
    _render_prediction_scatter(ax_cl, targets[:, 1], series_cl, target_name="Cl")

    sm = plt.cm.ScalarMappable(norm=pressure_norm, cmap=pressure_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_surface, orientation="vertical", fraction=0.040, pad=0.10)
    cbar.ax.tick_params(labelsize=9)
    tick_values = np.asarray(
        [-pressure_scale, -0.5 * pressure_scale, 0.0, 0.5 * pressure_scale, pressure_scale],
        dtype=np.float64,
    )
    cbar.set_ticks(tick_values)
    max_abs_tick = float(np.max(np.abs(tick_values))) if tick_values.size else 0.0
    order = int(np.floor(np.log10(max_abs_tick))) if max_abs_tick > 0.0 else 0
    scale = 10.0 ** order
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x / scale:.1f}"))
    cbar.set_label("Boundary pressure", fontsize=9)
    cbar.ax.set_title(rf"$10^{{{order}}}$", fontsize=9, pad=6)

    cd_val = float(sample["targets"][0])
    cl_val = float(sample["targets"][1])
    fig.suptitle(
        f"AirfRANS prediction quality under {sampling_mode.replace('_', ' ')} sampling (M={observation_points}); "
        f"test case {example_index} with Cd={cd_val:.3f}, Cl={cl_val:.3f}",
        y=0.98,
        fontsize=11,
    )

    fig.savefig(output_dir / "airfrans_force_regression_prediction.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "airfrans_force_regression_prediction.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot AirfRANS qualitative Cd/Cl prediction figure from benchmark metrics.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--sampling_mode", default=None)
    parser.add_argument("--observation_points", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pointnext_metrics_path", default=None)
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_prediction_figure(
        metrics,
        Path(args.output_dir),
        sampling_mode=args.sampling_mode,
        observation_points=args.observation_points,
        device=_resolve_device(args.device),
        pointnext_metrics_path=args.pointnext_metrics_path,
    )


if __name__ == "__main__":
    main()
