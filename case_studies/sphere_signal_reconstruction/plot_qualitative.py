#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from case_studies.sphere_signal_reconstruction.benchmark import load_model_checkpoint
from case_studies.sphere_signal_reconstruction.dataset import SphereSignalDataset


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


def _make_map_grid(n_lon: int = 120, n_lat: int = 60) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon = torch.linspace(-math.pi, math.pi, int(n_lon))
    lat = torch.linspace(-0.5 * math.pi, 0.5 * math.pi, int(n_lat))
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")
    x = torch.cos(lat_grid) * torch.cos(lon_grid)
    y = torch.cos(lat_grid) * torch.sin(lon_grid)
    z = torch.sin(lat_grid)
    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    return (
        points,
        lon_grid.cpu().numpy(),
        lat_grid.cpu().numpy(),
        x.cpu().numpy(),
        y.cpu().numpy(),
        z.cpu().numpy(),
    )


def _spherical_coords(points: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    lon = torch.atan2(points[:, 1], points[:, 0]) * 180.0 / math.pi
    lat = torch.asin(points[:, 2].clamp(-1.0, 1.0)) * 180.0 / math.pi
    return lon.cpu().numpy(), lat.cpu().numpy()


def _render_surface(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, values: np.ndarray, *, cmap, norm, title: str) -> None:
    facecolors = cmap(norm(values))
    ax.plot_surface(x, y, z, facecolors=facecolors, linewidth=0.0, antialiased=False, shade=False)
    ax.set_title(title, pad=6)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=35)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def _render_scatter_3d(ax, points: torch.Tensor, values: torch.Tensor, *, cmap, norm, title: str) -> None:
    points_np = points.cpu().numpy()
    values_np = values.cpu().numpy()
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=values_np, cmap=cmap, norm=norm, s=18.0, linewidths=0.0)
    ax.set_title(title, pad=6)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=35)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def _imshow_map(ax, values: np.ndarray, *, title: str, cmap, norm) -> None:
    ax.imshow(
        values,
        extent=(-180.0, 180.0, -90.0, 90.0),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def _scatter_map(ax, points: torch.Tensor, values: torch.Tensor, *, title: str, cmap, norm) -> None:
    lon, lat = _spherical_coords(points)
    ax.scatter(lon, lat, c=values.cpu().numpy(), cmap=cmap, norm=norm, s=18.0, linewidths=0.0)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_title(title, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])


def _choose_shift_mode(metrics: dict, fixed_points: int) -> str:
    rmse_by_mode = metrics["models"]["uniform"]["metrics"]["aggregate"]["rmse_by_count"][str(fixed_points)]
    nonuniform_modes = [mode for mode in metrics["sampling_modes"] if mode != "uniform"]
    return max(nonuniform_modes, key=lambda mode: rmse_by_mode[mode])


def _view_seed(local_index: int, *, n_points: int, sampling_mode: str, replica_idx: int = 1) -> int:
    mode_offset = {"uniform": 11, "polar": 23, "equatorial": 31, "clustered": 47, "hemisphere": 59}[sampling_mode]
    return int(local_index) * 65_537 + 307 + mode_offset * 997 + replica_idx * 7_919 + int(n_points)


def _select_example_index(
    dataset: SphereSignalDataset,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    *,
    fixed_points: int,
    sampling_mode: str,
    candidate_count: int = 32,
) -> int:
    query_coords = dataset.get_query_coords(device=device).unsqueeze(0)
    best_index = 0
    best_gain = -float("inf")
    for local_index in range(min(candidate_count, dataset.split_size("test"))):
        seeds = torch.tensor(
            [_view_seed(local_index, n_points=fixed_points, sampling_mode=sampling_mode, replica_idx=1)],
            dtype=torch.long,
        )
        obs_coords, obs_values = dataset.collate_observations(
            "test",
            torch.tensor([local_index], dtype=torch.long),
            n_points=fixed_points,
            sampling_mode=sampling_mode,
            view_seeds=seeds,
            standardized=True,
            device=device,
        )
        target = dataset.get_query_targets("test", torch.tensor([local_index]), standardized=True, device=device)
        with torch.no_grad():
            pred_uniform = models["uniform"](obs_coords, obs_values, query_coords)
            pred_geom = models["geometry_aware"](obs_coords, obs_values, query_coords)
        target_raw = dataset.destandardize_values(target).cpu()
        err_uniform = (dataset.destandardize_values(pred_uniform).cpu() - target_raw).square().mean().sqrt().item()
        err_geom = (dataset.destandardize_values(pred_geom).cpu() - target_raw).square().mean().sqrt().item()
        gain = err_uniform - err_geom
        if gain > best_gain:
            best_gain = gain
            best_index = local_index
    return best_index


def plot_qualitative(
    metrics: dict,
    output_dir: Path,
    *,
    dataset: SphereSignalDataset | None = None,
    models: dict[str, torch.nn.Module] | None = None,
    device: torch.device | None = None,
    fixed_points: int | None = None,
    reference_points: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(None if device is None else str(device))
    if dataset is None:
        dataset = SphereSignalDataset(**metrics["dataset"])
    if models is None:
        models = {}
        for model_name in ["uniform", "geometry_aware"]:
            model, _ = load_model_checkpoint(Path(metrics["models"][model_name]["checkpoint_dir"]), device)
            models[model_name] = model

    point_counts = [int(v) for v in metrics["point_counts"]]
    fixed_points = min(point_counts) if fixed_points is None else int(fixed_points)
    reference_points = max(point_counts) if reference_points is None else min(int(reference_points), dataset.query_points)
    shift_mode = _choose_shift_mode(metrics, fixed_points)
    example_index = _select_example_index(dataset, models, device, fixed_points=fixed_points, sampling_mode=shift_mode)

    query_grid, lon_grid, lat_grid, x_grid, y_grid, z_grid = _make_map_grid()
    query_grid_device = query_grid.to(device).unsqueeze(0)
    fixed_seed = torch.tensor(
        [_view_seed(example_index, n_points=fixed_points, sampling_mode=shift_mode, replica_idx=1)],
        dtype=torch.long,
    )
    ref_seed = torch.tensor(
        [_view_seed(example_index, n_points=reference_points, sampling_mode="uniform", replica_idx=0)],
        dtype=torch.long,
    )

    obs_coords, obs_values = dataset.collate_observations(
        "test",
        torch.tensor([example_index], dtype=torch.long),
        n_points=fixed_points,
        sampling_mode=shift_mode,
        view_seeds=fixed_seed,
        standardized=True,
        device=device,
    )
    ref_coords, ref_values = dataset.collate_observations(
        "test",
        torch.tensor([example_index], dtype=torch.long),
        n_points=reference_points,
        sampling_mode="uniform",
        view_seeds=ref_seed,
        deterministic_uniform=True,
        standardized=True,
        device=device,
    )

    target_grid_raw = dataset.evaluate_split_object_raw("test", example_index, query_grid).reshape(lat_grid.shape).cpu().numpy()
    with torch.no_grad():
        pred_uniform = dataset.destandardize_values(models["uniform"](obs_coords, obs_values, query_grid_device)).cpu().reshape(lat_grid.shape).numpy()
        pred_geom = dataset.destandardize_values(models["geometry_aware"](obs_coords, obs_values, query_grid_device)).cpu().reshape(lat_grid.shape).numpy()
    err_uniform = np.abs(pred_uniform - target_grid_raw)
    err_geom = np.abs(pred_geom - target_grid_raw)

    scale = float(np.quantile(np.abs(target_grid_raw), 0.98))
    err_scale = float(np.quantile(np.concatenate([err_uniform.reshape(-1), err_geom.reshape(-1)]), 0.98))
    signal_norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale)
    error_norm = colors.Normalize(vmin=0.0, vmax=max(err_scale, 1e-6))
    signal_cmap = plt.get_cmap("coolwarm")
    error_cmap = plt.get_cmap("magma")

    fig = plt.figure(figsize=(16.5, 10.5))
    gs = fig.add_gridspec(3, 4, hspace=0.18, wspace=0.06)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    _render_surface(ax, x_grid, y_grid, z_grid, target_grid_raw, cmap=signal_cmap, norm=signal_norm, title="Ground truth")
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    _render_scatter_3d(
        ax,
        obs_coords[0].cpu(),
        dataset.destandardize_values(obs_values[0].squeeze(-1)).cpu(),
        cmap=signal_cmap,
        norm=signal_norm,
        title=f"Shifted input ({shift_mode}, $M$={fixed_points})",
    )
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    _render_surface(ax, x_grid, y_grid, z_grid, pred_uniform, cmap=signal_cmap, norm=signal_norm, title="Uniform prediction")
    ax = fig.add_subplot(gs[0, 3], projection="3d")
    _render_surface(ax, x_grid, y_grid, z_grid, pred_geom, cmap=signal_cmap, norm=signal_norm, title="Geometry-aware prediction")

    ax = fig.add_subplot(gs[1, 0])
    _imshow_map(ax, target_grid_raw, title="Ground truth map", cmap=signal_cmap, norm=signal_norm)
    ax = fig.add_subplot(gs[1, 1])
    _scatter_map(
        ax,
        obs_coords[0].cpu(),
        dataset.destandardize_values(obs_values[0].squeeze(-1)).cpu(),
        title="Shifted input map",
        cmap=signal_cmap,
        norm=signal_norm,
    )
    ax = fig.add_subplot(gs[1, 2])
    _imshow_map(ax, pred_uniform, title="Uniform map", cmap=signal_cmap, norm=signal_norm)
    ax = fig.add_subplot(gs[1, 3])
    _imshow_map(ax, pred_geom, title="Geometry-aware map", cmap=signal_cmap, norm=signal_norm)

    ax = fig.add_subplot(gs[2, 0], projection="3d")
    _render_scatter_3d(
        ax,
        ref_coords[0].cpu(),
        dataset.destandardize_values(ref_values[0].squeeze(-1)).cpu(),
        cmap=signal_cmap,
        norm=signal_norm,
        title=f"Dense reference input ($M$={reference_points})",
    )
    ax = fig.add_subplot(gs[2, 1])
    _scatter_map(
        ax,
        ref_coords[0].cpu(),
        dataset.destandardize_values(ref_values[0].squeeze(-1)).cpu(),
        title="Dense reference map",
        cmap=signal_cmap,
        norm=signal_norm,
    )
    ax = fig.add_subplot(gs[2, 2])
    _imshow_map(ax, err_uniform, title="Uniform absolute error", cmap=error_cmap, norm=error_norm)
    ax = fig.add_subplot(gs[2, 3])
    _imshow_map(ax, err_geom, title="Geometry-aware absolute error", cmap=error_cmap, norm=error_norm)

    signal_cax = fig.add_axes([0.92, 0.56, 0.015, 0.28])
    signal_sm = plt.cm.ScalarMappable(norm=signal_norm, cmap=signal_cmap)
    signal_sm.set_array([])
    signal_cb = fig.colorbar(signal_sm, cax=signal_cax)
    signal_cb.set_label("Field value")

    error_cax = fig.add_axes([0.92, 0.14, 0.015, 0.22])
    error_sm = plt.cm.ScalarMappable(norm=error_norm, cmap=error_cmap)
    error_sm.set_array([])
    error_cb = fig.colorbar(error_sm, cax=error_cax)
    error_cb.set_label("Absolute error")

    fig.suptitle(
        "Sphere signal reconstruction under discretization shift: one object, sparse shifted input, dense field prediction",
        y=0.97,
        fontsize=15,
    )
    fig.text(
        0.02,
        0.02,
        f"Held-out object {example_index}; shifted input uses mode='{shift_mode}' at M={fixed_points}; "
        f"dense reference uses deterministic uniform observations at M={reference_points}.",
        fontsize=10,
    )
    fig.savefig(output_dir / "sphere_signal_reconstruction_qualitative.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "sphere_signal_reconstruction_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot qualitative sphere reconstructions.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fixed_points", type=int, default=None)
    parser.add_argument("--reference_points", type=int, default=None)
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_qualitative(
        metrics,
        Path(args.output_dir),
        device=_resolve_device(args.device),
        fixed_points=args.fixed_points,
        reference_points=args.reference_points,
    )


if __name__ == "__main__":
    main()
