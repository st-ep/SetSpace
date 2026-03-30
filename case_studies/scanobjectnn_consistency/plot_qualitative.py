#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from case_studies.scanobjectnn_consistency.benchmark import load_model_checkpoint
from case_studies.scanobjectnn_consistency.dataset import SAMPLING_MODES, ScanObjectNNConsistencyDataset


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


def _view_seed(local_index: int, *, n_points: int, sampling_mode: str, replica_idx: int = 1) -> int:
    mode_offset = {
        "uniform_object": 11,
        "clustered_object": 23,
        "occluded_object": 31,
        "background_heavy": 47,
    }[sampling_mode]
    return int(local_index) * 65_537 + 307 + mode_offset * 997 + replica_idx * 7_919 + int(n_points)


def _predict(models: dict[str, torch.nn.Module], coords: torch.Tensor, values: torch.Tensor, dataset: ScanObjectNNConsistencyDataset) -> dict[str, dict]:
    out = {}
    with torch.no_grad():
        for name, model in models.items():
            logits = model(coords, values)
            probs = torch.softmax(logits, dim=-1)
            pred = int(probs.argmax(dim=-1).item())
            conf = float(probs.max(dim=-1).values.item())
            out[name] = {
                "pred": pred,
                "label_name": dataset.get_label_name(pred),
                "confidence": conf,
            }
    return out


def _score_candidate(
    dataset: ScanObjectNNConsistencyDataset,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    *,
    local_index: int,
    fixed_points: int,
    nonuniform_modes: list[str],
) -> float:
    comparison_model = "geometry_aware"
    label = dataset._record("test", local_index).label
    score = 0.0
    for mode in nonuniform_modes:
        seed = torch.tensor([_view_seed(local_index, n_points=fixed_points, sampling_mode=mode, replica_idx=1)], dtype=torch.long)
        coords, values, _labels = dataset.collate_views(
            "test",
            torch.tensor([local_index], dtype=torch.long),
            n_points=fixed_points,
            sampling_mode=mode,
            view_seeds=seed,
            device=device,
        )
        preds = _predict(models, coords, values, dataset)
        uniform_correct = int(preds["uniform"]["pred"] == label)
        cmp_correct = int(preds[comparison_model]["pred"] == label)
        true_conf_gap = (preds[comparison_model]["confidence"] if cmp_correct else 0.0) - (
            preds["uniform"]["confidence"] if uniform_correct else 0.0
        )
        score += 2.0 * (cmp_correct - uniform_correct) + true_conf_gap
    return score


def _select_example_index(
    dataset: ScanObjectNNConsistencyDataset,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    *,
    fixed_points: int,
    candidate_count: int = 64,
) -> int:
    nonuniform_modes = [mode for mode in SAMPLING_MODES if mode != "uniform_object"]
    best_index = 0
    best_score = -float("inf")
    for local_index in range(min(candidate_count, dataset.split_size("test"))):
        score = _score_candidate(
            dataset,
            models,
            device,
            local_index=local_index,
            fixed_points=fixed_points,
            nonuniform_modes=nonuniform_modes,
        )
        if score > best_score:
            best_score = score
            best_index = local_index
    return best_index


def _scatter_view(ax, coords: torch.Tensor, point_is_foreground: torch.Tensor, title: str, depth: torch.Tensor | None = None) -> None:
    coords_np = coords.cpu().numpy()
    foreground = point_is_foreground.cpu().numpy().astype(bool)
    ax.scatter(coords_np[~foreground, 0], coords_np[~foreground, 1], coords_np[~foreground, 2], c="#9e9e9e", s=12, alpha=0.75)
    if foreground.any():
        if depth is not None:
            colors = depth.cpu().numpy()
            ax.scatter(
                coords_np[foreground, 0],
                coords_np[foreground, 1],
                coords_np[foreground, 2],
                c=colors[foreground],
                cmap="viridis",
                s=18,
                alpha=0.95,
            )
        else:
            ax.scatter(coords_np[foreground, 0], coords_np[foreground, 1], coords_np[foreground, 2], c="#1f77b4", s=18, alpha=0.95)
    ax.set_title(title, pad=8)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.grid(False)


def plot_qualitative(
    metrics: dict,
    output_dir: Path,
    *,
    dataset: ScanObjectNNConsistencyDataset | None = None,
    models: dict[str, torch.nn.Module] | None = None,
    device: torch.device | None = None,
    fixed_points: int | None = None,
    reference_points: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(None if device is None else str(device))
    if dataset is None:
        dataset = ScanObjectNNConsistencyDataset(**metrics["dataset"])
    model_order = [name for name in ["uniform", "geometry_aware"] if name in metrics["models"]]
    if models is None:
        models = {}
        for model_name in model_order:
            model, _ = load_model_checkpoint(Path(metrics["models"][model_name]["checkpoint_dir"]), device)
            models[model_name] = model

    point_counts = [int(v) for v in metrics["point_counts"]]
    fixed_points = min(point_counts) if fixed_points is None else int(fixed_points)
    reference_points = int(metrics.get("reference_points", max(point_counts))) if reference_points is None else int(reference_points)
    example_index = _select_example_index(dataset, models, device, fixed_points=fixed_points)
    true_label = dataset._record("test", example_index).label
    true_name = dataset.get_label_name(true_label)

    columns = ["uniform_object", "clustered_object", "occluded_object", "background_heavy"]
    titles = {
        "uniform_object": f"Reference view\n(uniform_object, M={reference_points})",
        "clustered_object": f"Clustered object\n(M={fixed_points})",
        "occluded_object": f"Occluded object\n(M={fixed_points})",
        "background_heavy": f"Background-heavy\n(M={fixed_points})",
    }

    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(2, len(columns), height_ratios=[4.0, 1.4], hspace=0.15, wspace=0.08)
    legend_handles = []
    legend_labels = []

    for col, mode in enumerate(columns):
        n_points = reference_points if mode == "uniform_object" else fixed_points
        replica = 0 if mode == "uniform_object" else 1
        seed = torch.tensor([_view_seed(example_index, n_points=n_points, sampling_mode=mode, replica_idx=replica)], dtype=torch.long)
        coords, values, _labels, metadata_list = dataset.collate_views(
            "test",
            torch.tensor([example_index], dtype=torch.long),
            n_points=n_points,
            sampling_mode=mode,
            view_seeds=seed,
            device=device,
            return_metadata=True,
        )
        metadata = metadata_list[0]
        predictions = _predict(models, coords, values, dataset)

        ax = fig.add_subplot(gs[0, col], projection="3d")
        depth = None
        if mode == "occluded_object" and "occlusion_direction" in metadata:
            direction = metadata["occlusion_direction"]
            depth = coords[0].cpu() @ direction.cpu()
        _scatter_view(ax, coords[0].cpu(), metadata["point_is_foreground"].cpu(), titles[mode], depth=depth)

        text_ax = fig.add_subplot(gs[1, col])
        text_ax.axis("off")
        text_ax.text(0.0, 0.92, f"True label: {true_name}", fontsize=11, fontweight="bold")
        line_y = 0.66
        style = {
            "uniform": ("Uniform", "#d95f02"),
            "geometry_aware": ("kNN density", "#1b9e77"),
        }
        for model_name in model_order:
            label_name, color = style[model_name]
            text_ax.text(
                0.0,
                line_y,
                f"{label_name}: {predictions[model_name]['label_name']} ({predictions[model_name]['confidence']:.2f})",
                fontsize=10.5,
                color=color,
            )
            line_y -= 0.22
        if mode == "background_heavy":
            text_ax.text(
                0.0,
                0.02,
                f"Background fallback points: {metadata['empty_background_points']}",
                fontsize=9.5,
                color="#616161",
            )

    fig.suptitle(
        "ScanObjectNN discretization-consistency benchmark: one held-out object under same-object resampling",
        y=0.98,
        fontsize=15,
    )
    fig.text(
        0.02,
        0.02,
        "Blue points are object points. Gray points are sampled background points. "
        "The occluded view colors object points by visibility depth.",
        fontsize=10,
    )
    fig.savefig(output_dir / "scanobjectnn_consistency_qualitative.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "scanobjectnn_consistency_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot qualitative ScanObjectNN consistency views.")
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
