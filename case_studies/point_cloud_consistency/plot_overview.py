#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_seed(*values: int) -> int:
    seed = 17
    for value in values:
        seed = (seed * 1_000_003 + int(value) * 97 + 13) % (2**31 - 1)
    return seed


def _select_example_index(
    dataset: SyntheticSurfaceSignalDataset,
    split: str,
    candidate_count: int = 32,
) -> int:
    limit = min(dataset.split_size(split), int(candidate_count))
    best_index = 0
    best_score = -float("inf")
    for local_index in range(limit):
        points, values, _ = dataset.sample_view(
            split,
            local_index,
            n_points=1024,
            sampling_mode="uniform",
            view_seed=_stable_seed(local_index, 1024, 0),
        )
        values = values.squeeze(-1)
        score = float(values.std().item())
        if score > best_score:
            best_score = score
            best_index = local_index
    return best_index


def _select_count_examples(point_counts: list[int], train_points: int) -> list[int]:
    ordered = sorted({int(v) for v in point_counts})
    examples: list[int] = []

    def add(count: int) -> None:
        if count in ordered and count not in examples:
            examples.append(count)

    add(ordered[0])
    below_train = [count for count in ordered if count < train_points]
    at_or_above_train = [count for count in ordered if count >= train_points]
    if below_train:
        add(below_train[-1])
    if at_or_above_train:
        add(at_or_above_train[0])
        add(at_or_above_train[-1])

    for count in ordered:
        if len(examples) >= 4:
            break
        add(count)

    return examples[:4]


def _scatter_surface(ax, points, values, *, title: str, cmap, norm, marker_size: float) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        cmap=cmap,
        norm=norm,
        s=marker_size,
        alpha=0.92,
        linewidths=0.0,
    )
    ax.set_title(title, pad=8)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=35)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def plot_benchmark_overview(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticSurfaceSignalDataset(**metrics["dataset"])
    point_counts = [int(v) for v in metrics["point_counts"]]
    train_points = int(metrics.get("training", {}).get("train_points", 512))
    sampling_modes = list(metrics["sampling_modes"])

    example_index = _select_example_index(dataset, split="test")
    obj = dataset.objects[dataset._global_index("test", example_index)]

    dense_points, dense_values, label = dataset.sample_view(
        "test",
        example_index,
        n_points=2048,
        sampling_mode="uniform",
        view_seed=_stable_seed(example_index, 2048, 1),
    )
    dense_values_np = dense_values.squeeze(-1).cpu().numpy()
    scale = float(np.quantile(np.abs(dense_values_np), 0.98))
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale)
    cmap = "coolwarm"

    fig = plt.figure(figsize=(16.5, 7.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.0], hspace=0.16, wspace=0.02)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    _scatter_surface(
        ax,
        dense_points.cpu().numpy(),
        dense_values_np,
        title=f"(a) Same object, dense reference\nlabel={label}, continuum avg={obj.integral_estimate:+.2f}",
        cmap=cmap,
        norm=norm,
        marker_size=6.0,
    )

    seen_counts = _select_count_examples(point_counts, train_points)

    for col, count in enumerate(seen_counts, start=1):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        points, values, _ = dataset.sample_view(
            "test",
            example_index,
            n_points=int(count),
            sampling_mode="uniform",
            view_seed=_stable_seed(example_index, count, 2),
        )
        _scatter_surface(
            ax,
            points.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
            title=(
                f"Uniform sampling\n$M={count}$"
                if count != train_points
                else f"Uniform sampling\n$M={count}$ (train)"
            ),
            cmap=cmap,
            norm=norm,
            marker_size=max(10.0, 1100.0 / float(count)),
        )

    fixed_points = min(point_counts)
    for col, mode in enumerate(sampling_modes):
        ax = fig.add_subplot(gs[1, col], projection="3d")
        points, values, _ = dataset.sample_view(
            "test",
            example_index,
            n_points=fixed_points,
            sampling_mode=mode,
            view_seed=_stable_seed(example_index, fixed_points, col + 11),
        )
        title = f"(b) Shifted views at $M={fixed_points}$\n{mode}" if col == 0 else mode
        _scatter_surface(
            ax,
            points.cpu().numpy(),
            values.squeeze(-1).cpu().numpy(),
            title=title,
            cmap=cmap,
            norm=norm,
            marker_size=max(10.0, 1100.0 / float(fixed_points)),
        )

    cax = fig.add_axes([0.92, 0.17, 0.015, 0.66])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Observed scalar value", rotation=90)

    fig.suptitle(
        "Synthetic point-cloud consistency benchmark: one continuum object, many point-set observations",
        y=0.98,
        fontsize=15,
    )
    fig.text(
        0.02,
        0.02,
        "Top row: uniform resampling across point counts. Bottom row: density-shifted resamplings of the same object.",
        fontsize=10,
    )

    fig.savefig(output_dir / "point_cloud_consistency_overview.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "point_cloud_consistency_overview.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the synthetic point-cloud consistency benchmark overview.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_benchmark_overview(metrics, Path(args.output_dir))


if __name__ == "__main__":
    main()
