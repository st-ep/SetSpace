#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_metrics(metrics: dict, output_dir: Path, fixed_points: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    point_counts = [int(v) for v in metrics["point_counts"]]
    fixed_points = min(point_counts) if fixed_points is None else int(fixed_points)
    fixed_key = str(fixed_points if fixed_points in point_counts else point_counts[0])
    sampling_modes = metrics["sampling_modes"]

    colors = {"uniform": "#d95f02", "geometry_aware": "#1b9e77", "moment2": "#7570b3"}
    labels = {"uniform": "Uniform encoder", "geometry_aware": "kNN density encoder", "moment2": "MMQ-2 encoder"}
    model_order = [name for name in ["uniform", "geometry_aware", "moment2"] if name in metrics["models"]]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    for model_name in model_order:
        agg = metrics["models"][model_name]["metrics"]["aggregate"]
        y = [agg["worst_case_accuracy"][str(p)] for p in point_counts]
        ax.plot(point_counts, y, marker="o", lw=2.2, ms=6, color=colors[model_name], label=labels[model_name])
    ax.set_xlabel("Number of observed points ($M$)")
    ax.set_ylabel("Worst-case accuracy")
    ax.set_title("(a) Worst-case accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    for model_name in model_order:
        agg = metrics["models"][model_name]["metrics"]["aggregate"]
        y = [agg["avg_nonuniform_embedding_drift"][str(p)] for p in point_counts]
        ax.plot(point_counts, y, marker="s", lw=2.2, ms=6, color=colors[model_name], label=labels[model_name])
    ax.set_xlabel("Number of observed points ($M$)")
    ax.set_ylabel("Average nonuniform embedding drift")
    ax.set_title("(b) Same-object embedding drift")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    x = np.arange(len(sampling_modes))
    width = 0.78 / max(len(model_order), 1)
    offsets = np.linspace(-0.5 * (len(model_order) - 1) * width, 0.5 * (len(model_order) - 1) * width, len(model_order))
    for offset, model_name in zip(offsets, model_order):
        scores = [
            metrics["models"][model_name]["metrics"]["aggregate"]["accuracy_by_count"][fixed_key][mode]
            for mode in sampling_modes
        ]
        ax.bar(x + offset, scores, width=width, color=colors[model_name], label=labels[model_name])
    ax.set_xticks(x)
    ax.set_xticklabels([mode.replace("_", "\n") for mode in sampling_modes])
    ax.set_ylabel("Accuracy")
    ax.set_title(f"(c) Accuracy by mode at $M$={fixed_key}")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(output_dir / "scanobjectnn_consistency.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "scanobjectnn_consistency.pdf", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ScanObjectNN consistency metrics.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--fixed_points", type=int, default=None)
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_metrics(metrics, Path(args.output_dir), fixed_points=args.fixed_points)


if __name__ == "__main__":
    main()
