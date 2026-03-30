#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"uniform": "#d95f02", "geometry_aware": "#1b9e77"}
LABELS = {"uniform": "Uniform encoder", "geometry_aware": "kNN density encoder"}
MARKERS = {"uniform": "s", "geometry_aware": "o"}


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _power_law_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(slope), float(intercept)


def _fit_curve(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return np.exp(intercept) * np.power(x, slope)


def _plot_series(ax, x: np.ndarray, y: np.ndarray, *, color: str, marker: str, label: str) -> float:
    slope, intercept = _power_law_fit(x, y)
    ax.loglog(x, y, marker + "-", color=color, lw=2.2, ms=6, label=label)
    ax.loglog(x, _fit_curve(x, slope, intercept), ":", color=color, lw=1.2, alpha=0.8)
    return slope


def plot_metrics(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    point_counts = np.array([int(v) for v in metrics["point_counts"]], dtype=float)
    model_order = [name for name in ["uniform", "geometry_aware"] if name in metrics["models"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.set_box_aspect(1)

    ax = axes[0]
    rmse_annotations = []
    for model_name in model_order:
        agg = metrics["models"][model_name]["metrics"]["aggregate"]
        series = np.array([agg["avg_nonuniform_rmse"][str(int(p))] for p in point_counts], dtype=float)
        slope = _plot_series(
            ax,
            point_counts,
            series,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            label=LABELS[model_name],
        )
        rmse_annotations.append(f"{LABELS[model_name]}: $M^{{{slope:.2f}}}$")
    ax.set_xlabel("$M$ (observed points)")
    ax.set_ylabel("RMSE")
    ax.set_title("(a) Average nonuniform reconstruction error")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.04,
        0.08,
        "\n".join(rmse_annotations),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )

    ax = axes[1]
    drift_annotations = []
    for model_name in model_order:
        agg = metrics["models"][model_name]["metrics"]["aggregate"]
        series = np.array([agg["avg_nonuniform_prediction_drift"][str(int(p))] for p in point_counts], dtype=float)
        slope = _plot_series(
            ax,
            point_counts,
            series,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            label=LABELS[model_name],
        )
        drift_annotations.append(f"{LABELS[model_name]}: $M^{{{slope:.2f}}}$")
    ax.set_xlabel("$M$ (observed points)")
    ax.set_ylabel("Average nonuniform prediction drift")
    ax.set_title("(b) Average nonuniform prediction drift")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.04,
        0.08,
        "\n".join(drift_annotations),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )

    fig.tight_layout(w_pad=2.0)
    fig.savefig(output_dir / "sphere_signal_reconstruction_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "sphere_signal_reconstruction_convergence.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot nonuniform scaling curves for sphere signal reconstruction.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_metrics(metrics, Path(args.output_dir))


if __name__ == "__main__":
    main()
