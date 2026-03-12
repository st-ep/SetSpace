#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CLR_GEOM = "#1b9e77"
CLR_UNIF = "#d95f02"


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _power_law_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    return float(slope), float(intercept)


def _fit_curve(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return np.exp(intercept) * np.power(x, slope)


def _plot_series_with_fit(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
) -> float:
    slope, intercept = _power_law_fit(x, y)
    ax.loglog(x, y, marker + "-", color=color, lw=2.2, ms=6, label=label)
    ax.loglog(x, _fit_curve(x, slope, intercept), ":", color=color, lw=1.2, alpha=0.8)
    return slope


def plot_metrics(metrics: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    point_counts = np.array([int(v) for v in metrics["point_counts"]], dtype=float)
    uniform_agg = metrics["models"]["uniform"]["metrics"]["aggregate"]
    geom_agg = metrics["models"]["geometry_aware"]["metrics"]["aggregate"]

    uniform_rmse = np.array([uniform_agg["avg_nonuniform_rmse"][str(int(p))] for p in point_counts], dtype=float)
    geom_rmse = np.array([geom_agg["avg_nonuniform_rmse"][str(int(p))] for p in point_counts], dtype=float)
    uniform_drift = np.array(
        [uniform_agg["avg_nonuniform_prediction_drift"][str(int(p))] for p in point_counts],
        dtype=float,
    )
    geom_drift = np.array(
        [geom_agg["avg_nonuniform_prediction_drift"][str(int(p))] for p in point_counts],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.set_box_aspect(1)

    ax = axes[0]
    slope_unif_rmse = _plot_series_with_fit(
        ax,
        point_counts,
        uniform_rmse,
        color=CLR_UNIF,
        marker="s",
        label="Uniform encoder",
    )
    slope_geom_rmse = _plot_series_with_fit(
        ax,
        point_counts,
        geom_rmse,
        color=CLR_GEOM,
        marker="o",
        label="Geometry-aware encoder",
    )
    ax.set_xlabel("$M$ (number of sampled points)")
    ax.set_ylabel("Average nonuniform RMSE")
    ax.set_title("(a) Mean-regression error under refinement")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.04,
        0.08,
        f"Uniform fit: $M^{{{slope_unif_rmse:.2f}}}$\nGeometry-aware fit: $M^{{{slope_geom_rmse:.2f}}}$",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )

    ax = axes[1]
    slope_unif_drift = _plot_series_with_fit(
        ax,
        point_counts,
        uniform_drift,
        color=CLR_UNIF,
        marker="s",
        label="Uniform encoder",
    )
    slope_geom_drift = _plot_series_with_fit(
        ax,
        point_counts,
        geom_drift,
        color=CLR_GEOM,
        marker="o",
        label="Geometry-aware encoder",
    )
    ax.set_xlabel("$M$ (number of sampled points)")
    ax.set_ylabel("Average nonuniform prediction drift")
    ax.set_title("(b) Same-object prediction drift under refinement")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.04,
        0.08,
        f"Uniform fit: $M^{{{slope_unif_drift:.2f}}}$\nGeometry-aware fit: $M^{{{slope_geom_drift:.2f}}}$",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
    )

    fig.tight_layout(w_pad=2.0)
    fig.savefig(output_dir / "point_cloud_mean_regression_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "point_cloud_mean_regression_convergence.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot log-log convergence for the point-cloud mean-regression benchmark.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_metrics(metrics, Path(args.output_dir))


if __name__ == "__main__":
    main()
