#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

COLORS = {
    "uniform": "#d95f02",
    "geometry_aware": "#1b9e77",
    "voronoi": "#1f78b4",
    "pointnext": "#7570b3",
}
LABELS = {
    "uniform": "Set-Key (Unif)",
    "geometry_aware": "Set-Key (kNN)",
    "voronoi": "Set-Value (Vor)",
    "pointnext": "PointNeXt",
}
MARKERS = {"uniform": "s", "geometry_aware": "o", "voronoi": "D", "pointnext": "^"}


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_optional_pointnext_metrics(output_dir: Path, explicit_path: Path | None = None) -> dict | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.append(output_dir.parent / "point_cloud_mean_regression_pointnext_run" / "metrics.json")
    for path in candidates:
        if not path.exists():
            continue
        payload = load_metrics(path)
        if "pointnext" in payload.get("models", {}):
            return payload
    return None


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


def plot_metrics(metrics: dict, output_dir: Path, *, pointnext_metrics: dict | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pointnext_metrics = pointnext_metrics or _load_optional_pointnext_metrics(output_dir)
    point_counts = np.array([int(v) for v in metrics["point_counts"]], dtype=float)
    model_sources: list[tuple[str, dict]] = [
        (name, metrics)
        for name in ["uniform", "geometry_aware", "voronoi"]
        if name in metrics["models"]
    ]
    if pointnext_metrics is not None:
        pointnext_counts = np.array([int(v) for v in pointnext_metrics["point_counts"]], dtype=float)
        if np.array_equal(point_counts, pointnext_counts):
            model_sources.append(("pointnext", pointnext_metrics))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.set_box_aspect(1)

    ax = axes[0]
    rmse_annotations = []
    for model_name, payload in model_sources:
        agg = payload["models"][model_name]["metrics"]["aggregate"]
        series = np.array([agg["avg_nonuniform_rmse"][str(int(p))] for p in point_counts], dtype=float)
        slope = _plot_series_with_fit(
            ax,
            point_counts,
            series,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            label=LABELS[model_name],
        )
        rmse_annotations.append(f"{LABELS[model_name]}: $M^{{{slope:.2f}}}$")
    ax.set_xlabel("$M$ (number of sampled points)")
    ax.set_ylabel("Average nonuniform RMSE")
    ax.set_title("(a) Mean-regression error under refinement")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(point_counts[0] * 0.9, point_counts[-1] * 1.1)
    ax.set_xticks(point_counts)
    ax.set_xticklabels([str(int(p)) for p in point_counts], fontsize=9)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if model_sources:
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
    for model_name, payload in model_sources:
        agg = payload["models"][model_name]["metrics"]["aggregate"]
        series = np.array([agg["avg_nonuniform_prediction_drift"][str(int(p))] for p in point_counts], dtype=float)
        slope = _plot_series_with_fit(
            ax,
            point_counts,
            series,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            label=LABELS[model_name],
        )
        drift_annotations.append(f"{LABELS[model_name]}: $M^{{{slope:.2f}}}$")
    ax.set_xlabel("$M$ (number of sampled points)")
    ax.set_ylabel("Average nonuniform prediction drift")
    ax.set_title("(b) Same-object prediction drift under refinement")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(point_counts[0] * 0.9, point_counts[-1] * 1.1)
    ax.set_xticks(point_counts)
    ax.set_xticklabels([str(int(p)) for p in point_counts], fontsize=9)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if model_sources:
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
    fig.savefig(output_dir / "point_cloud_mean_regression_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "point_cloud_mean_regression_convergence.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot log-log convergence for the point-cloud mean-regression benchmark.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--pointnext_metrics_path", default=None)
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    pointnext_metrics = (
        load_metrics(Path(args.pointnext_metrics_path))
        if args.pointnext_metrics_path is not None
        else None
    )
    plot_metrics(metrics, Path(args.output_dir), pointnext_metrics=pointnext_metrics)


if __name__ == "__main__":
    main()
