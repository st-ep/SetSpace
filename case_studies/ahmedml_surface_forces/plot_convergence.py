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

COLORS = {"uniform": "#d95f02", "geometry_aware": "#1b9e77", "pointnext": "#7570b3"}
LABELS = {"uniform": "Uniform encoder", "geometry_aware": "kNN density encoder", "pointnext": "PointNeXt"}
MARKERS = {"uniform": "s", "geometry_aware": "o", "pointnext": "^"}


def load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_optional_pointnext_metrics(output_dir: Path, explicit_path: Path | None = None) -> dict | None:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates.append(output_dir.parent / "ahmedml_surface_forces_pointnext_run" / "metrics.json")
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


def _rate_label(slope: float) -> str:
    return rf"$O(M^{{{slope:.2f}}})$"


def _plot_series(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    color: str,
    marker: str,
    label: str,
) -> tuple[float, float]:
    slope, intercept = _power_law_fit(x, y)
    fit = _fit_curve(x, slope, intercept)
    ax.loglog(x, y, marker + "-", color=color, lw=2.2, ms=6, label=label)
    ax.loglog(x, fit, ":", color=color, lw=1.3, alpha=0.75)
    return slope, intercept


def _annotate_rate(ax, x: np.ndarray, slope: float, intercept: float, *, color: str, y_scale: float, x_index: int) -> None:
    anchor_x = float(x[x_index])
    anchor_y = float(_fit_curve(np.asarray([anchor_x]), slope, intercept)[0]) * y_scale
    ax.text(anchor_x, anchor_y, _rate_label(slope), color=color, fontsize=11)


def plot_metrics(
    metrics: dict,
    output_dir: Path,
    *,
    pointnext_metrics: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pointnext_metrics = pointnext_metrics or _load_optional_pointnext_metrics(output_dir)
    point_counts = np.array([int(v) for v in metrics["point_counts"]], dtype=float)
    model_sources: list[tuple[str, dict]] = [(name, metrics) for name in ["uniform", "geometry_aware"] if name in metrics["models"]]
    if pointnext_metrics is not None:
        pointnext_counts = np.array([int(v) for v in pointnext_metrics["point_counts"]], dtype=float)
        if np.array_equal(point_counts, pointnext_counts):
            model_sources.append(("pointnext", pointnext_metrics))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))
    for ax in axes:
        ax.set_box_aspect(1)
        ax.grid(True, alpha=0.28, which="both")

    ax = axes[0]
    for model_name, payload in model_sources:
        agg = payload["models"][model_name]["metrics"]["aggregate"]
        series = np.array([agg["worst_case_rmse"][str(int(p))] for p in point_counts], dtype=float)
        slope, intercept = _plot_series(
            ax,
            point_counts,
            series,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            label=LABELS[model_name],
        )
        _annotate_rate(
            ax,
            point_counts,
            slope,
            intercept,
            color=COLORS[model_name],
            y_scale=1.12 if model_name == "uniform" else 0.82,
            x_index=-2 if len(point_counts) > 2 else -1,
        )
    ax.set_xlabel(r"$M$ (number of sampled surface cells)")
    ax.set_ylabel("Worst-case RMSE")
    ax.set_title(r"(a) AhmedML worst-case surface-force error")
    ax.set_xlim(point_counts[0] * 0.9, point_counts[-1] * 1.1)
    ax.set_xticks(point_counts)
    ax.set_xticklabels([str(int(p)) for p in point_counts], fontsize=9)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if model_sources:
        ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    for model_name, payload in model_sources:
        agg = payload["models"][model_name]["metrics"]["aggregate"]
        series = np.array([agg["avg_nonuniform_prediction_drift"][str(int(p))] for p in point_counts], dtype=float)
        slope, intercept = _plot_series(
            ax,
            point_counts,
            series,
            color=COLORS[model_name],
            marker=MARKERS[model_name],
            label=LABELS[model_name],
        )
        _annotate_rate(
            ax,
            point_counts,
            slope,
            intercept,
            color=COLORS[model_name],
            y_scale=1.18 if model_name == "uniform" else 0.7,
            x_index=-3 if len(point_counts) > 3 else -1,
        )
    ax.set_xlabel(r"$M$ (number of sampled surface cells)")
    ax.set_ylabel("Average nonuniform prediction drift")
    ax.set_title(r"(b) AhmedML same-object prediction drift")
    ax.set_xlim(point_counts[0] * 0.9, point_counts[-1] * 1.1)
    ax.set_xticks(point_counts)
    ax.set_xticklabels([str(int(p)) for p in point_counts], fontsize=9)
    ax.xaxis.set_minor_formatter(NullFormatter())
    if model_sources:
        ax.legend(frameon=False, loc="lower left")

    fig.tight_layout(w_pad=2.2)
    fig.savefig(output_dir / "ahmedml_surface_forces_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "ahmedml_surface_forces_convergence.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot AhmedML convergence curves from benchmark metrics.")
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
