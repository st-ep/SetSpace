#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors

from case_studies.point_cloud_consistency.benchmark import load_model_checkpoint
from case_studies.point_cloud_consistency.common import load_json
from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset

MODEL_ORDER = ("uniform", "geometry_aware", "moment2")
MODEL_LABELS = {
    "uniform": "Uniform encoder",
    "geometry_aware": "kNN density encoder",
    "moment2": "MMQ-2 encoder",
}
MODE_OFFSETS = {
    "uniform": 11,
    "polar": 23,
    "equatorial": 31,
    "clustered": 47,
    "hemisphere": 59,
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


def _view_seed(local_index: int, n_points: int, mode: str) -> int:
    return int(local_index) * 65_537 + 307 + MODE_OFFSETS[mode] * 997 + int(n_points)


def _train_points_from_metrics(metrics: dict) -> int | None:
    training = metrics.get("training")
    if isinstance(training, dict) and "train_points" in training:
        return int(training["train_points"])

    model_payload = next(iter(metrics["models"].values()), None)
    if model_payload is None:
        return None
    checkpoint_dir = Path(model_payload["checkpoint_dir"])
    cfg = load_json(checkpoint_dir / "experiment_config.json")
    return int(cfg["training"]["train_points"])


def _load_models(metrics: dict, device: torch.device) -> dict[str, torch.nn.Module]:
    loaded = {}
    for model_name in [name for name in MODEL_ORDER if name in metrics["models"]]:
        checkpoint_dir = Path(metrics["models"][model_name]["checkpoint_dir"])
        model, _ = load_model_checkpoint(checkpoint_dir, device)
        loaded[model_name] = model
    return loaded


def _predict_summary(model, coords: torch.Tensor, values: torch.Tensor, label: int) -> tuple[np.ndarray, int, float]:
    with torch.no_grad():
        logits = model(coords, values)
        probs = logits.softmax(dim=-1)[0]
        return logits[0].cpu().numpy(), int(probs.argmax().item()), float(probs[int(label)].item())


def _leave_one_out_attribution(
    model,
    coords: torch.Tensor,
    values: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    n_points = coords.shape[1]
    with torch.no_grad():
        full_logits = model(coords, values)
        target_logit = full_logits[0, int(target_class)]

        masks = torch.ones((n_points, n_points), device=coords.device, dtype=torch.bool)
        masks.fill_diagonal_(False)
        masked_logits = model(
            coords.expand(n_points, -1, -1),
            values.expand(n_points, -1, -1),
            point_mask=masks,
        )
        attributions = target_logit - masked_logits[:, int(target_class)]
    return attributions.cpu().numpy()


def _field_variation(dataset: SyntheticSurfaceSignalDataset, split: str, local_index: int) -> float:
    points, values, _ = dataset.sample_view(
        split,
        local_index,
        n_points=512,
        sampling_mode="uniform",
        view_seed=_view_seed(local_index, 512, "uniform"),
    )
    del points
    return float(values.std().item())


def _score_example(
    dataset: SyntheticSurfaceSignalDataset,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    split: str,
    local_index: int,
    fixed_points: int,
    sampling_modes: list[str],
) -> tuple[int, int, float, float]:
    comparison_model = "moment2" if "moment2" in models else "geometry_aware"
    gain_modes = 0
    correct_gap = 0
    margin_gain = 0.0

    for mode in sampling_modes:
        coords, values, label = dataset.sample_view(
            split,
            local_index,
            n_points=fixed_points,
            sampling_mode=mode,
            view_seed=_view_seed(local_index, fixed_points, mode),
        )
        coords = coords.unsqueeze(0).to(device)
        values = values.unsqueeze(0).to(device)

        _, pred_uniform, p_true_uniform = _predict_summary(models["uniform"], coords, values, label)
        _, pred_cmp, p_true_cmp = _predict_summary(models[comparison_model], coords, values, label)
        gain_modes += int(pred_cmp == label and pred_uniform != label)
        correct_gap += int(pred_cmp == label) - int(pred_uniform == label)
        margin_gain += p_true_cmp - p_true_uniform

    return gain_modes, correct_gap, margin_gain, _field_variation(dataset, split, local_index)


def _select_example_index(
    dataset: SyntheticSurfaceSignalDataset,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    split: str,
    fixed_points: int,
    sampling_modes: list[str],
    candidate_count: int,
) -> int:
    limit = min(dataset.split_size(split), int(candidate_count))
    best_index = 0
    best_score = None
    for local_index in range(limit):
        score = _score_example(dataset, models, device, split, local_index, fixed_points, sampling_modes)
        if best_score is None or score > best_score:
            best_index = local_index
            best_score = score
    return best_index


def _scatter_panel(ax, points: np.ndarray, values: np.ndarray, *, cmap: str, norm, marker_size: float) -> None:
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=values,
        cmap=cmap,
        norm=norm,
        s=marker_size,
        alpha=0.95,
        linewidths=0.0,
    )
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=35)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def plot_qualitative_responses(
    metrics: dict,
    output_dir: Path,
    *,
    dataset: SyntheticSurfaceSignalDataset | None = None,
    models: dict[str, torch.nn.Module] | None = None,
    device: torch.device | None = None,
    fixed_points: int | None = None,
    split: str = "test",
    example_index: int | None = None,
    candidate_count: int = 64,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(None if device is None else str(device))
    if dataset is None:
        dataset = SyntheticSurfaceSignalDataset(**metrics["dataset"])
    if models is None:
        models = _load_models(metrics, device)

    for model in models.values():
        model.eval()

    point_counts = [int(v) for v in metrics["point_counts"]]
    sampling_modes = list(metrics["sampling_modes"])
    if fixed_points is None:
        fixed_points = min(point_counts)
    else:
        fixed_points = int(fixed_points)
    if fixed_points not in point_counts:
        fixed_points = min(point_counts)

    train_points = _train_points_from_metrics(metrics)
    if example_index is None:
        example_index = _select_example_index(
            dataset,
            models,
            device,
            split,
            fixed_points,
            sampling_modes,
            candidate_count,
        )

    object_data = dataset.objects[dataset._global_index(split, example_index)]
    model_order = [name for name in MODEL_ORDER if name in models]
    sampled_views = {}
    field_values = []
    attribution_values = []
    model_summaries: dict[str, dict[str, tuple[int, float]]] = {name: {} for name in model_order}

    for mode in sampling_modes:
        coords, values, label = dataset.sample_view(
            split,
            example_index,
            n_points=fixed_points,
            sampling_mode=mode,
            view_seed=_view_seed(example_index, fixed_points, mode),
        )
        coords_b = coords.unsqueeze(0).to(device)
        values_b = values.unsqueeze(0).to(device)
        view_payload = {
            "points": coords.cpu().numpy(),
            "field": values.squeeze(-1).cpu().numpy(),
            "label": int(label),
        }
        for model_name in model_order:
            view_payload[model_name] = {}
        field_values.append(view_payload["field"])

        for model_name in model_order:
            attribution = _leave_one_out_attribution(models[model_name], coords_b, values_b, label)
            _, pred_class, p_true = _predict_summary(models[model_name], coords_b, values_b, label)
            view_payload[model_name]["attribution"] = attribution
            view_payload[model_name]["pred_class"] = pred_class
            view_payload[model_name]["p_true"] = p_true
            model_summaries[model_name][mode] = (pred_class, p_true)
            attribution_values.append(attribution)

        sampled_views[mode] = view_payload

    field_scale = float(np.quantile(np.abs(np.concatenate(field_values)), 0.98))
    attr_scale = float(np.quantile(np.abs(np.concatenate(attribution_values)), 0.98))
    field_norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-field_scale, vmax=field_scale)
    attr_norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-attr_scale, vmax=attr_scale)
    cmap = "coolwarm"

    n_rows = 1 + len(model_order)
    fig = plt.figure(figsize=(16.8, 3.0 + 2.05 * n_rows))
    gs = fig.add_gridspec(n_rows, len(sampling_modes), hspace=0.18, wspace=0.02)

    row_labels = ["Ground-truth field"] + [f"{MODEL_LABELS[name]}\nleave-one-out attribution" for name in model_order]
    row_names = [None] + model_order
    marker_size = max(10.0, 1100.0 / float(fixed_points))
    annotation_specs: list[tuple[plt.Axes, str]] = []

    for row in range(n_rows):
        for col, mode in enumerate(sampling_modes):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            view_payload = sampled_views[mode]
            if row == 0:
                title = f"{mode}\n$M={fixed_points}$"
                values_to_plot = view_payload["field"]
                norm = field_norm
            else:
                model_name = row_names[row]
                pred_class, p_true = model_summaries[model_name][mode]
                title = f"pred={pred_class}, $p_{{true}}$={p_true:.2f}"
                values_to_plot = view_payload[model_name]["attribution"]
                norm = attr_norm

            _scatter_panel(
                ax,
                view_payload["points"],
                values_to_plot,
                cmap=cmap,
                norm=norm,
                marker_size=marker_size,
            )
            if row == 0:
                ax.set_title(title, fontsize=11, pad=6)
            else:
                annotation_specs.append((ax, title))

    for ax, title in annotation_specs:
        bbox = ax.get_position()
        fig.text(
            0.5 * (bbox.x0 + bbox.x1),
            bbox.y1 + 0.008,
            title,
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

    y_positions = np.linspace(0.82, 0.16, n_rows)
    for row_idx, row_label in enumerate(row_labels):
        fig.text(0.015, float(y_positions[row_idx]), row_label, rotation=90, va="center", ha="center", fontsize=12)

    field_cax = fig.add_axes([0.92, 0.68, 0.015, 0.20])
    field_sm = plt.cm.ScalarMappable(norm=field_norm, cmap=cmap)
    field_sm.set_array([])
    field_cb = fig.colorbar(field_sm, cax=field_cax)
    field_cb.set_label("Field value")

    attr_cax = fig.add_axes([0.92, 0.16, 0.015, 0.40])
    attr_sm = plt.cm.ScalarMappable(norm=attr_norm, cmap=cmap)
    attr_sm.set_array([])
    attr_cb = fig.colorbar(attr_sm, cax=attr_cax)
    attr_cb.set_label("Attribution to true-class logit")

    header = (
        f"Qualitative benchmark view: same test object under density shift | "
        f"label={sampled_views[sampling_modes[0]]['label']}, continuum avg={object_data.integral_estimate:+.2f}"
    )
    if train_points is not None:
        header += f", trained at $M={train_points}$"
    fig.suptitle(header, y=0.97, fontsize=15)
    fig.text(
        0.02,
        0.02,
        "Each column is the same object resampled under a different shift mode. "
        "Model rows show leave-one-out point attribution for the true class.",
        fontsize=10,
    )

    fig.savefig(output_dir / "point_cloud_consistency_qualitative.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "point_cloud_consistency_qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot qualitative point-cloud consistency responses.")
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fixed_points", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--example_index", type=int, default=None)
    parser.add_argument("--candidate_count", type=int, default=64)
    args = parser.parse_args()

    metrics = load_metrics(Path(args.metrics_path))
    plot_qualitative_responses(
        metrics,
        Path(args.output_dir),
        device=_resolve_device(args.device),
        fixed_points=args.fixed_points,
        split=args.split,
        example_index=args.example_index,
        candidate_count=args.candidate_count,
    )


if __name__ == "__main__":
    main()
