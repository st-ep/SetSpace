from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from case_studies.point_cloud_consistency.models import PointCloudSetClassifier
from case_studies.scanobjectnn_consistency.common import build_model_from_config, load_json, save_json
from case_studies.scanobjectnn_consistency.dataset import ScanObjectNNConsistencyDataset


def _make_view_seeds(
    split: str,
    local_indices: torch.Tensor,
    *,
    n_points: int,
    sampling_mode: str,
    replica_idx: int,
) -> torch.Tensor:
    mode_offset = {
        "uniform_object": 11,
        "clustered_object": 23,
        "occluded_object": 31,
        "background_heavy": 47,
    }[sampling_mode]
    split_offset = {"train": 101, "val": 211, "test": 307}[split]
    return local_indices.to(dtype=torch.long) * 65_537 + split_offset + mode_offset * 997 + replica_idx * 7_919 + int(n_points)


def train_classifier(
    model: PointCloudSetClassifier,
    dataset: ScanObjectNNConsistencyDataset,
    *,
    run_name: str | None = None,
    device: torch.device,
    train_points: int,
    train_sampling_mode: str,
    batch_size: int,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    eval_every: int,
    val_sampling_modes: list[str],
    val_objects: int,
    reference_points: int,
    seed: int,
) -> dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf")
    history = []
    ema_loss = None
    ema_acc = None
    tag = f"[{run_name}] " if run_name else ""

    bar = trange(1, steps + 1)
    for step in bar:
        model.train()
        coords, values, labels = dataset.sample_batch(
            "train",
            batch_size=batch_size,
            n_points=train_points,
            sampling_mode=train_sampling_mode,
            device=device,
        )
        logits = model(coords, values)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            train_acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            ema_loss = float(loss.item()) if ema_loss is None else 0.95 * ema_loss + 0.05 * float(loss.item())
            ema_acc = train_acc if ema_acc is None else 0.95 * ema_acc + 0.05 * train_acc

        if step % eval_every == 0 or step == steps:
            summary = evaluate_classifier(
                model,
                dataset,
                split="val",
                device=device,
                point_counts=[train_points],
                sampling_modes=val_sampling_modes,
                n_resamples=1,
                reference_points=reference_points,
                batch_size=batch_size,
                max_objects=val_objects,
            )
            val_acc = summary["aggregate"]["avg_nonuniform_accuracy"][str(train_points)]
            history.append({"step": step, "train_loss": float(loss.item()), "train_accuracy": train_acc, "val_score": val_acc})
            if val_acc > best_score:
                best_score = float(val_acc)
                best_state = copy.deepcopy(model.state_dict())

            bar.set_description(
                f"{tag}Step {step} | Batch Loss {loss.item():.4f} | Batch Acc {train_acc:.3f} | "
                f"EMA Acc {ema_acc:.3f} | Val Robust Acc {val_acc:.3f} | Grad {float(grad_norm):.2f}"
            )
        else:
            bar.set_description(
                f"{tag}Step {step} | Batch Loss {loss.item():.4f} | Batch Acc {train_acc:.3f} | EMA Acc {ema_acc:.3f}"
            )

    model.load_state_dict(best_state)
    return {"seed": seed, "best_val_score": best_score, "history_tail": history[-10:]}


def _aggregate_metrics(
    per_setting: dict[str, dict[str, dict[str, float]]],
    point_counts: list[int],
    sampling_modes: list[str],
    replacement_stats: dict[str, dict[str, dict[str, int]]],
) -> dict:
    accuracy_by_count = {}
    embedding_drift_by_count = {}
    logit_drift_by_count = {}
    prediction_consistency_by_count = {}
    worst_case_accuracy = {}
    for n_points in point_counts:
        key = str(n_points)
        accuracy_by_count[key] = {mode: per_setting[mode][key]["accuracy"] for mode in sampling_modes}
        embedding_drift_by_count[key] = {mode: per_setting[mode][key]["embedding_drift"] for mode in sampling_modes}
        logit_drift_by_count[key] = {mode: per_setting[mode][key]["logit_drift"] for mode in sampling_modes}
        prediction_consistency_by_count[key] = {
            mode: per_setting[mode][key]["prediction_consistency"] for mode in sampling_modes
        }
        worst_case_accuracy[key] = min(accuracy_by_count[key].values())

    nonuniform_modes = [mode for mode in sampling_modes if mode != "uniform_object"]
    avg_nonuniform_accuracy = {
        str(n_points): sum(per_setting[mode][str(n_points)]["accuracy"] for mode in nonuniform_modes) / len(nonuniform_modes)
        for n_points in point_counts
    }
    avg_nonuniform_embedding_drift = {
        str(n_points): sum(per_setting[mode][str(n_points)]["embedding_drift"] for mode in nonuniform_modes)
        / len(nonuniform_modes)
        for n_points in point_counts
    }
    avg_nonuniform_logit_drift = {
        str(n_points): sum(per_setting[mode][str(n_points)]["logit_drift"] for mode in nonuniform_modes)
        / len(nonuniform_modes)
        for n_points in point_counts
    }

    return {
        "accuracy_by_count": accuracy_by_count,
        "embedding_drift_by_count": embedding_drift_by_count,
        "logit_drift_by_count": logit_drift_by_count,
        "prediction_consistency_by_count": prediction_consistency_by_count,
        "worst_case_accuracy": worst_case_accuracy,
        "avg_nonuniform_accuracy": avg_nonuniform_accuracy,
        "avg_nonuniform_embedding_drift": avg_nonuniform_embedding_drift,
        "avg_nonuniform_logit_drift": avg_nonuniform_logit_drift,
        "replacement_stats": replacement_stats,
    }


def evaluate_classifier(
    model: PointCloudSetClassifier,
    dataset: ScanObjectNNConsistencyDataset,
    *,
    split: str,
    device: torch.device,
    point_counts: list[int],
    sampling_modes: list[str],
    n_resamples: int,
    reference_points: int,
    batch_size: int,
    max_objects: int | None = None,
) -> dict:
    model.eval()
    n_objects = dataset.split_size(split)
    if max_objects is not None:
        n_objects = min(n_objects, int(max_objects))

    local_indices = torch.arange(n_objects, dtype=torch.long)
    reference_embeddings = []
    reference_logits = []
    reference_preds = []

    with torch.no_grad():
        for start in range(0, n_objects, batch_size):
            idx = local_indices[start : start + batch_size]
            ref_seeds = _make_view_seeds(split, idx, n_points=reference_points, sampling_mode="uniform_object", replica_idx=0)
            coords, values, _labels = dataset.collate_views(
                split,
                idx,
                n_points=reference_points,
                sampling_mode="uniform_object",
                view_seeds=ref_seeds,
                device=device,
            )
            logits = model(coords, values)
            embeddings = model.embed(coords, values)
            reference_logits.append(logits.cpu())
            reference_embeddings.append(embeddings.cpu())
            reference_preds.append(logits.argmax(dim=-1).cpu())

    ref_logits_all = torch.cat(reference_logits, dim=0)
    ref_embeddings_all = torch.cat(reference_embeddings, dim=0)
    ref_preds_all = torch.cat(reference_preds, dim=0)

    per_setting: dict[str, dict[str, dict[str, float]]] = {}
    replacement_stats: dict[str, dict[str, dict[str, int]]] = {str(n_points): {} for n_points in point_counts}
    for mode in sampling_modes:
        per_setting[mode] = {}
        for n_points in point_counts:
            totals = {
                "accuracy": 0.0,
                "embedding_drift": 0.0,
                "logit_drift": 0.0,
                "prediction_consistency": 0.0,
                "count": 0,
            }
            replacement_totals = {
                "num_views": 0,
                "foreground_replacement_points": 0,
                "background_replacement_points": 0,
                "foreground_views_with_replacement": 0,
                "background_views_with_replacement": 0,
                "empty_background_points": 0,
            }
            with torch.no_grad():
                for replica_idx in range(n_resamples):
                    for start in range(0, n_objects, batch_size):
                        idx = local_indices[start : start + batch_size]
                        view_seeds = _make_view_seeds(
                            split,
                            idx,
                            n_points=n_points,
                            sampling_mode=mode,
                            replica_idx=replica_idx + 1,
                        )
                        coords, values, labels, metadata_list = dataset.collate_views(
                            split,
                            idx,
                            n_points=n_points,
                            sampling_mode=mode,
                            view_seeds=view_seeds,
                            device=device,
                            return_metadata=True,
                        )
                        logits = model(coords, values).cpu()
                        embeddings = model.embed(coords, values).cpu()
                        preds = logits.argmax(dim=-1)

                        ref_logits = ref_logits_all[start : start + len(idx)]
                        ref_embeddings = ref_embeddings_all[start : start + len(idx)]
                        ref_preds = ref_preds_all[start : start + len(idx)]

                        totals["accuracy"] += float((preds == labels.cpu()).float().sum().item())
                        totals["prediction_consistency"] += float((preds == ref_preds).float().sum().item())
                        totals["embedding_drift"] += float(
                            (
                                (embeddings - ref_embeddings).norm(dim=1)
                                / ref_embeddings.norm(dim=1).clamp_min(1e-8)
                            ).sum().item()
                        )
                        totals["logit_drift"] += float(
                            ((logits - ref_logits).norm(dim=1) / ref_logits.norm(dim=1).clamp_min(1e-8)).sum().item()
                        )
                        totals["count"] += len(idx)

                        for metadata in metadata_list:
                            replacement_totals["num_views"] += 1
                            replacement_totals["foreground_replacement_points"] += int(metadata["replacement_points"])
                            replacement_totals["background_replacement_points"] += int(metadata["background_replacement_points"])
                            replacement_totals["foreground_views_with_replacement"] += int(metadata["used_replacement"])
                            replacement_totals["background_views_with_replacement"] += int(metadata["background_used_replacement"])
                            replacement_totals["empty_background_points"] += int(metadata["empty_background_points"])

            denom = float(totals["count"])
            per_setting[mode][str(n_points)] = {
                "accuracy": totals["accuracy"] / denom,
                "embedding_drift": totals["embedding_drift"] / denom,
                "logit_drift": totals["logit_drift"] / denom,
                "prediction_consistency": totals["prediction_consistency"] / denom,
            }
            replacement_stats[str(n_points)][mode] = replacement_totals

    return {
        "per_setting": per_setting,
        "aggregate": _aggregate_metrics(per_setting, point_counts, sampling_modes, replacement_stats),
    }


def save_training_artifacts(
    output_dir: Path,
    *,
    model: PointCloudSetClassifier,
    dataset: ScanObjectNNConsistencyDataset,
    model_config: dict,
    training_config: dict,
    training_summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    save_json(
        output_dir / "experiment_config.json",
        {
            "dataset": dataset.get_config(),
            "dataset_metadata": dataset.get_metadata(),
            "model": model_config,
            "training": training_config,
            "training_summary": training_summary,
        },
    )


def load_model_checkpoint(checkpoint_dir: Path, device: torch.device) -> tuple[PointCloudSetClassifier, dict]:
    cfg = load_json(checkpoint_dir / "experiment_config.json")
    model = build_model_from_config(cfg["model"]).to(device)
    state_dict = torch.load(checkpoint_dir / "model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg
