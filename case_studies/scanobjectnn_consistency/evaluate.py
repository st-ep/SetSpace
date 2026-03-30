#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.scanobjectnn_consistency.benchmark import evaluate_classifier, load_model_checkpoint
from case_studies.scanobjectnn_consistency.common import DEFAULT_EVAL_MODES, DEFAULT_POINT_COUNTS, save_json
from case_studies.scanobjectnn_consistency.dataset import ScanObjectNNConsistencyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ScanObjectNN consistency checkpoints.")
    parser.add_argument("--uniform_checkpoint", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--moment2_checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "ScanObjectNN" / "h5_files"))
    parser.add_argument("--variant", default="pb_t50_rs")
    parser.add_argument("--output_path", default=str(REPO_ROOT / "results" / "scanobjectnn_consistency_metrics.json"))
    parser.add_argument("--reference_points", type=int, default=512)
    parser.add_argument("--n_resamples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_test_objects", type=int, default=512)
    parser.add_argument("--point_counts", type=int, nargs="+", default=DEFAULT_POINT_COUNTS)
    parser.add_argument("--sampling_modes", nargs="+", default=DEFAULT_EVAL_MODES)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    checkpoints = [
        ("uniform", args.uniform_checkpoint),
        ("geometry_aware", args.geometry_checkpoint),
    ]
    if args.moment2_checkpoint:
        checkpoints.append(("moment2", args.moment2_checkpoint))

    loaded = [(name, *load_model_checkpoint(Path(checkpoint_dir), device), checkpoint_dir) for name, checkpoint_dir in checkpoints]
    base_cfg = loaded[0][2]
    dataset_cfg = dict(base_cfg["dataset"])
    dataset_cfg["data_root"] = args.data_root
    dataset_cfg["variant"] = args.variant
    for name, _model, cfg, _checkpoint_dir in loaded[1:]:
        if cfg["dataset"].get("variant", "").lower() != dataset_cfg["variant"]:
            raise ValueError(f"Checkpoint '{name}' must use the same ScanObjectNN variant as the uniform checkpoint.")

    dataset = ScanObjectNNConsistencyDataset(**dataset_cfg)
    payload = {
        "dataset": dataset.get_config(),
        "dataset_metadata": dataset.get_metadata(),
        "training": base_cfg.get("training", {}),
        "reference_points": args.reference_points,
        "n_resamples": args.n_resamples,
        "point_counts": args.point_counts,
        "sampling_modes": args.sampling_modes,
        "models": {},
    }

    for name, model, _cfg, checkpoint_dir in loaded:
        metrics = evaluate_classifier(
            model,
            dataset,
            split="test",
            device=device,
            point_counts=args.point_counts,
            sampling_modes=args.sampling_modes,
            n_resamples=args.n_resamples,
            reference_points=args.reference_points,
            batch_size=args.batch_size,
            max_objects=args.max_test_objects,
        )
        payload["models"][name] = {
            "checkpoint_dir": str(checkpoint_dir),
            "metrics": metrics,
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, payload)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
