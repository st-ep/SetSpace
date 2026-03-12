#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.point_cloud_consistency.benchmark import evaluate_regressor, load_model_checkpoint
from case_studies.point_cloud_consistency.common import (
    DEFAULT_EVAL_MODES,
    DEFAULT_POINT_COUNTS,
    build_dataset_from_config,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate point-cloud mean-regression checkpoints under density shifts.")
    parser.add_argument("--uniform_checkpoint", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", default=str(REPO_ROOT / "results" / "point_cloud_mean_regression_metrics.json"))
    parser.add_argument("--reference_points", type=int, default=2048)
    parser.add_argument("--n_resamples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_test_objects", type=int, default=256)
    parser.add_argument("--point_counts", type=int, nargs="+", default=DEFAULT_POINT_COUNTS)
    parser.add_argument("--sampling_modes", nargs="+", default=DEFAULT_EVAL_MODES)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    uniform_model, uniform_cfg = load_model_checkpoint(Path(args.uniform_checkpoint), device)
    geometry_model, geometry_cfg = load_model_checkpoint(Path(args.geometry_checkpoint), device)

    if uniform_cfg["dataset"] != geometry_cfg["dataset"]:
        raise ValueError("Uniform and geometry-aware checkpoints must use the same dataset configuration.")

    dataset = build_dataset_from_config(uniform_cfg["dataset"])
    payload = {
        "task": "mean_regression",
        "dataset": uniform_cfg["dataset"],
        "training": uniform_cfg.get("training", {}),
        "reference_points": args.reference_points,
        "n_resamples": args.n_resamples,
        "point_counts": args.point_counts,
        "sampling_modes": args.sampling_modes,
        "models": {},
    }

    for name, model, checkpoint_dir in [
        ("uniform", uniform_model, args.uniform_checkpoint),
        ("geometry_aware", geometry_model, args.geometry_checkpoint),
    ]:
        summary = evaluate_regressor(
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
            "metrics": summary,
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, payload)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
