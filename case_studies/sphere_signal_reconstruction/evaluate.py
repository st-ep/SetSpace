#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.sphere_signal_reconstruction.benchmark import (
    evaluate_deterministic_convergence,
    evaluate_reconstructor,
    load_model_checkpoint,
)
from case_studies.sphere_signal_reconstruction.common import (
    DEFAULT_EVAL_MODES,
    DEFAULT_POINT_COUNTS,
    build_dataset_from_config,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate sphere signal reconstruction checkpoints.")
    parser.add_argument("--uniform_checkpoint", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--moment2_checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", default=str(REPO_ROOT / "results" / "sphere_signal_reconstruction_metrics.json"))
    parser.add_argument("--reference_points", type=int, default=1024)
    parser.add_argument("--n_resamples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_test_objects", type=int, default=256)
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
    for name, _model, cfg, _checkpoint_dir in loaded[1:]:
        if cfg["dataset"] != base_cfg["dataset"]:
            raise ValueError(f"Checkpoint '{name}' must use the same dataset configuration as the uniform checkpoint.")

    dataset = build_dataset_from_config(base_cfg["dataset"])
    payload = {
        "dataset": base_cfg["dataset"],
        "training": base_cfg.get("training", {}),
        "normalization": base_cfg.get("normalization", dataset.get_normalization_stats()),
        "reference_points": args.reference_points,
        "n_resamples": args.n_resamples,
        "point_counts": args.point_counts,
        "sampling_modes": args.sampling_modes,
        "models": {},
        "deterministic_convergence": {},
    }

    for name, model, _cfg, checkpoint_dir in loaded:
        metrics = evaluate_reconstructor(
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
        convergence = evaluate_deterministic_convergence(
            model,
            dataset,
            split="test",
            device=device,
            point_counts=args.point_counts,
            reference_points=args.reference_points,
            batch_size=args.batch_size,
            max_objects=args.max_test_objects,
        )
        payload["models"][name] = {
            "checkpoint_dir": str(checkpoint_dir),
            "metrics": metrics,
        }
        payload["deterministic_convergence"][name] = convergence

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, payload)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
