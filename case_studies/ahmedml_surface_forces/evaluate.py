#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.ahmedml_surface_forces.benchmark import evaluate_regressor, load_model_checkpoint
from case_studies.ahmedml_surface_forces.common import DEFAULT_EVAL_MODES, DEFAULT_POINT_COUNTS, save_json
from case_studies.ahmedml_surface_forces.dataset import AhmedMLSurfaceForceDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AhmedML surface-force checkpoints.")
    parser.add_argument("--uniform_checkpoint", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_path", default=str(REPO_ROOT / "results" / "ahmedml_surface_forces_metrics.json"))
    parser.add_argument("--point_counts", type=int, nargs="+", default=DEFAULT_POINT_COUNTS)
    parser.add_argument("--sampling_modes", nargs="+", default=DEFAULT_EVAL_MODES)
    parser.add_argument("--reference_points", type=int, default=2048)
    parser.add_argument("--n_resamples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_objects", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    uniform_model, uniform_cfg = load_model_checkpoint(Path(args.uniform_checkpoint), device)
    geometry_model, _ = load_model_checkpoint(Path(args.geometry_checkpoint), device)
    dataset = AhmedMLSurfaceForceDataset(**uniform_cfg["dataset"])
    payload = {
        "dataset": dataset.get_config(),
        "reference_points": args.reference_points,
        "n_resamples": args.n_resamples,
        "point_counts": args.point_counts,
        "sampling_modes": args.sampling_modes,
        "models": {
            "uniform": {
                "checkpoint_dir": str(args.uniform_checkpoint),
                "metrics": evaluate_regressor(
                    uniform_model,
                    dataset,
                    split="test",
                    device=device,
                    point_counts=args.point_counts,
                    sampling_modes=args.sampling_modes,
                    n_resamples=args.n_resamples,
                    reference_points=args.reference_points,
                    batch_size=args.batch_size,
                    max_objects=args.test_objects,
                ),
            },
            "geometry_aware": {
                "checkpoint_dir": str(args.geometry_checkpoint),
                "metrics": evaluate_regressor(
                    geometry_model,
                    dataset,
                    split="test",
                    device=device,
                    point_counts=args.point_counts,
                    sampling_modes=args.sampling_modes,
                    n_resamples=args.n_resamples,
                    reference_points=args.reference_points,
                    batch_size=args.batch_size,
                    max_objects=args.test_objects,
                ),
            },
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, payload)
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()

