#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.point_cloud_consistency.benchmark import (
    evaluate_classifier,
    save_training_artifacts,
    train_classifier,
)
from case_studies.point_cloud_consistency.common import (
    DEFAULT_EVAL_MODES,
    DEFAULT_POINT_COUNTS,
    save_json,
    set_random_seed,
)
from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset
from case_studies.point_cloud_consistency.models import PointCloudSetClassifier
from case_studies.point_cloud_consistency.plot_consistency import plot_metrics
from case_studies.point_cloud_consistency.plot_overview import plot_benchmark_overview
from case_studies.point_cloud_consistency.plot_qualitative import plot_qualitative_responses


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full synthetic point-cloud consistency benchmark.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "point_cloud_consistency_run"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=2048)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--n_bumps", type=int, default=4)
    parser.add_argument("--label_reference_points", type=int, default=4096)
    parser.add_argument("--train_points", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--val_objects", type=int, default=128)
    parser.add_argument("--test_objects", type=int, default=256)
    parser.add_argument("--reference_points", type=int, default=2048)
    parser.add_argument("--n_resamples", type=int, default=3)
    parser.add_argument("--point_counts", type=int, nargs="+", default=DEFAULT_POINT_COUNTS)
    parser.add_argument("--sampling_modes", nargs="+", default=DEFAULT_EVAL_MODES)
    parser.add_argument("--n_tokens", type=int, default=16)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--key_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--basis_activation", default="softplus")
    parser.add_argument("--value_mode", default="mlp_xu")
    parser.add_argument("--normalize", default="total")
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--intrinsic_dim", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticSurfaceSignalDataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        n_bumps=args.n_bumps,
        label_reference_points=args.label_reference_points,
    )

    base_model_config = {
        "value_input_dim": 1,
        "num_classes": 2,
        "n_tokens": args.n_tokens,
        "token_dim": args.token_dim,
        "key_dim": args.key_dim,
        "hidden_dim": args.hidden_dim,
        "activation_fn": "gelu",
        "basis_activation": args.basis_activation,
        "value_mode": args.value_mode,
        "normalize": args.normalize,
        "knn_k": args.knn_k,
        "intrinsic_dim": args.intrinsic_dim,
    }
    training_config = {
        "train_points": args.train_points,
        "train_sampling_mode": "uniform",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "eval_every": args.eval_every,
        "val_objects": args.val_objects,
        "seed": args.seed,
    }

    trained_models = {}
    for model_name, weight_mode in [("uniform", "uniform"), ("geometry_aware", "knn")]:
        model_config = {**base_model_config, "weight_mode": weight_mode}
        model = PointCloudSetClassifier(**{k: v for k, v in model_config.items() if k != "activation_fn"})
        summary = train_classifier(
            model,
            dataset,
            run_name=model_name,
            device=device,
            train_points=args.train_points,
            train_sampling_mode="uniform",
            batch_size=args.batch_size,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            eval_every=args.eval_every,
            val_sampling_modes=args.sampling_modes,
            val_objects=args.val_objects,
            seed=args.seed,
        )
        model_dir = checkpoints_dir / model_name
        save_training_artifacts(
            model_dir,
            model=model,
            dataset=dataset,
            model_config=model_config,
            training_config=training_config,
            training_summary=summary,
        )
        trained_models[model_name] = model

    payload = {
        "dataset": dataset.get_config(),
        "training": training_config,
        "reference_points": args.reference_points,
        "n_resamples": args.n_resamples,
        "point_counts": args.point_counts,
        "sampling_modes": args.sampling_modes,
        "models": {},
    }
    for model_name, model in trained_models.items():
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
            max_objects=args.test_objects,
        )
        payload["models"][model_name] = {
            "checkpoint_dir": str(checkpoints_dir / model_name),
            "metrics": metrics,
        }

    metrics_path = output_dir / "metrics.json"
    save_json(metrics_path, payload)
    plot_metrics(payload, output_dir, fixed_points=min(args.point_counts))
    plot_benchmark_overview(payload, output_dir)
    plot_qualitative_responses(
        payload,
        output_dir,
        dataset=dataset,
        models=trained_models,
        device=device,
        fixed_points=min(args.point_counts),
    )
    print(f"Saved benchmark outputs to {output_dir}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
