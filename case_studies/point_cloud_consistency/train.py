#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.point_cloud_consistency.benchmark import save_training_artifacts, train_classifier
from case_studies.point_cloud_consistency.common import DEFAULT_EVAL_MODES, set_random_seed
from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset
from case_studies.point_cloud_consistency.models import PointCloudSetClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train the synthetic point-cloud consistency benchmark.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "checkpoints" / "point_cloud_consistency" / "run"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=2048)
    parser.add_argument("--n_val", type=int, default=256)
    parser.add_argument("--n_test", type=int, default=512)
    parser.add_argument("--n_bumps", type=int, default=4)
    parser.add_argument("--label_reference_points", type=int, default=4096)
    parser.add_argument("--train_points", type=int, default=128)
    parser.add_argument("--train_sampling_mode", default="uniform")
    parser.add_argument("--weight_mode", choices=["uniform", "knn"], default="uniform")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--val_objects", type=int, default=128)
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

    dataset = SyntheticSurfaceSignalDataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        n_bumps=args.n_bumps,
        label_reference_points=args.label_reference_points,
    )

    model_config = {
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
        "weight_mode": args.weight_mode,
        "knn_k": args.knn_k,
        "intrinsic_dim": args.intrinsic_dim,
    }
    model = PointCloudSetClassifier(**{k: v for k, v in model_config.items() if k != "activation_fn"})

    training_summary = train_classifier(
        model,
        dataset,
        run_name=args.weight_mode,
        device=device,
        train_points=args.train_points,
        train_sampling_mode=args.train_sampling_mode,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        val_sampling_modes=DEFAULT_EVAL_MODES,
        val_objects=args.val_objects,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    save_training_artifacts(
        output_dir,
        model=model,
        dataset=dataset,
        model_config=model_config,
        training_config={
            "train_points": args.train_points,
            "train_sampling_mode": args.train_sampling_mode,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "eval_every": args.eval_every,
            "val_objects": args.val_objects,
            "seed": args.seed,
        },
        training_summary=training_summary,
    )
    print(f"Saved checkpoint to {output_dir}")
    print(f"Best validation score: {training_summary['best_val_score']:.4f}")


if __name__ == "__main__":
    main()
