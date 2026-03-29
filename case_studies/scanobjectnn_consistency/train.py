#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.point_cloud_consistency.models import PointCloudSetClassifier
from case_studies.scanobjectnn_consistency.benchmark import save_training_artifacts, train_classifier
from case_studies.scanobjectnn_consistency.common import DEFAULT_EVAL_MODES, set_random_seed
from case_studies.scanobjectnn_consistency.dataset import ScanObjectNNConsistencyDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train one ScanObjectNN consistency model.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "checkpoints" / "scanobjectnn_consistency" / "run"))
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "ScanObjectNN" / "h5_files"))
    parser.add_argument("--variant", default="pb_t50_rs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weight_mode", choices=["uniform", "knn"], default="knn")
    parser.add_argument("--train_points", type=int, default=512)
    parser.add_argument("--reference_points", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--val_objects", type=int, default=256)
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

    dataset = ScanObjectNNConsistencyDataset(
        data_root=args.data_root,
        variant=args.variant,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    model_config = {
        "task": "classification",
        "value_input_dim": 1,
        "num_classes": dataset.num_classes,
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
    training_config = {
        "train_points": args.train_points,
        "train_sampling_mode": "uniform_object",
        "reference_points": args.reference_points,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "eval_every": args.eval_every,
        "val_objects": args.val_objects,
        "seed": args.seed,
    }

    model = PointCloudSetClassifier(**{k: v for k, v in model_config.items() if k not in {"task", "activation_fn"}})
    summary = train_classifier(
        model,
        dataset,
        run_name=args.weight_mode,
        device=device,
        train_points=args.train_points,
        train_sampling_mode="uniform_object",
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        val_sampling_modes=args.sampling_modes,
        val_objects=args.val_objects,
        reference_points=args.reference_points,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    save_training_artifacts(
        output_dir,
        model=model,
        dataset=dataset,
        model_config=model_config,
        training_config=training_config,
        training_summary=summary,
    )
    print(f"Saved checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
