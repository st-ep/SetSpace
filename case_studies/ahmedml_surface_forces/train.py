#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.ahmedml_surface_forces.benchmark import save_training_artifacts, train_regressor
from case_studies.ahmedml_surface_forces.common import DEFAULT_EVAL_MODES, set_random_seed
from case_studies.ahmedml_surface_forces.dataset import AhmedMLSurfaceForceDataset
from case_studies.ahmedml_surface_forces.models import build_force_regressor


def parse_args():
    parser = argparse.ArgumentParser(description="Train a set-encoder regressor on AhmedML sparse surface samples.")
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "checkpoints" / "ahmedml_surface_forces" / "geometry_aware"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_train", type=int, default=None)
    parser.add_argument("--n_val", type=int, default=None)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--train_points", type=int, default=256)
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--val_objects", type=int, default=64)
    parser.add_argument("--val_sampling_modes", nargs="+", default=DEFAULT_EVAL_MODES)
    parser.add_argument("--n_tokens", type=int, default=16)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--key_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--basis_activation", default="softplus")
    parser.add_argument("--value_mode", default="mlp_xu")
    parser.add_argument("--normalize", default="total")
    parser.add_argument("--weight_mode", choices=["uniform", "knn"], default="knn")
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--intrinsic_dim", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = AhmedMLSurfaceForceDataset(
        processed_root=args.processed_root,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
    )
    model_config = {
        "value_input_dim": dataset.value_input_dim,
        "output_dim": dataset.target_dim,
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
    model = build_force_regressor(
        value_input_dim=dataset.value_input_dim,
        output_dim=dataset.target_dim,
        activation_fn=torch.nn.GELU,
        n_tokens=args.n_tokens,
        token_dim=args.token_dim,
        key_dim=args.key_dim,
        hidden_dim=args.hidden_dim,
        basis_activation=args.basis_activation,
        value_mode=args.value_mode,
        normalize=args.normalize,
        weight_mode=args.weight_mode,
        knn_k=args.knn_k,
        intrinsic_dim=args.intrinsic_dim,
    )
    training_summary = train_regressor(
        model,
        dataset,
        run_name=args.weight_mode,
        device=device,
        train_points=args.train_points,
        train_sampling_mode="uniform",
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_every=args.eval_every,
        val_sampling_modes=args.val_sampling_modes,
        val_objects=args.val_objects,
        seed=args.seed,
    )
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
    save_training_artifacts(
        Path(args.output_dir),
        model=model,
        dataset=dataset,
        model_config=model_config,
        training_config=training_config,
        training_summary=training_summary,
    )


if __name__ == "__main__":
    main()
