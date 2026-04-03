#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.ahmedml_surface_forces.benchmark import (
    evaluate_regressor,
    save_training_artifacts,
    train_regressor,
)
from case_studies.ahmedml_surface_forces.common import (
    DEFAULT_EVAL_MODES,
    DEFAULT_POINT_COUNTS,
    save_json,
    set_random_seed,
)
from case_studies.ahmedml_surface_forces.dataset import AhmedMLSurfaceForceDataset
from case_studies.ahmedml_surface_forces.models import build_force_regressor
from case_studies.ahmedml_surface_forces.plot_convergence import plot_metrics
from case_studies.ahmedml_surface_forces.plot_qualitative import plot_prediction_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AhmedML sparse-surface force benchmark.")
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "ahmedml_surface_forces_run"))
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
    parser.add_argument("--test_objects", type=int, default=128)
    parser.add_argument("--reference_points", type=int, default=2048)
    parser.add_argument("--n_resamples", type=int, default=3)
    parser.add_argument("--point_counts", type=int, nargs="+", default=DEFAULT_POINT_COUNTS)
    parser.add_argument("--sampling_modes", nargs="+", default=DEFAULT_EVAL_MODES)
    parser.add_argument("--backbone", choices=["set_encoder", "pointnext"], default="set_encoder")
    parser.add_argument("--n_tokens", type=int, default=16)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--key_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--basis_activation", default="softplus")
    parser.add_argument("--value_mode", default="mlp_xu")
    parser.add_argument("--normalize", default="total")
    parser.add_argument("--knn_k", type=int, default=8)
    parser.add_argument("--intrinsic_dim", type=int, default=2)
    parser.add_argument("--pointnext_width", type=int, default=32)
    parser.add_argument("--pointnext_blocks", type=int, nargs="+", default=[1, 1, 1, 1, 1, 1])
    parser.add_argument("--pointnext_strides", type=int, nargs="+", default=[1, 2, 2, 2, 2, 1])
    parser.add_argument("--pointnext_radius", type=float, default=0.15)
    parser.add_argument("--pointnext_radius_scaling", type=float, default=1.5)
    parser.add_argument("--pointnext_nsample", type=int, default=32)
    parser.add_argument("--pointnext_expansion", type=int, default=4)
    parser.add_argument("--pointnext_sa_layers", type=int, default=2)
    parser.add_argument("--pointnext_sa_use_res", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pointnext_normalize_dp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pointnext_head_hidden_dim", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    dataset = AhmedMLSurfaceForceDataset(
        processed_root=args.processed_root,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
    )
    base_model_config = {
        "value_input_dim": dataset.value_input_dim,
        "output_dim": dataset.target_dim,
        "backbone": args.backbone,
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
        "pointnext_width": args.pointnext_width,
        "pointnext_blocks": list(args.pointnext_blocks),
        "pointnext_strides": list(args.pointnext_strides),
        "pointnext_radius": args.pointnext_radius,
        "pointnext_radius_scaling": args.pointnext_radius_scaling,
        "pointnext_nsample": args.pointnext_nsample,
        "pointnext_expansion": args.pointnext_expansion,
        "pointnext_sa_layers": args.pointnext_sa_layers,
        "pointnext_sa_use_res": args.pointnext_sa_use_res,
        "pointnext_normalize_dp": args.pointnext_normalize_dp,
        "pointnext_head_hidden_dim": args.pointnext_head_hidden_dim,
    }
    training_config = {
        "train_points": args.train_points,
        "train_sampling_mode": "uniform",
        "backbone": args.backbone,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "eval_every": args.eval_every,
        "val_objects": args.val_objects,
        "seed": args.seed,
    }
    model_specs = (
        [{"name": "pointnext", "weight_mode": "uniform"}]
        if args.backbone == "pointnext"
        else [
            {"name": "uniform", "weight_mode": "uniform"},
            {"name": "geometry_aware", "weight_mode": "knn"},
        ]
    )
    trained_models = {}
    for spec in model_specs:
        model_name = spec["name"]
        model_config = {
            **base_model_config,
            "weight_mode": spec["weight_mode"],
        }
        model = build_force_regressor(
            value_input_dim=dataset.value_input_dim,
            output_dim=dataset.target_dim,
            activation_fn=torch.nn.GELU,
            backbone=args.backbone,
            n_tokens=args.n_tokens,
            token_dim=args.token_dim,
            key_dim=args.key_dim,
            hidden_dim=args.hidden_dim,
            basis_activation=args.basis_activation,
            value_mode=args.value_mode,
            normalize=args.normalize,
            weight_mode=spec["weight_mode"],
            knn_k=args.knn_k,
            intrinsic_dim=args.intrinsic_dim,
            pointnext_width=args.pointnext_width,
            pointnext_blocks=tuple(args.pointnext_blocks),
            pointnext_strides=tuple(args.pointnext_strides),
            pointnext_radius=args.pointnext_radius,
            pointnext_radius_scaling=args.pointnext_radius_scaling,
            pointnext_nsample=args.pointnext_nsample,
            pointnext_expansion=args.pointnext_expansion,
            pointnext_sa_layers=args.pointnext_sa_layers,
            pointnext_sa_use_res=args.pointnext_sa_use_res,
            pointnext_normalize_dp=args.pointnext_normalize_dp,
            pointnext_head_hidden_dim=args.pointnext_head_hidden_dim,
        )
        summary = train_regressor(
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
        "normalization": dataset.get_normalization_stats(),
        "reference_points": args.reference_points,
        "n_resamples": args.n_resamples,
        "point_counts": args.point_counts,
        "sampling_modes": args.sampling_modes,
        "models": {},
    }
    for model_name, model in trained_models.items():
        metrics = evaluate_regressor(
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
    if args.backbone == "set_encoder":
        plot_metrics(payload, output_dir)
        plot_prediction_figure(payload, output_dir, dataset=dataset, device=device)
    print(f"Saved benchmark outputs to {output_dir}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
