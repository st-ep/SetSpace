#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.point_cloud_consistency.models import build_point_cloud_classifier
from case_studies.scanobjectnn_consistency.benchmark import evaluate_classifier, save_training_artifacts, train_classifier
from case_studies.scanobjectnn_consistency.common import DEFAULT_EVAL_MODES, DEFAULT_POINT_COUNTS, save_json, set_random_seed
from case_studies.scanobjectnn_consistency.dataset import ScanObjectNNConsistencyDataset
from case_studies.scanobjectnn_consistency.plot_consistency import plot_metrics
from case_studies.scanobjectnn_consistency.plot_qualitative import plot_qualitative


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full ScanObjectNN consistency benchmark.")
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results" / "scanobjectnn_consistency_run"))
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "ScanObjectNN" / "h5_files"))
    parser.add_argument("--variant", default="pb_t50_rs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--backbone", choices=["set_encoder", "pointnext"], default="set_encoder")
    parser.add_argument("--train_points", type=int, default=512)
    parser.add_argument("--reference_points", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--val_objects", type=int, default=256)
    parser.add_argument("--test_objects", type=int, default=512)
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

    dataset = ScanObjectNNConsistencyDataset(
        data_root=args.data_root,
        variant=args.variant,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    base_model_config = {
        "task": "classification",
        "value_input_dim": 1,
        "num_classes": dataset.num_classes,
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
        "pointnext_width": 32,
        "pointnext_blocks": [1, 1, 1, 1, 1, 1],
        "pointnext_strides": [1, 2, 2, 2, 2, 1],
        "pointnext_radius": 0.15,
        "pointnext_radius_scaling": 1.5,
        "pointnext_nsample": 32,
        "pointnext_expansion": 4,
        "pointnext_sa_layers": 2,
        "pointnext_sa_use_res": True,
        "pointnext_normalize_dp": True,
        "pointnext_head_hidden_dim": args.hidden_dim,
    }
    training_config = {
        "train_points": args.train_points,
        "train_sampling_mode": "uniform_object",
        "backbone": args.backbone,
        "reference_points": args.reference_points,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "eval_every": args.eval_every,
        "val_objects": args.val_objects,
        "test_objects": args.test_objects,
        "seed": args.seed,
    }

    model_specs = [("uniform", "uniform")] if args.backbone == "pointnext" else [("uniform", "uniform"), ("geometry_aware", "knn")]
    trained_models = {}
    for model_name, weight_mode in model_specs:
        model_config = {**base_model_config, "weight_mode": weight_mode}
        model = build_point_cloud_classifier(
            backbone=args.backbone,
            activation_fn=torch.nn.GELU,
            value_input_dim=1,
            num_classes=dataset.num_classes,
            n_tokens=args.n_tokens,
            token_dim=args.token_dim,
            key_dim=args.key_dim,
            hidden_dim=args.hidden_dim,
            basis_activation=args.basis_activation,
            value_mode=args.value_mode,
            normalize=args.normalize,
            weight_mode=weight_mode,
            knn_k=args.knn_k,
            intrinsic_dim=args.intrinsic_dim,
            pointnext_width=base_model_config["pointnext_width"],
            pointnext_blocks=tuple(base_model_config["pointnext_blocks"]),
            pointnext_strides=tuple(base_model_config["pointnext_strides"]),
            pointnext_radius=base_model_config["pointnext_radius"],
            pointnext_radius_scaling=base_model_config["pointnext_radius_scaling"],
            pointnext_nsample=base_model_config["pointnext_nsample"],
            pointnext_expansion=base_model_config["pointnext_expansion"],
            pointnext_sa_layers=base_model_config["pointnext_sa_layers"],
            pointnext_sa_use_res=base_model_config["pointnext_sa_use_res"],
            pointnext_normalize_dp=base_model_config["pointnext_normalize_dp"],
            pointnext_head_hidden_dim=base_model_config["pointnext_head_hidden_dim"],
        )
        summary = train_classifier(
            model,
            dataset,
            run_name=model_name,
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
        "dataset_metadata": dataset.get_metadata(),
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
    plot_qualitative(
        payload,
        output_dir,
        dataset=dataset,
        models=trained_models,
        device=device,
        fixed_points=min(args.point_counts),
        reference_points=args.reference_points,
    )
    print(f"Saved benchmark outputs to {output_dir}")
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
