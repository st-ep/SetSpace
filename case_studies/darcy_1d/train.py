#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.darcy_1d.common import get_activation
from data.darcy_1d import DarcyDataGenerator, create_query_points, create_sensor_points, load_darcy_dataset
from set_encoders import SetEncoderOperator
from set_encoders.utils import calculate_l2_relative_error


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Darcy 1D set-encoder case study.")
    parser.add_argument("--data_path", default=str(REPO_ROOT / "data" / "darcy_1d" / "darcy_1d_dataset_501"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "checkpoints" / "run"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensor_size", type=int, default=300)
    parser.add_argument("--n_query_points", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=125000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--uniform_weights", action="store_true")
    parser.add_argument("--p", type=int, default=32)
    parser.add_argument("--phi_hidden", type=int, default=256)
    parser.add_argument("--rho_hidden", type=int, default=200)
    parser.add_argument("--trunk_hidden", type=int, default=256)
    parser.add_argument("--n_trunk_layers", type=int, default=4)
    parser.add_argument("--phi_output_size", type=int, default=32)
    parser.add_argument("--pos_encoding_dim", type=int, default=64)
    parser.add_argument("--pos_encoding_max_freq", type=float, default=0.1)
    parser.add_argument("--activation_fn", default="relu")
    parser.add_argument("--key_dim", type=int, default=64)
    parser.add_argument("--basis_activation", default="softplus")
    parser.add_argument("--value_mode", default="linear_u")
    parser.add_argument(
        "--lr_schedule_steps",
        type=int,
        nargs="+",
        default=[25000, 75000, 125000, 175000, 1250000, 1500000],
    )
    parser.add_argument("--lr_schedule_gammas", type=float, nargs="+", default=[0.2, 0.5, 0.2, 0.5, 0.2, 0.5])
    return parser.parse_args()


def evaluate(model, dataset, sensor_indices, query_indices, device):
    test_data = dataset["test"]
    grid = torch.tensor(dataset["train"][0]["X"], dtype=torch.float32, device=device).view(-1, 1)
    sensor_x = grid[sensor_indices]
    query_x = grid[query_indices]
    total_mse = 0.0
    total_rel = 0.0
    with torch.no_grad():
        for sample in test_data:
            u = torch.tensor(sample["u"], dtype=torch.float32, device=device)[sensor_indices].view(1, -1, 1)
            target = torch.tensor(sample["s"], dtype=torch.float32, device=device)[query_indices].view(1, -1, 1)
            pred = model(sensor_x.unsqueeze(0), u, query_x.unsqueeze(0))
            total_mse += torch.nn.functional.mse_loss(pred, target).item()
            total_rel += calculate_l2_relative_error(pred.squeeze(-1), target.squeeze(-1)).item()
    return total_mse / len(test_data), total_rel / len(test_data)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = load_darcy_dataset(args.data_path)
    grid_points = torch.tensor(dataset["train"][0]["X"], dtype=torch.float32)
    _, sensor_indices = create_sensor_points(args.sensor_size, device, grid_points)
    _, query_indices = create_query_points(device, grid_points, args.n_query_points)
    train_data = DarcyDataGenerator(dataset, sensor_indices, query_indices, device, args.batch_size, grid_points)

    model = SetEncoderOperator(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=args.p,
        phi_hidden_size=args.phi_hidden,
        rho_hidden_size=args.rho_hidden,
        trunk_hidden_size=args.trunk_hidden,
        n_trunk_layers=args.n_trunk_layers,
        activation_fn=get_activation(args.activation_fn),
        use_deeponet_bias=True,
        phi_output_size=args.phi_output_size,
        initial_lr=args.lr,
        lr_schedule_steps=args.lr_schedule_steps,
        lr_schedule_gammas=args.lr_schedule_gammas,
        use_positional_encoding=True,
        pos_encoding_type="sinusoidal",
        pos_encoding_dim=args.pos_encoding_dim,
        pos_encoding_max_freq=args.pos_encoding_max_freq,
        key_dim=args.key_dim,
        basis_activation=args.basis_activation,
        value_mode=args.value_mode,
        uniform_sensor_weights=args.uniform_weights,
    ).to(device)

    model.train_model(train_data, epochs=args.epochs, progress_bar=True)
    mse, rel = evaluate(model, dataset, sensor_indices, query_indices, device)
    torch.save(model.state_dict(), output_dir / "darcy1d_setonet_model.pth")

    config = {
        "paper_context": {
            "framing": "Set encoders as discretizations of continuum functionals",
            "case_study": "Darcy 1D operator learning",
        },
        "model_architecture": {
            "activation_fn": args.activation_fn,
            "input_size_src": 1,
            "output_size_src": 1,
            "input_size_tgt": 1,
            "output_size_tgt": 1,
            "son_p_dim": args.p,
            "son_phi_hidden": args.phi_hidden,
            "son_rho_hidden": args.rho_hidden,
            "son_trunk_hidden": args.trunk_hidden,
            "son_n_trunk_layers": args.n_trunk_layers,
            "son_phi_output_size": args.phi_output_size,
            "use_positional_encoding": True,
            "pos_encoding_type": "sinusoidal",
            "pos_encoding_dim": args.pos_encoding_dim,
            "pos_encoding_max_freq": args.pos_encoding_max_freq,
        },
        "training_parameters": vars(args),
        "evaluation_results": {"test_mse_loss": mse, "test_relative_l2_error": rel},
    }
    with open(output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved model to {output_dir}")
    print(f"Test MSE: {mse:.6e}")
    print(f"Test relative L2: {rel:.6f}")


if __name__ == "__main__":
    main()
