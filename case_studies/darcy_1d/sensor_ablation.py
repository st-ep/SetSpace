#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.darcy_1d.common import build_geometry_aware_model, build_uniform_model
from data.darcy_1d import load_darcy_dataset

SEED = 0
SENSOR_COUNTS = [5, 15, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
N_GRID = 501
N_QUERY = 300


def run_ablation(data_path: str, device: torch.device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    dataset = load_darcy_dataset(data_path)
    test_data = dataset["test"]
    grid = torch.tensor(dataset["train"][0]["X"], dtype=torch.float32, device=device).view(-1, 1)
    query_indices = torch.linspace(0, N_GRID - 1, N_QUERY, dtype=torch.long, device=device)
    query_x = grid[query_indices]

    models = {
        "Geometry-aware encoder": build_geometry_aware_model(device),
        "Uniform encoder": build_uniform_model(device),
    }
    results = {name: {} for name in models}

    for n_sensors in SENSOR_COUNTS:
        sensor_indices = torch.linspace(0, N_GRID - 1, n_sensors, dtype=torch.long, device=device)
        sensor_x = grid[sensor_indices]
        print(f"Sensor count: {n_sensors}")
        for name, model in models.items():
            total_mse = 0.0
            with torch.no_grad():
                for sample in test_data:
                    u = torch.tensor(sample["u"], dtype=torch.float32, device=device)[sensor_indices].view(1, -1, 1)
                    target = torch.tensor(sample["s"], dtype=torch.float32, device=device)[query_indices].view(1, -1, 1)
                    pred = model(sensor_x.unsqueeze(0), u, query_x.unsqueeze(0))
                    total_mse += torch.nn.functional.mse_loss(pred, target).item()
            results[name][n_sensors] = total_mse / len(test_data)
            print(f"  {name}: MSE = {results[name][n_sensors]:.6e}")
    return results


def plot_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Geometry-aware encoder": "#1b9e77", "Uniform encoder": "#d95f02"}
    markers = {"Geometry-aware encoder": "o", "Uniform encoder": "s"}
    for name, mse_by_n in results.items():
        x = list(mse_by_n.keys())
        y = [mse_by_n[n] for n in x]
        ax.plot(x, y, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=6)
    ax.set_xlabel("Number of sensors ($M$)")
    ax.set_ylabel("Test MSE")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.savefig(output_dir / "darcy_1d_sensor_ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "darcy_1d_sensor_ablation.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Darcy 1D sensor-count ablation.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", default=str(REPO_ROOT / "data" / "darcy_1d" / "darcy_1d_dataset_501"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results = run_ablation(args.data_path, device)
    plot_results(results, Path(args.output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
