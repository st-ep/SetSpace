#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

from case_studies.darcy_1d.common import build_geometry_aware_model, build_uniform_model
from data.darcy_1d import load_darcy_dataset

SENSOR_COUNTS = [5, 15, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
N_GRID = 501
N_QUERY = 300
CLR_GEOM = "#1b9e77"
CLR_UNIF = "#d95f02"


def main():
    parser = argparse.ArgumentParser(description="Weight analysis for the Darcy 1D case study.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", default=str(REPO_ROOT / "data" / "darcy_1d" / "darcy_1d_dataset_501"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--sample_idx", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_darcy_dataset(args.data_path)
    test_data = dataset["test"]
    grid = torch.tensor(dataset["train"][0]["X"], dtype=torch.float32, device=device).view(-1, 1)
    query_indices = torch.linspace(0, N_GRID - 1, N_QUERY, dtype=torch.long, device=device)
    query_x = grid[query_indices]

    geom = build_geometry_aware_model(device)
    unif = build_uniform_model(device)
    n_test = min(args.n_test, len(test_data))
    sample_range = range(n_test)

    def get_branch_and_pred(model, n_sensors, indices):
        sensor_indices = torch.linspace(0, N_GRID - 1, n_sensors, dtype=torch.long, device=device)
        sensor_x = grid[sensor_indices]
        branches, preds = [], []
        with torch.no_grad():
            for i in indices:
                u_full = torch.tensor(test_data[i]["u"], dtype=torch.float32, device=device)
                u = u_full[sensor_indices].view(1, -1, 1)
                y = query_x.unsqueeze(0)
                b = model.forward_branch(sensor_x.unsqueeze(0), u)
                p = model(sensor_x.unsqueeze(0), u, y)
                branches.append(b.squeeze(0).squeeze(-1))
                preds.append(p.squeeze(0).squeeze(-1))
        return torch.stack(branches), torch.stack(preds)

    ref_b_geom, _ = get_branch_and_pred(geom, 300, sample_range)
    ref_b_unif, _ = get_branch_and_pred(unif, 300, sample_range)
    targets = torch.stack(
        [torch.tensor(test_data[i]["s"], dtype=torch.float32, device=device)[query_indices] for i in sample_range]
    )

    drift_geom, drift_unif = [], []
    mse_per_sample_250 = {}
    for M in SENSOR_COUNTS:
        bg, pg = get_branch_and_pred(geom, M, sample_range)
        bu, pu = get_branch_and_pred(unif, M, sample_range)
        drift_geom.append((torch.norm(bg - ref_b_geom, dim=1) / torch.norm(ref_b_geom, dim=1).clamp_min(1e-12)).mean().item())
        drift_unif.append((torch.norm(bu - ref_b_unif, dim=1) / torch.norm(ref_b_unif, dim=1).clamp_min(1e-12)).mean().item())
        if M == 250:
            mse_per_sample_250["geom"] = ((pg - targets) ** 2).mean(dim=1).cpu().numpy()
            mse_per_sample_250["unif"] = ((pu - targets) ** 2).mean(dim=1).cpu().numpy()

    sample = test_data[args.sample_idx]
    sensor_indices = torch.linspace(0, N_GRID - 1, 250, dtype=torch.long, device=device)
    sensor_x = grid[sensor_indices]
    u_full = torch.tensor(sample["u"], dtype=torch.float32, device=device)
    s_full = torch.tensor(sample["s"], dtype=torch.float32, device=device)
    target_np = s_full[query_indices].cpu().numpy()
    query_np = query_x.squeeze(-1).cpu().numpy()
    with torch.no_grad():
        u = u_full[sensor_indices].view(1, -1, 1)
        y = query_x.unsqueeze(0)
        err_geom = geom(sensor_x.unsqueeze(0), u, y).squeeze().cpu().numpy() - target_np
        err_unif = unif(sensor_x.unsqueeze(0), u, y).squeeze().cpu().numpy() - target_np

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ax.plot(SENSOR_COUNTS, drift_geom, "o-", color=CLR_GEOM, lw=2.2, ms=6, label="Geometry-aware encoder")
    ax.plot(SENSOR_COUNTS, drift_unif, "s--", color=CLR_UNIF, lw=2.2, ms=6, label="Uniform encoder")
    ax.set_xlabel("Number of sampled elements ($M$)")
    ax.set_ylabel(r"$\|\mathbf{b}(M)-\mathbf{b}(300)\|/\|\mathbf{b}(300)\|$")
    ax.set_title("(a) Branch output drift from $M$=300")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(query_np, err_geom, 0, alpha=0.35, color=CLR_GEOM, label="Geometry-aware error")
    ax.fill_between(query_np, err_unif, 0, alpha=0.35, color=CLR_UNIF, label="Uniform error")
    ax.plot(query_np, err_geom, "-", color=CLR_GEOM, lw=1.5)
    ax.plot(query_np, err_unif, "-", color=CLR_UNIF, lw=1.5)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$\hat{s}(x)-s(x)$")
    ax.set_title(f"(b) Pointwise error at $M$=250 (sample {args.sample_idx})")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    sort_idx = np.argsort(mse_per_sample_250["unif"])[::-1]
    x_range = np.arange(len(sort_idx))
    ax.semilogy(x_range, mse_per_sample_250["unif"][sort_idx], ".", color=CLR_UNIF, alpha=0.5, ms=4, label="Uniform")
    ax.semilogy(x_range, mse_per_sample_250["geom"][sort_idx], ".", color=CLR_GEOM, alpha=0.5, ms=4, label="Geometry-aware")
    ax.set_xlabel("Test sample (sorted by uniform MSE)")
    ax.set_ylabel("Per-sample MSE")
    ax.set_title("(c) Per-sample MSE at $M$=250")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=2.5)
    fig.savefig(out_dir / "weight_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "weight_analysis.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
