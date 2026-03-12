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

SENSOR_COUNTS = [5, 10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
N_GRID = 501
CLR_GEOM = "#1b9e77"
CLR_UNIF = "#d95f02"


def synthetic_convergence():
    true_integral = 3 * np.pi * (np.e + 1) / (1 + 9 * np.pi**2)
    Ms = np.arange(20, 301, 1)
    err_geom = np.empty(len(Ms))
    err_unif = np.empty(len(Ms))
    for j, M in enumerate(Ms):
        x = np.linspace(0.0, 1.0, M)
        f = np.exp(x) * np.sin(3 * np.pi * x)
        dx = x[1:] - x[:-1]
        w_geom = np.zeros(M)
        w_geom[0] = dx[0] / 2
        w_geom[-1] = dx[-1] / 2
        w_geom[1:-1] = (dx[:-1] + dx[1:]) / 2
        err_geom[j] = abs(np.sum(w_geom * f) / np.sum(w_geom) - true_integral)
        err_unif[j] = abs(np.mean(f) - true_integral)
    return Ms, err_geom, err_unif


def branch_drift(device: torch.device, data_path: str, n_test: int = 200):
    dataset = load_darcy_dataset(data_path)
    test_data = dataset["test"]
    grid = torch.tensor(dataset["train"][0]["X"], dtype=torch.float32, device=device).view(-1, 1)
    m_geom = build_geometry_aware_model(device)
    m_unif = build_uniform_model(device)
    sample_range = range(min(n_test, len(test_data)))

    def get_branches(model, n_sensors: int):
        sensor_indices = torch.linspace(0, N_GRID - 1, n_sensors, dtype=torch.long, device=device)
        sensor_x = grid[sensor_indices]
        vecs = []
        with torch.no_grad():
            for i in sample_range:
                u_full = torch.tensor(test_data[i]["u"], dtype=torch.float32, device=device)
                branch = model.forward_branch(sensor_x.unsqueeze(0), u_full[sensor_indices].view(1, -1, 1))
                vecs.append(branch.squeeze(0).squeeze(-1))
        return torch.stack(vecs)

    ref_bg = get_branches(m_geom, 300)
    ref_bu = get_branches(m_unif, 300)
    drift_g, drift_u = [], []
    for M in SENSOR_COUNTS:
        bg = get_branches(m_geom, M)
        bu = get_branches(m_unif, M)
        drift_g.append((torch.norm(bg - ref_bg, dim=1) / torch.norm(ref_bg, dim=1).clamp_min(1e-12)).mean().item())
        drift_u.append((torch.norm(bu - ref_bu, dim=1) / torch.norm(ref_bu, dim=1).clamp_min(1e-12)).mean().item())
    return SENSOR_COUNTS, drift_g, drift_u


def main():
    parser = argparse.ArgumentParser(description="Continuum-functional convergence vs branch drift.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_path", default=str(REPO_ROOT / "data" / "darcy_1d" / "darcy_1d_dataset_501"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--n_test", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Ms_syn, err_geom_syn, err_unif_syn = synthetic_convergence()
    eval_Ms, drift_g, drift_u = branch_drift(device, args.data_path, args.n_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax in axes:
        ax.set_box_aspect(1)

    ax = axes[0]
    ax.loglog(Ms_syn, err_geom_syn, "-", color=CLR_GEOM, lw=1.5, alpha=0.7, label="Geometry-aware")
    ax.loglog(Ms_syn, err_unif_syn, "-", color=CLR_UNIF, lw=1.5, alpha=0.7, label="Uniform / $M$")
    M_ref = np.array([20, 300])
    mid = len(Ms_syn) // 2
    c_unif = err_unif_syn[mid] * Ms_syn[mid]
    c_geom = err_geom_syn[mid] * Ms_syn[mid] ** 2
    ax.loglog(M_ref, c_unif / M_ref, ":", color=CLR_UNIF, lw=1, alpha=0.8)
    ax.loglog(M_ref, c_geom / M_ref**2, ":", color=CLR_GEOM, lw=1, alpha=0.8)
    ax.text(200, c_unif / 130, r"$O(1/M)$", color=CLR_UNIF, fontsize=11)
    ax.text(25, c_geom / 40**2 * 0.25, r"$O(1/M^2)$", color=CLR_GEOM, fontsize=11)
    ax.set_xlabel("$M$ (number of quadrature points)")
    ax.set_ylabel("Quadrature error")
    ax.set_title(r"(a) Continuum functional approximation")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(min(err_geom_syn.min(), err_unif_syn.min()) * 0.3, max(err_geom_syn.max(), err_unif_syn.max()) * 3)

    ax = axes[1]
    plot_mask = [i for i, M in enumerate(eval_Ms) if 20 <= M < 300]
    plot_Ms = [eval_Ms[i] for i in plot_mask]
    plot_dg = [drift_g[i] for i in plot_mask]
    plot_du = [drift_u[i] for i in plot_mask]
    ax.loglog(plot_Ms, plot_dg, "o-", color=CLR_GEOM, lw=2, ms=5, label="Geometry-aware encoder")
    ax.loglog(plot_Ms, plot_du, "s--", color=CLR_UNIF, lw=2, ms=5, label="Uniform encoder")
    M_ref2 = np.array([20, 300])
    c1 = drift_u[5] * eval_Ms[5]
    c2 = drift_g[5] * eval_Ms[5] ** 2
    ax.loglog(M_ref2, c1 / M_ref2, ":", color=CLR_UNIF, lw=1, alpha=0.6)
    ax.loglog(M_ref2, c2 / M_ref2**2, ":", color=CLR_GEOM, lw=1, alpha=0.6)
    ax.text(100, c1 / 60 * 2.5, r"$O(1/M)$", color=CLR_UNIF, fontsize=11)
    ax.text(100, c2 / 150**2 * 0.4, r"$O(1/M^2)$", color=CLR_GEOM, fontsize=11)
    ax.set_xlabel("$M$ (number of sampled elements)")
    ax.set_ylabel("Relative branch output error")
    ax.set_title("(b) Set-encoder drift under refinement")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(out_dir / "quadrature_convergence.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "quadrature_convergence.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
