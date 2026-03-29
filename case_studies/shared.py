from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_activation(name: str | None) -> type[nn.Module]:
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "swish": nn.SiLU}.get(
        (name or "relu").lower(),
        nn.ReLU,
    )


def make_view_seeds(
    split: str,
    local_indices: torch.Tensor,
    *,
    n_points: int,
    sampling_mode: str,
    replica_idx: int,
    mode_offsets: dict[str, int],
) -> torch.Tensor:
    mode_offset = mode_offsets[sampling_mode]
    split_offset = {"train": 101, "val": 211, "test": 307}[split]
    return (
        local_indices.to(dtype=torch.long) * 65_537
        + split_offset
        + mode_offset * 997
        + replica_idx * 7_919
        + int(n_points)
    )


def avg_over_nonuniform_modes(
    per_setting: dict[str, dict[str, dict[str, float]]],
    metric: str,
    point_counts: list[int],
    sampling_modes: list[str],
    *,
    uniform_key: str = "uniform",
) -> dict[str, float]:
    nonuniform = [m for m in sampling_modes if m != uniform_key]
    if not nonuniform:
        nonuniform = [uniform_key]
    return {
        str(n): sum(per_setting[m][str(n)][metric] for m in nonuniform) / len(nonuniform)
        for n in point_counts
    }


def train_loop(
    model: nn.Module,
    *,
    run_name: str | None = None,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    eval_every: int,
    seed: int,
    train_step_fn,
    eval_fn,
    higher_is_better: bool = True,
) -> dict:
    """
    Generic training loop shared across case studies.

    Args:
        train_step_fn: callable(model, step) -> (loss: Tensor, metrics: dict[str, float])
            Called each step. Returns scalar loss and display metrics dict.
        eval_fn: callable(model) -> float
            Called every eval_every steps. Returns the validation score.
        higher_is_better: If True, maximize eval score; if False, minimize.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_score = -float("inf") if higher_is_better else float("inf")
    history: list[dict] = []
    ema_vals: dict[str, float] = {}
    tag = f"[{run_name}] " if run_name else ""

    def _is_better(score: float) -> bool:
        return score > best_score if higher_is_better else score < best_score

    bar = trange(1, steps + 1)
    for step in bar:
        model.train()
        loss, metrics = train_step_fn(model, step)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Update EMA for all metrics
        for k, v in metrics.items():
            ema_vals[k] = v if k not in ema_vals else 0.95 * ema_vals[k] + 0.05 * v

        if step % eval_every == 0 or step == steps:
            val_score = eval_fn(model)
            entry = {"step": step, "val_score": val_score, **metrics}
            history.append(entry)
            if _is_better(val_score):
                best_score = float(val_score)
                best_state = copy.deepcopy(model.state_dict())

            ema_str = " | ".join(f"EMA {k} {v:.3f}" for k, v in ema_vals.items())
            bar.set_description(
                f"{tag}Step {step} | Loss {loss.item():.4f} | {ema_str} | Val {val_score:.3f} | Grad {float(grad_norm):.2f}"
            )
        else:
            ema_str = " | ".join(f"EMA {k} {v:.3f}" for k, v in ema_vals.items())
            bar.set_description(f"{tag}Step {step} | Loss {loss.item():.4f} | {ema_str}")

    model.load_state_dict(best_state)
    return {"seed": seed, "best_val_score": best_score, "history_tail": history[-10:]}


def save_training_artifacts(
    output_dir: Path,
    *,
    model: nn.Module,
    dataset_config: dict,
    model_config: dict,
    training_config: dict,
    training_summary: dict,
    **extra_config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")
    config = {
        "dataset": dataset_config,
        **extra_config,
        "model": model_config,
        "training": training_config,
        "training_summary": training_summary,
    }
    save_json(output_dir / "experiment_config.json", config)


def load_model_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
    build_model_fn,
) -> tuple[nn.Module, dict]:
    cfg = load_json(checkpoint_dir / "experiment_config.json")
    model = build_model_fn(cfg["model"]).to(device)
    state_dict = torch.load(checkpoint_dir / "model.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg
