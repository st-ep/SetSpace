from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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
