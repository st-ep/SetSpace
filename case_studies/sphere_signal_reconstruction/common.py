from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_studies.sphere_signal_reconstruction.dataset import SAMPLING_MODES, SphereSignalDataset
from case_studies.sphere_signal_reconstruction.models import SphereSignalReconstructor

DEFAULT_POINT_COUNTS = [32, 64, 128, 256, 512]
DEFAULT_EVAL_MODES = list(SAMPLING_MODES)


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


def build_dataset_from_config(dataset_config: dict) -> SphereSignalDataset:
    return SphereSignalDataset(**dataset_config)


def build_model_from_config(model_config: dict) -> SphereSignalReconstructor:
    activation_name = model_config.get("activation_fn", "gelu").lower()
    activation_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(activation_name, nn.GELU)
    return SphereSignalReconstructor(
        weight_mode=model_config["weight_mode"],
        n_basis=model_config["n_basis"],
        key_dim=model_config["key_dim"],
        value_dim=model_config["value_dim"],
        encoder_hidden_dim=model_config["encoder_hidden_dim"],
        trunk_hidden_dim=model_config["trunk_hidden_dim"],
        n_trunk_layers=model_config["n_trunk_layers"],
        activation_fn=activation_fn,
        basis_activation=model_config["basis_activation"],
        value_mode=model_config["value_mode"],
        encoder_normalize=model_config["encoder_normalize"],
        use_deeponet_bias=model_config["use_deeponet_bias"],
        knn_k=model_config["knn_k"],
        intrinsic_dim=model_config["intrinsic_dim"],
    )
