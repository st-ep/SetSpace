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

from case_studies.point_cloud_consistency.models import PointCloudSetClassifier
from case_studies.scanobjectnn_consistency.dataset import SAMPLING_MODES, ScanObjectNNConsistencyDataset

DEFAULT_POINT_COUNTS = [32, 64, 128, 256, 512, 1024]
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


def build_dataset_from_config(dataset_config: dict) -> ScanObjectNNConsistencyDataset:
    return ScanObjectNNConsistencyDataset(**dataset_config)


def build_model_from_config(model_config: dict) -> PointCloudSetClassifier:
    activation_name = model_config.get("activation_fn", "gelu").lower()
    activation_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(activation_name, nn.GELU)
    return PointCloudSetClassifier(
        value_input_dim=model_config["value_input_dim"],
        num_classes=model_config["num_classes"],
        n_tokens=model_config["n_tokens"],
        token_dim=model_config["token_dim"],
        key_dim=model_config["key_dim"],
        hidden_dim=model_config["hidden_dim"],
        activation_fn=activation_fn,
        basis_activation=model_config["basis_activation"],
        value_mode=model_config["value_mode"],
        normalize=model_config["normalize"],
        weight_mode=model_config["weight_mode"],
        knn_k=model_config["knn_k"],
        intrinsic_dim=model_config["intrinsic_dim"],
    )
