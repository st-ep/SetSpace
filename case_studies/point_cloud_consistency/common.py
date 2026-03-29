from __future__ import annotations

import torch
import torch.nn as nn

from case_studies.point_cloud_consistency.dataset import SAMPLING_MODES, SyntheticSurfaceSignalDataset
from case_studies.point_cloud_consistency.models import PointCloudMeanRegressor, PointCloudSetClassifier
from case_studies.shared import get_activation, load_json, save_json, set_random_seed

DEFAULT_POINT_COUNTS = [32, 64, 128, 256, 512]
DEFAULT_EVAL_MODES = list(SAMPLING_MODES)


def build_dataset_from_config(dataset_config: dict) -> SyntheticSurfaceSignalDataset:
    return SyntheticSurfaceSignalDataset(**dataset_config)


def build_model_from_config(model_config: dict) -> torch.nn.Module:
    activation_fn = get_activation(model_config.get("activation_fn", "gelu"))
    common_kwargs = {
        "value_input_dim": model_config["value_input_dim"],
        "n_tokens": model_config["n_tokens"],
        "token_dim": model_config["token_dim"],
        "key_dim": model_config["key_dim"],
        "hidden_dim": model_config["hidden_dim"],
        "activation_fn": activation_fn,
        "basis_activation": model_config["basis_activation"],
        "value_mode": model_config["value_mode"],
        "normalize": model_config["normalize"],
        "weight_mode": model_config["weight_mode"],
        "knn_k": model_config["knn_k"],
        "intrinsic_dim": model_config["intrinsic_dim"],
    }
    task = model_config.get("task", "classification").lower()
    if task == "regression":
        return PointCloudMeanRegressor(**common_kwargs)
    return PointCloudSetClassifier(num_classes=model_config["num_classes"], **common_kwargs)
