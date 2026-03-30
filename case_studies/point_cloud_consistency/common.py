from __future__ import annotations

import torch
import torch.nn as nn

from case_studies.point_cloud_consistency.dataset import SAMPLING_MODES, SyntheticSurfaceSignalDataset
from case_studies.point_cloud_consistency.models import (
    PointCloudMeanRegressor,
    build_point_cloud_classifier,
)
from case_studies.shared import get_activation, load_json, save_json, set_random_seed

DEFAULT_POINT_COUNTS = [32, 64, 128, 256, 512]
DEFAULT_EVAL_MODES = list(SAMPLING_MODES)


def build_dataset_from_config(dataset_config: dict) -> SyntheticSurfaceSignalDataset:
    return SyntheticSurfaceSignalDataset(**dataset_config)


def build_model_from_config(model_config: dict) -> torch.nn.Module:
    activation_fn = get_activation(model_config.get("activation_fn", "gelu"))
    backbone = model_config.get("backbone", "set_encoder")
    shared_kwargs = {
        "value_input_dim": model_config["value_input_dim"],
        "activation_fn": activation_fn,
        "weight_mode": model_config["weight_mode"],
        "knn_k": model_config["knn_k"],
        "intrinsic_dim": model_config["intrinsic_dim"],
    }
    task = model_config.get("task", "classification").lower()
    if task == "regression":
        if backbone != "set_encoder":
            raise ValueError(f"Regression currently supports only the set_encoder backbone, got {backbone}.")
        return PointCloudMeanRegressor(
            **shared_kwargs,
            n_tokens=model_config.get("n_tokens", 16),
            token_dim=model_config.get("token_dim", 32),
            key_dim=model_config.get("key_dim", 64),
            hidden_dim=model_config.get("hidden_dim", 128),
            basis_activation=model_config.get("basis_activation", "softplus"),
            value_mode=model_config.get("value_mode", "mlp_xu"),
            normalize=model_config.get("normalize", "total"),
        )
    classifier_kwargs = {
        **shared_kwargs,
        "backbone": backbone,
        "pointnext_width": model_config.get("pointnext_width", 32),
        "pointnext_blocks": tuple(model_config.get("pointnext_blocks", [1, 1, 1, 1, 1, 1])),
        "pointnext_strides": tuple(model_config.get("pointnext_strides", [1, 2, 2, 2, 2, 1])),
        "pointnext_radius": model_config.get("pointnext_radius", 0.15),
        "pointnext_radius_scaling": model_config.get("pointnext_radius_scaling", 1.5),
        "pointnext_nsample": model_config.get("pointnext_nsample", 32),
        "pointnext_expansion": model_config.get("pointnext_expansion", 4),
        "pointnext_sa_layers": model_config.get("pointnext_sa_layers", 2),
        "pointnext_sa_use_res": model_config.get("pointnext_sa_use_res", True),
        "pointnext_normalize_dp": model_config.get("pointnext_normalize_dp", True),
        "pointnext_head_hidden_dim": model_config.get("pointnext_head_hidden_dim", 256),
    }
    set_encoder_kwargs = {
        "n_tokens": model_config.get("n_tokens", 16),
        "token_dim": model_config.get("token_dim", 32),
        "key_dim": model_config.get("key_dim", 64),
        "hidden_dim": model_config.get("hidden_dim", 128),
        "basis_activation": model_config.get("basis_activation", "softplus"),
        "value_mode": model_config.get("value_mode", "mlp_xu"),
        "normalize": model_config.get("normalize", "total"),
    }
    return build_point_cloud_classifier(
        num_classes=model_config["num_classes"],
        **classifier_kwargs,
        **set_encoder_kwargs,
    )
