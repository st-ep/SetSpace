from __future__ import annotations

import torch

from case_studies.ahmedml_surface_forces.dataset import AhmedMLSurfaceForceDataset, SAMPLING_MODES
from case_studies.ahmedml_surface_forces.models import build_force_regressor
from case_studies.shared import get_activation, load_json, save_json, set_random_seed

DEFAULT_POINT_COUNTS = [64, 128, 256, 512, 1024]
DEFAULT_EVAL_MODES = list(SAMPLING_MODES)


def build_dataset_from_config(dataset_config: dict) -> AhmedMLSurfaceForceDataset:
    return AhmedMLSurfaceForceDataset(
        processed_root=dataset_config["processed_root"],
        n_train=dataset_config.get("n_train"),
        n_val=dataset_config.get("n_val"),
        n_test=dataset_config.get("n_test"),
        seed=dataset_config.get("seed", 0),
    )


def build_model_from_config(model_config: dict) -> torch.nn.Module:
    activation_fn = get_activation(model_config.get("activation_fn", "gelu"))
    return build_force_regressor(
        value_input_dim=model_config["value_input_dim"],
        output_dim=model_config["output_dim"],
        activation_fn=activation_fn,
        backbone=model_config.get("backbone", "set_encoder"),
        n_tokens=model_config.get("n_tokens", 16),
        token_dim=model_config.get("token_dim", 32),
        key_dim=model_config.get("key_dim", 64),
        hidden_dim=model_config.get("hidden_dim", 128),
        basis_activation=model_config.get("basis_activation", "softplus"),
        value_mode=model_config.get("value_mode", "mlp_xu"),
        normalize=model_config.get("normalize", "total"),
        weight_mode=model_config.get("weight_mode", "uniform"),
        knn_k=model_config.get("knn_k", 8),
        intrinsic_dim=model_config.get("intrinsic_dim", 2),
        pointnext_width=model_config.get("pointnext_width", 32),
        pointnext_blocks=tuple(model_config.get("pointnext_blocks", [1, 1, 1, 1, 1, 1])),
        pointnext_strides=tuple(model_config.get("pointnext_strides", [1, 2, 2, 2, 2, 1])),
        pointnext_radius=model_config.get("pointnext_radius", 0.15),
        pointnext_radius_scaling=model_config.get("pointnext_radius_scaling", 1.5),
        pointnext_nsample=model_config.get("pointnext_nsample", 32),
        pointnext_expansion=model_config.get("pointnext_expansion", 4),
        pointnext_sa_layers=model_config.get("pointnext_sa_layers", 2),
        pointnext_sa_use_res=model_config.get("pointnext_sa_use_res", True),
        pointnext_normalize_dp=model_config.get("pointnext_normalize_dp", True),
        pointnext_head_hidden_dim=model_config.get("pointnext_head_hidden_dim", 256),
    )
