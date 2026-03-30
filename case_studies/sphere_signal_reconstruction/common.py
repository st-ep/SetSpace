from __future__ import annotations

from case_studies.shared import get_activation, load_json, save_json, set_random_seed
from case_studies.sphere_signal_reconstruction.dataset import SAMPLING_MODES, SphereSignalDataset
from case_studies.sphere_signal_reconstruction.models import SphereSignalReconstructor

DEFAULT_POINT_COUNTS = [32, 64, 128, 256, 512]
DEFAULT_EVAL_MODES = list(SAMPLING_MODES)


def build_dataset_from_config(dataset_config: dict) -> SphereSignalDataset:
    return SphereSignalDataset(**dataset_config)


def build_model_from_config(model_config: dict) -> SphereSignalReconstructor:
    activation_fn = get_activation(model_config.get("activation_fn", "gelu"))
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
        mmq_anchor_ratio=model_config.get("mmq_anchor_ratio", 0.125),
        mmq_max_anchors=model_config.get("mmq_max_anchors", 32),
        mmq_patch_k=model_config.get("mmq_patch_k", 16),
        mmq_tangent_k=model_config.get("mmq_tangent_k", 16),
        mmq_rank_tol=model_config.get("mmq_rank_tol", 1e-6),
    )
