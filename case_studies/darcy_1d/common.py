from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from set_encoders import SetEncoderOperator

CHECKPOINTS = REPO_ROOT / "checkpoints"


def get_activation(name: str | None):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "swish": nn.SiLU}.get(
        (name or "relu").lower(),
        nn.ReLU,
    )


def load_json(path: os.PathLike | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_operator_from_config(
    device: torch.device,
    checkpoint_dir: Path,
    *,
    uniform_sensor_weights: bool,
) -> SetEncoderOperator:
    cfg = load_json(checkpoint_dir / "experiment_config.json")["model_architecture"]
    model = SetEncoderOperator(
        input_size_src=1,
        output_size_src=1,
        input_size_tgt=1,
        output_size_tgt=1,
        p=cfg["son_p_dim"],
        phi_hidden_size=cfg["son_phi_hidden"],
        rho_hidden_size=cfg["son_rho_hidden"],
        trunk_hidden_size=cfg["son_trunk_hidden"],
        n_trunk_layers=cfg["son_n_trunk_layers"],
        activation_fn=get_activation(cfg.get("activation_fn")),
        use_deeponet_bias=True,
        phi_output_size=cfg["son_phi_output_size"],
        initial_lr=5e-4,
        use_positional_encoding=cfg["use_positional_encoding"],
        pos_encoding_type=cfg["pos_encoding_type"],
        pos_encoding_dim=cfg["pos_encoding_dim"],
        pos_encoding_max_freq=cfg["pos_encoding_max_freq"],
        key_dim=64,
        basis_activation="softplus",
        value_mode="linear_u",
        uniform_sensor_weights=uniform_sensor_weights,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "darcy1d_setonet_model.pth", map_location=device))
    model.eval()
    return model


def build_geometry_aware_model(device: torch.device) -> SetEncoderOperator:
    return build_operator_from_config(
        device,
        CHECKPOINTS / "setonet_key_trapezoidal",
        uniform_sensor_weights=False,
    )


def build_uniform_model(device: torch.device) -> SetEncoderOperator:
    return build_operator_from_config(
        device,
        CHECKPOINTS / "setonet_key_uniform",
        uniform_sensor_weights=True,
    )


def build_trapezoidal_model(device: torch.device) -> SetEncoderOperator:
    return build_geometry_aware_model(device)
