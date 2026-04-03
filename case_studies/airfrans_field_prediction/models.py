from __future__ import annotations

import torch.nn as nn

from case_studies.point_cloud_consistency.models import PointCloudSetPredictor
from case_studies.point_cloud_consistency.pointnext import PointNeXtClassifier


class AirfRANSForceRegressor(PointCloudSetPredictor):
    def __init__(self, *, output_dim: int, **kwargs) -> None:
        super().__init__(output_dim=int(output_dim), **kwargs)
        self.task = "regression"


def build_force_regressor(
    *,
    value_input_dim: int,
    output_dim: int,
    activation_fn=nn.GELU,
    backbone: str = "set_encoder",
    pointnext_width: int = 32,
    pointnext_blocks: tuple[int, ...] = (1, 1, 1, 1, 1, 1),
    pointnext_strides: tuple[int, ...] = (1, 2, 2, 2, 2, 1),
    pointnext_radius: float = 0.15,
    pointnext_radius_scaling: float = 1.5,
    pointnext_nsample: int = 32,
    pointnext_expansion: int = 4,
    pointnext_sa_layers: int = 2,
    pointnext_sa_use_res: bool = True,
    pointnext_normalize_dp: bool = True,
    pointnext_head_hidden_dim: int = 256,
    **kwargs,
) -> nn.Module:
    backbone = str(backbone).lower()
    if backbone == "set_encoder":
        return AirfRANSForceRegressor(
            value_input_dim=value_input_dim,
            output_dim=output_dim,
            activation_fn=activation_fn,
            **kwargs,
        )
    if backbone == "pointnext":
        weight_mode = str(kwargs.get("weight_mode", "uniform")).lower()
        if weight_mode != "uniform":
            raise ValueError("AirfRANS PointNeXt uses the original uniform neighborhood reduction.")
        return PointNeXtClassifier(
            value_input_dim=int(value_input_dim),
            num_classes=int(output_dim),
            width=int(pointnext_width),
            blocks=tuple(int(v) for v in pointnext_blocks),
            strides=tuple(int(v) for v in pointnext_strides),
            radius=float(pointnext_radius),
            radius_scaling=float(pointnext_radius_scaling),
            nsample=int(pointnext_nsample),
            expansion=int(pointnext_expansion),
            sa_layers=int(pointnext_sa_layers),
            sa_use_res=bool(pointnext_sa_use_res),
            normalize_dp=bool(pointnext_normalize_dp),
            head_hidden_dim=int(pointnext_head_hidden_dim),
        )
    raise ValueError(f"Unsupported AirfRANS backbone: {backbone}")
