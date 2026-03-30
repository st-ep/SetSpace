from __future__ import annotations

import torch
import torch.nn as nn

from case_studies.point_cloud_consistency.pointnext import PointNeXtClassifier
from set_encoders import WeightedSetEncoder, infer_knn_density_weights, infer_uniform_weights


class PointCloudSetPredictor(nn.Module):
    def __init__(
        self,
        *,
        value_input_dim: int = 1,
        output_dim: int,
        n_tokens: int = 16,
        token_dim: int = 32,
        key_dim: int = 64,
        hidden_dim: int = 128,
        activation_fn=nn.GELU,
        basis_activation: str = "softplus",
        value_mode: str = "mlp_xu",
        normalize: str = "total",
        weight_mode: str = "uniform",
        knn_k: int = 8,
        intrinsic_dim: int = 2,
    ) -> None:
        super().__init__()

        self.weight_mode = weight_mode.lower()
        self.knn_k = int(knn_k)
        self.intrinsic_dim = int(intrinsic_dim)
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)

        if self.weight_mode not in ["uniform", "knn"]:
            raise ValueError(f"weight_mode must be 'uniform' or 'knn', got {weight_mode}")

        self.encoder = WeightedSetEncoder(
            n_tokens=self.n_tokens,
            coord_dim=3,
            value_input_dim=value_input_dim,
            output_dim=self.token_dim,
            key_dim=key_dim,
            value_dim=self.token_dim,
            hidden_dim=hidden_dim,
            activation_fn=activation_fn,
            basis_activation=basis_activation,
            value_mode=value_mode,
            normalize=normalize,
        )

        flattened_dim = self.n_tokens * self.token_dim
        self.head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def _infer_weights(
        self,
        coords: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.weight_mode == "uniform":
            return infer_uniform_weights(coords, point_mask)

        return infer_knn_density_weights(
            coords,
            sensor_mask=point_mask,
            k=self.knn_k,
            intrinsic_dim=self.intrinsic_dim,
            normalize=True,
        ).to(dtype=coords.dtype)

    def encode_tokens(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = self._infer_weights(coords, point_mask=point_mask)
        return self.encoder(
            coords,
            values,
            element_mask=point_mask,
            element_weights=weights,
        )

    def embed(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_tokens(coords, values, point_mask=point_mask).flatten(1)

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.embed(coords, values, point_mask=point_mask))


class PointCloudSetClassifier(PointCloudSetPredictor):
    def __init__(self, *, num_classes: int = 2, **kwargs) -> None:
        super().__init__(output_dim=num_classes, **kwargs)


class PointCloudMeanRegressor(PointCloudSetPredictor):
    def __init__(self, **kwargs) -> None:
        super().__init__(output_dim=1, **kwargs)

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(coords, values, point_mask=point_mask).squeeze(-1)


def build_point_cloud_classifier(
    *,
    backbone: str = "set_encoder",
    activation_fn=nn.GELU,
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
    backbone = backbone.lower()
    if backbone == "set_encoder":
        return PointCloudSetClassifier(
            activation_fn=activation_fn,
            **kwargs,
        )
    if backbone == "pointnext":
        return PointNeXtClassifier(
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
            num_classes=kwargs["num_classes"],
            value_input_dim=kwargs.get("value_input_dim", 1),
            weight_mode=kwargs.get("weight_mode", "uniform"),
            density_k=kwargs.get("knn_k", 8),
            intrinsic_dim=kwargs.get("intrinsic_dim", 2),
        )
    raise ValueError(f"Unsupported classifier backbone: {backbone}")
