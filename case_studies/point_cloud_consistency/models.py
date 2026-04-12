from __future__ import annotations

import torch
import torch.nn as nn

from case_studies.point_cloud_consistency.pointnext import PointNeXtClassifier
from set_encoders import (
    WeightedSetEncoder,
    infer_knn_density_weights,
    infer_spherical_voronoi_weights,
    infer_uniform_weights,
)


def _canonical_weight_mode(weight_mode: str) -> str:
    normalized = str(weight_mode).lower()
    return "voronoi" if normalized == "voronoi_oracle" else normalized


def _infer_point_weights(
    *,
    coords: torch.Tensor,
    weight_mode: str,
    knn_k: int,
    intrinsic_dim: int,
    point_mask: torch.Tensor | None = None,
    point_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weight_mode = _canonical_weight_mode(weight_mode)
    if point_weights is not None:
        return point_weights.to(device=coords.device, dtype=coords.dtype)
    if weight_mode == "uniform":
        return infer_uniform_weights(coords, point_mask)
    if weight_mode == "oracle_density":
        raise ValueError("oracle_density weight_mode requires explicit point_weights from the dataset.")
    if weight_mode == "voronoi":
        return infer_spherical_voronoi_weights(
            coords,
            sensor_mask=point_mask,
            normalize=True,
        ).to(dtype=coords.dtype)

    return infer_knn_density_weights(
        coords,
        sensor_mask=point_mask,
        k=knn_k,
        intrinsic_dim=intrinsic_dim,
        normalize=True,
    ).to(dtype=coords.dtype)


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

        self.weight_mode = _canonical_weight_mode(weight_mode)
        self.knn_k = int(knn_k)
        self.intrinsic_dim = int(intrinsic_dim)
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.output_dim = int(output_dim)
        self.value_mode = value_mode.lower()
        self.task = "classification" if int(output_dim) > 1 else "regression"

        if self.weight_mode not in ["uniform", "knn", "oracle_density", "voronoi"]:
            raise ValueError(
                f"weight_mode must be 'uniform', 'knn', 'oracle_density', or 'voronoi', got {weight_mode}"
            )

        self.encoder = WeightedSetEncoder(
            n_tokens=self.n_tokens,
            coord_dim=3,
            value_coord_dim=(3 if self.value_mode == "mlp_xu" else None),
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
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _infer_point_weights(
            coords=coords,
            weight_mode=self.weight_mode,
            knn_k=self.knn_k,
            intrinsic_dim=self.intrinsic_dim,
            point_mask=point_mask,
            point_weights=point_weights,
        )

    def encode_tokens(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weights = self._infer_weights(coords, point_mask=point_mask, point_weights=point_weights)
        return self.encoder(
            coords,
            values,
            element_mask=point_mask,
            element_weights=weights,
            value_coords=(coords if self.value_mode == "mlp_xu" else None),
        )

    def embed(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_tokens(coords, values, point_mask=point_mask, point_weights=point_weights).flatten(1)

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.embed(coords, values, point_mask=point_mask, point_weights=point_weights))


class PointCloudSetClassifier(PointCloudSetPredictor):
    def __init__(self, *, num_classes: int = 2, **kwargs) -> None:
        super().__init__(output_dim=num_classes, **kwargs)
        self.task = "classification"


class PointCloudMeanRegressor(PointCloudSetPredictor):
    def __init__(self, **kwargs) -> None:
        super().__init__(output_dim=1, **kwargs)
        self.task = "regression"

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            coords,
            values,
            point_mask=point_mask,
            point_weights=point_weights,
        ).squeeze(-1)


class PointCloudWeightedMeanRegressor(nn.Module):
    def __init__(
        self,
        *,
        value_input_dim: int = 1,
        weight_mode: str = "uniform",
        knn_k: int = 8,
        intrinsic_dim: int = 2,
        eps: float = 1e-8,
        **_kwargs,
    ) -> None:
        super().__init__()

        self.value_input_dim = int(value_input_dim)
        self.weight_mode = _canonical_weight_mode(weight_mode)
        self.knn_k = int(knn_k)
        self.intrinsic_dim = int(intrinsic_dim)
        self.eps = float(eps)
        self.task = "regression"

        if self.value_input_dim != 1:
            raise ValueError(
                f"PointCloudWeightedMeanRegressor expects value_input_dim=1, got {value_input_dim}."
            )
        if self.weight_mode not in ["uniform", "knn", "oracle_density", "voronoi"]:
            raise ValueError(
                f"weight_mode must be 'uniform', 'knn', 'oracle_density', or 'voronoi', got {weight_mode}"
            )

    def _infer_weights(
        self,
        coords: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _infer_point_weights(
            coords=coords,
            weight_mode=self.weight_mode,
            knn_k=self.knn_k,
            intrinsic_dim=self.intrinsic_dim,
            point_mask=point_mask,
            point_weights=point_weights,
        )

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values.dim() != 3 or values.shape[-1] != 1:
            raise ValueError(f"values must be shaped (B, N, 1), got {values.shape=}")
        weights = self._infer_weights(coords, point_mask=point_mask, point_weights=point_weights)
        weighted_values = weights * values.squeeze(-1)
        denom = weights.sum(dim=1).clamp_min(self.eps)
        return weighted_values.sum(dim=1) / denom


class PointNeXtRegressor(nn.Module):
    def __init__(
        self,
        *,
        value_input_dim: int = 1,
        output_dim: int = 1,
        width: int = 32,
        blocks: tuple[int, ...] = (1, 1, 1, 1, 1, 1),
        strides: tuple[int, ...] = (1, 2, 2, 2, 2, 1),
        radius: float = 0.15,
        radius_scaling: float = 1.5,
        nsample: int = 32,
        expansion: int = 4,
        sa_layers: int = 2,
        sa_use_res: bool = True,
        normalize_dp: bool = True,
        head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.task = "regression"
        self.backbone = PointNeXtClassifier(
            value_input_dim=int(value_input_dim),
            num_classes=self.output_dim,
            width=int(width),
            blocks=tuple(int(v) for v in blocks),
            strides=tuple(int(v) for v in strides),
            radius=float(radius),
            radius_scaling=float(radius_scaling),
            nsample=int(nsample),
            expansion=int(expansion),
            sa_layers=int(sa_layers),
            sa_use_res=bool(sa_use_res),
            normalize_dp=bool(normalize_dp),
            head_hidden_dim=int(head_hidden_dim),
        )

    def embed(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone.embed(coords, values, point_mask=point_mask)

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        point_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del point_weights
        out = self.backbone(coords, values, point_mask=point_mask)
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out


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
        weight_mode = kwargs.get("weight_mode", "uniform")
        if str(weight_mode).lower() != "uniform":
            raise ValueError("PointNeXtClassifier only supports the original uniform neighborhood reduction.")
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
        )
    raise ValueError(f"Unsupported classifier backbone: {backbone}")


def build_point_cloud_regressor(
    *,
    backbone: str = "set_encoder",
    output_dim: int = 1,
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
        if int(output_dim) != 1:
            raise ValueError(f"PointCloudMeanRegressor expects output_dim=1, got {output_dim}.")
        return PointCloudMeanRegressor(
            activation_fn=activation_fn,
            **kwargs,
        )
    if backbone == "weighted_mean":
        if int(output_dim) != 1:
            raise ValueError(f"PointCloudWeightedMeanRegressor expects output_dim=1, got {output_dim}.")
        return PointCloudWeightedMeanRegressor(**kwargs)
    if backbone == "pointnext":
        weight_mode = kwargs.get("weight_mode", "uniform")
        if str(weight_mode).lower() != "uniform":
            raise ValueError("PointNeXtRegressor only supports the original uniform neighborhood reduction.")
        return PointNeXtRegressor(
            value_input_dim=kwargs.get("value_input_dim", 1),
            output_dim=int(output_dim),
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
    raise ValueError(f"Unsupported regressor backbone: {backbone}")
