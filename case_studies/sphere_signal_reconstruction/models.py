from __future__ import annotations

import torch
import torch.nn as nn

from set_encoders import (
    SetEncoderOperator,
    infer_knn_density_weights,
    infer_uniform_weights,
)


class SphereSignalReconstructor(nn.Module):
    def __init__(
        self,
        *,
        weight_mode: str = "uniform",
        n_basis: int = 32,
        key_dim: int = 64,
        value_dim: int = 64,
        encoder_hidden_dim: int = 128,
        trunk_hidden_dim: int = 128,
        n_trunk_layers: int = 4,
        activation_fn=nn.GELU,
        basis_activation: str = "softplus",
        value_mode: str = "mlp_xu",
        encoder_normalize: str = "total",
        use_deeponet_bias: bool = True,
        knn_k: int = 8,
        intrinsic_dim: int = 2,
    ) -> None:
        super().__init__()

        self.weight_mode = weight_mode.lower()
        self.knn_k = int(knn_k)
        self.intrinsic_dim = int(intrinsic_dim)
        self.value_mode = value_mode.lower()

        if self.weight_mode not in {"uniform", "knn", "oracle_density"}:
            raise ValueError(f"weight_mode must be 'uniform', 'knn', or 'oracle_density', got {weight_mode}")

        self.operator = SetEncoderOperator(
            input_size_src=3,
            output_size_src=1,
            input_size_tgt=3,
            output_size_tgt=1,
            value_coord_dim=(3 if self.value_mode == "mlp_xu" else None),
            p=int(n_basis),
            rho_hidden_size=int(encoder_hidden_dim),
            trunk_hidden_size=int(trunk_hidden_dim),
            n_trunk_layers=int(n_trunk_layers),
            activation_fn=activation_fn,
            use_deeponet_bias=bool(use_deeponet_bias),
            phi_output_size=max(int(key_dim), int(value_dim)),
            use_positional_encoding=False,
            key_dim=int(key_dim),
            value_dim=int(value_dim),
            basis_activation=basis_activation,
            value_mode=value_mode,
            encoder_normalize=encoder_normalize,
            uniform_sensor_weights=False,
        )

    def _infer_sensor_weights(
        self,
        obs_coords: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sensor_weights is not None:
            return sensor_weights.to(device=obs_coords.device, dtype=obs_coords.dtype)
        if self.weight_mode == "uniform":
            return infer_uniform_weights(obs_coords, point_mask)
        if self.weight_mode == "oracle_density":
            raise ValueError("oracle_density weight_mode requires explicit sensor_weights from the dataset.")

        return infer_knn_density_weights(
            obs_coords,
            sensor_mask=point_mask,
            k=self.knn_k,
            intrinsic_dim=self.intrinsic_dim,
            normalize=True,
        ).to(dtype=obs_coords.dtype)

    def encode_observations(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sensor_weights = self._infer_sensor_weights(
            obs_coords,
            point_mask=point_mask,
            sensor_weights=sensor_weights,
        )
        return self.operator.forward_branch(
            obs_coords,
            obs_values,
            sensor_mask=point_mask,
            sensor_weights=sensor_weights,
            value_xs=(obs_coords if self.value_mode == "mlp_xu" else None),
        )

    def forward(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        query_coords: torch.Tensor,
        point_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sensor_weights = self._infer_sensor_weights(
            obs_coords,
            point_mask=point_mask,
            sensor_weights=sensor_weights,
        )
        out = self.operator(
            obs_coords,
            obs_values,
            query_coords,
            sensor_mask=point_mask,
            sensor_weights=sensor_weights,
            value_xs=(obs_coords if self.value_mode == "mlp_xu" else None),
        )
        return out.squeeze(-1)
