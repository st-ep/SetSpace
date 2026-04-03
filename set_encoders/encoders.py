from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import _coerce_mask, _coerce_weights


class WeightedSetEncoder(nn.Module):
    """
    Geometry-aware additive encoder for sampled sets.

    The encoder learns test functions over coordinates and uses them to pool
    element values with optional geometry-derived weights:

        z_k = Σ_i w_i φ_k(x_i) V(x_i, v_i)

    This is intended for settings where set elements are samples from an
    underlying continuum object and aggregation should remain stable under
    discretization changes.
    """

    def __init__(
        self,
        *,
        n_tokens: int,
        coord_dim: int,
        value_coord_dim: int | None = None,
        value_input_dim: int,
        output_dim: int,
        key_dim: int,
        value_dim: int,
        hidden_dim: int,
        activation_fn: type[nn.Module],
        key_hidden_dim: int | None = None,
        key_layers: int = 3,
        basis_activation: str = "tanh",
        value_mode: str = "linear_u",
        eps: float = 1e-8,
        normalize: str = "total",
        learn_temperature: bool = False,
    ) -> None:
        super().__init__()

        self.n_tokens = int(n_tokens)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.output_dim = int(output_dim)
        self.coord_dim = int(coord_dim)
        self.value_coord_dim = int(coord_dim if value_coord_dim is None else value_coord_dim)
        self.eps = float(eps)
        self.normalize = normalize.lower()
        self.learn_temperature = bool(learn_temperature)
        self.basis_activation = basis_activation.lower()
        self.value_mode = value_mode.lower()

        if self.normalize not in ["none", "total", "token"]:
            raise ValueError(f"normalize must be 'none', 'total', or 'token', got {normalize}")
        if self.basis_activation not in ["tanh", "softsign", "softplus"]:
            raise ValueError(
                f"basis_activation must be 'tanh', 'softsign', or 'softplus', got {basis_activation}"
            )
        if self.value_mode not in ["linear_u", "mlp_u", "mlp_xu"]:
            raise ValueError(
                f"value_mode must be 'linear_u', 'mlp_u', or 'mlp_xu', got {value_mode}"
            )

        key_hidden_dim = int(hidden_dim if key_hidden_dim is None else key_hidden_dim)
        key_layers = int(key_layers)
        if key_layers < 2:
            raise ValueError(f"key_layers must be >= 2, got {key_layers}")

        key_layers_list = [nn.Linear(coord_dim, key_hidden_dim), activation_fn()]
        for _ in range(key_layers - 2):
            key_layers_list.append(nn.Linear(key_hidden_dim, key_hidden_dim))
            key_layers_list.append(activation_fn())
        key_layers_list.append(nn.Linear(key_hidden_dim, key_dim))
        self.key_net = nn.Sequential(*key_layers_list)

        if self.value_mode == "linear_u":
            self.value_net = nn.Linear(value_input_dim, value_dim)
            self._value_includes_x = False
        elif self.value_mode == "mlp_u":
            self.value_net = nn.Sequential(
                nn.Linear(value_input_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, value_dim),
            )
            self._value_includes_x = False
        else:
            self.value_net = nn.Sequential(
                nn.Linear(self.value_coord_dim + value_input_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, value_dim),
            )
            self._value_includes_x = True

        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, key_dim))

        if self.learn_temperature:
            self.log_tau = nn.Parameter(torch.zeros(1))
        else:
            self.log_tau = None

        self.rho_token = (
            nn.Identity()
            if value_dim == output_dim
            else nn.Sequential(
                nn.Linear(value_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, output_dim),
            )
        )

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        element_mask: torch.Tensor | None = None,
        element_weights: torch.Tensor | None = None,
        value_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if coords.dim() != 3 or values.dim() != 3:
            raise ValueError(f"coords and values must be 3D, got {coords.shape=} and {values.shape=}")
        if coords.shape[:2] != values.shape[:2]:
            raise ValueError(
                f"coords and values must share (B, N), got {coords.shape=} and {values.shape=}"
            )

        batch_size, n_elements = coords.shape[0], coords.shape[1]
        element_mask = _coerce_mask(coords, element_mask)
        element_weights = _coerce_weights(coords, element_weights)

        keys = self.key_net(coords)
        if self._value_includes_x:
            if value_coords is None:
                value_coords = coords
            if value_coords.dim() != 3 or value_coords.shape[:2] != values.shape[:2]:
                raise ValueError(
                    f"value_coords must be 3D and share (B, N) with values, got {value_coords.shape=} and {values.shape=}"
                )
            if value_coords.shape[-1] != self.value_coord_dim:
                raise ValueError(
                    f"value_coords last dimension must be {self.value_coord_dim}, got {value_coords.shape[-1]}"
                )
            encoded_values = self.value_net(torch.cat([value_coords, values], dim=-1))
        else:
            encoded_values = self.value_net(values)

        queries = self.query_tokens.expand(batch_size, -1, -1)
        scores = torch.einsum("bpk,bnk->bpn", queries, keys) / math.sqrt(self.key_dim)
        if self.learn_temperature:
            tau = torch.exp(self.log_tau) + self.eps
            scores = scores / tau

        if self.basis_activation == "tanh":
            basis_values = torch.tanh(scores)
        elif self.basis_activation == "softsign":
            basis_values = scores / (1.0 + scores.abs())
        else:
            basis_values = F.softplus(scores)

        if element_weights is None:
            weights = torch.ones((batch_size, n_elements), device=coords.device, dtype=coords.dtype)
        else:
            weights = element_weights.to(dtype=coords.dtype)

        if element_mask is not None:
            mask = element_mask.to(dtype=coords.dtype)
            weights = weights * mask
            encoded_values = encoded_values * mask.unsqueeze(-1)
            basis_values = basis_values * mask.unsqueeze(1)

        pooled = torch.einsum("bpn,bn,bnd->bpd", basis_values, weights, encoded_values)

        if self.normalize == "total":
            denom = weights.sum(dim=1).clamp_min(self.eps)
            pooled = pooled / denom.view(batch_size, 1, 1)
        elif self.normalize == "token":
            mass = torch.einsum("bpn,bn->bp", basis_values, weights).clamp_min(self.eps)
            pooled = pooled / mass.unsqueeze(-1)
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize option: {self.normalize}")

        return self.rho_token(pooled)
