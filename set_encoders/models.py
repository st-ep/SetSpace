from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import trange

from .encoders import WeightedSetEncoder
from .utils import calculate_l2_relative_error
from .weights import infer_quadrature_weights, infer_uniform_weights


class SetEncoderOperator(nn.Module):
    """
    A minimal operator-learning model built around a weighted set encoder.

    The branch path encodes sampled source observations as a set, while the
    trunk path maps query coordinates to basis coefficients.
    """

    def __init__(
        self,
        input_size_src: int,
        output_size_src: int,
        input_size_tgt: int,
        output_size_tgt: int,
        value_coord_dim: int | None = None,
        p: int = 32,
        phi_hidden_size: int = 256,
        rho_hidden_size: int = 256,
        trunk_hidden_size: int = 256,
        n_trunk_layers: int = 4,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_deeponet_bias: bool = True,
        phi_output_size: int = 128,
        initial_lr: float = 5e-4,
        lr_schedule_steps=None,
        lr_schedule_gammas=None,
        use_positional_encoding: bool = True,
        pos_encoding_dim: int = 64,
        pos_encoding_type: str = "sinusoidal",
        pos_encoding_max_freq: float = 0.1,
        key_dim: int | None = None,
        value_dim: int | None = None,
        key_hidden_dim: int | None = None,
        key_layers: int = 3,
        basis_activation: str = "softplus",
        value_mode: str = "linear_u",
        encoder_normalize: str = "total",
        learn_temperature: bool = False,
        uniform_sensor_weights: bool = False,
    ):
        super().__init__()

        self.input_size_src = input_size_src
        self.output_size_src = output_size_src
        self.input_size_tgt = input_size_tgt
        self.output_size_tgt = output_size_tgt
        self.value_coord_dim = value_coord_dim
        self.use_positional_encoding = use_positional_encoding and pos_encoding_type != "skip"
        self.pos_encoding_dim = pos_encoding_dim if self.use_positional_encoding else 0
        self.pos_encoding_type = pos_encoding_type
        self.pos_encoding_max_freq = pos_encoding_max_freq
        self.p = p
        self.phi_hidden_size = phi_hidden_size
        self.phi_output_size = phi_output_size
        self.rho_hidden_size = rho_hidden_size
        self.trunk_hidden_size = trunk_hidden_size
        self.n_trunk_layers = n_trunk_layers
        self.initial_lr = initial_lr
        self.lr_schedule_steps = None
        self.lr_schedule_rates = None
        self.lr_schedule_gammas = None
        self.uniform_sensor_weights = uniform_sensor_weights

        if self.pos_encoding_type not in ["sinusoidal", "skip"]:
            raise ValueError(f"Unknown pos_encoding_type: {self.pos_encoding_type}")
        if self.use_positional_encoding and self.pos_encoding_dim % (2 * self.input_size_src) != 0:
            raise ValueError(
                f"pos_encoding_dim ({self.pos_encoding_dim}) must be divisible by 2 * input_size_src ({2 * self.input_size_src})."
            )

        if lr_schedule_steps is not None:
            if lr_schedule_gammas is None or len(lr_schedule_steps) != len(lr_schedule_gammas):
                raise ValueError("lr_schedule_gammas must match lr_schedule_steps in length.")
            self.lr_schedule_steps = sorted(lr_schedule_steps)
            self.lr_schedule_gammas = lr_schedule_gammas
            self.lr_schedule_rates = [initial_lr]
            current_lr = initial_lr
            for gamma in lr_schedule_gammas:
                current_lr *= gamma
                self.lr_schedule_rates.append(current_lr)

        encoded_coord_dim = self.pos_encoding_dim if self.use_positional_encoding else self.input_size_src
        key_dim = key_dim if key_dim is not None else self.phi_output_size
        value_dim = value_dim if value_dim is not None else self.phi_output_size
        self._set_encoder = WeightedSetEncoder(
            n_tokens=self.p,
            coord_dim=encoded_coord_dim,
            value_coord_dim=(encoded_coord_dim if value_coord_dim is None else int(value_coord_dim)),
            value_input_dim=self.output_size_src,
            output_dim=self.output_size_tgt,
            key_dim=key_dim,
            value_dim=value_dim,
            hidden_dim=self.rho_hidden_size,
            activation_fn=activation_fn,
            key_hidden_dim=key_hidden_dim,
            key_layers=key_layers,
            basis_activation=basis_activation,
            value_mode=value_mode,
            normalize=encoder_normalize,
            learn_temperature=learn_temperature,
        )

        trunk_layers = [nn.Linear(input_size_tgt, trunk_hidden_size), activation_fn()]
        for _ in range(n_trunk_layers - 2):
            trunk_layers.append(nn.Linear(trunk_hidden_size, trunk_hidden_size))
            trunk_layers.append(activation_fn())
        trunk_layers.append(nn.Linear(trunk_hidden_size, output_size_tgt * p))
        self.trunk = nn.Sequential(*trunk_layers)

        self.bias = nn.Parameter(torch.randn(output_size_tgt) * 0.1) if use_deeponet_bias else None
        self.opt = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        self.total_steps = 0

    @property
    def set_encoder(self) -> WeightedSetEncoder:
        return self._set_encoder

    def _sinusoidal_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coord_dim = coords.shape[-1]
        dims_per_coord = self.pos_encoding_dim // coord_dim
        half_dim = dims_per_coord // 2
        div_term = torch.exp(
            torch.arange(half_dim, device=coords.device)
            * -(torch.log(torch.tensor(self.pos_encoding_max_freq, device=coords.device)) / half_dim)
        )
        div_term = div_term.reshape(1, 1, 1, half_dim)
        coords_expanded = coords.unsqueeze(-1)
        angles = coords_expanded * div_term
        sin_embed = torch.sin(angles)
        cos_embed = torch.cos(angles)
        encoding = torch.cat([sin_embed, cos_embed], dim=-1).reshape(
            coords.shape[0], coords.shape[1], coord_dim, dims_per_coord
        )
        return encoding.reshape(coords.shape[0], coords.shape[1], self.pos_encoding_dim)

    def forward_branch(
        self,
        xs: torch.Tensor,
        us: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
        value_xs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded_coords = self._sinusoidal_encoding(xs) if self.use_positional_encoding else xs
        if sensor_weights is not None:
            inferred_weights = sensor_weights
        elif self.uniform_sensor_weights:
            inferred_weights = infer_uniform_weights(xs, sensor_mask)
        else:
            inferred_weights = infer_quadrature_weights(xs, sensor_mask)

        return self._set_encoder(
            encoded_coords,
            us,
            element_mask=sensor_mask,
            element_weights=inferred_weights,
            value_coords=value_xs,
        )

    def forward_trunk(self, ys: torch.Tensor) -> torch.Tensor:
        batch_size, n_points = ys.shape[0], ys.shape[1]
        trunk_out_flat = self.trunk(ys)
        return trunk_out_flat.reshape(batch_size, n_points, self.p, self.output_size_tgt)

    def forward(
        self,
        xs: torch.Tensor,
        us: torch.Tensor,
        ys: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
        sensor_weights: torch.Tensor | None = None,
        value_xs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        branch_tokens = self.forward_branch(
            xs,
            us,
            sensor_mask=sensor_mask,
            sensor_weights=sensor_weights,
            value_xs=value_xs,
        )
        trunk_tokens = self.forward_trunk(ys)
        out = torch.einsum("bpz,bdpz->bdz", branch_tokens, trunk_tokens)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _get_current_lr(self) -> float:
        if self.lr_schedule_steps is None or self.lr_schedule_rates is None:
            return self.initial_lr
        lr = self.initial_lr
        milestone_idx = -1
        for i, step_milestone in enumerate(self.lr_schedule_steps):
            if self.total_steps >= step_milestone:
                milestone_idx = i
            else:
                break
        if milestone_idx != -1:
            lr = self.lr_schedule_rates[milestone_idx + 1]
        return lr

    def _update_lr(self) -> float:
        new_lr = self._get_current_lr()
        for param_group in self.opt.param_groups:
            param_group["lr"] = new_lr
        return new_lr

    def train_model(self, dataset, epochs: int, progress_bar: bool = True):
        bar = trange(epochs) if progress_bar else range(epochs)
        for _ in bar:
            current_lr = self._update_lr()
            xs, us, ys, targets, sensor_mask = dataset.sample()
            preds = self.forward(xs, us, ys, sensor_mask=sensor_mask)
            prediction_loss = nn.MSELoss()(preds, targets)

            with torch.no_grad():
                pred_flat = preds.squeeze(-1) if preds.shape[-1] == 1 else preds.reshape(preds.shape[0], -1)
                target_flat = (
                    targets.squeeze(-1) if targets.shape[-1] == 1 else targets.reshape(targets.shape[0], -1)
                )
                rel_l2_error = calculate_l2_relative_error(pred_flat, target_flat)

            self.opt.zero_grad()
            prediction_loss.backward()
            norm = nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.opt.step()
            self.total_steps += 1

            if progress_bar:
                bar.set_description(
                    f"Step {self.total_steps} | Loss: {prediction_loss.item():.4e} | "
                    f"Rel L2: {rel_l2_error.item():.4f} | Grad Norm: {float(norm):.2f} | "
                    f"LR: {current_lr:.2e}"
                )
