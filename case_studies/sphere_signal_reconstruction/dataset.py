from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from case_studies.sphere_utils import sample_surface_points, sample_uniform_sphere

SAMPLING_MODES = ("uniform", "polar", "equatorial", "clustered", "hemisphere")
SPLITS = ("train", "val", "test")


def fibonacci_sphere(n_points: int) -> torch.Tensor:
    idx = torch.arange(int(n_points), dtype=torch.float32)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    theta = 2.0 * math.pi * idx / phi
    z = 1.0 - 2.0 * (idx + 0.5) / float(n_points)
    r = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=1)


def _bit_reverse(value: int, n_bits: int) -> int:
    out = 0
    for _ in range(n_bits):
        out = (out << 1) | (value & 1)
        value >>= 1
    return out


def nested_fibonacci_prefix_order(n_points: int) -> torch.Tensor:
    """
    Reorder the latitude-sorted Fibonacci sphere so dyadic prefixes cover the
    full sphere instead of only a polar cap.

    A naive prefix of fibonacci_sphere(n) is strongly biased because the native
    ordering is monotone in z. Bit-reversal interleaves those latitudes, giving
    a deterministic refinement family whose prefixes are much closer to uniform.
    """

    n_points = int(n_points)
    if n_points <= 1:
        return torch.arange(n_points, dtype=torch.long)

    n_bits = (n_points - 1).bit_length()
    limit = 1 << n_bits
    order = [_bit_reverse(i, n_bits) for i in range(limit) if _bit_reverse(i, n_bits) < n_points]
    return torch.tensor(order, dtype=torch.long)


def _real_harmonic_basis_raw(points: torch.Tensor) -> torch.Tensor:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return torch.stack(
        [
            torch.ones_like(x),
            x,
            y,
            z,
            x * y,
            y * z,
            z * x,
            x.square() - y.square(),
            3.0 * z.square() - 1.0,
            x * (x.square() - 3.0 * y.square()),
            y * (3.0 * x.square() - y.square()),
            z * (x.square() - y.square()),
            x * y * z,
            x * (5.0 * z.square() - 1.0),
            y * (5.0 * z.square() - 1.0),
            z * (5.0 * z.square() - 3.0),
        ],
        dim=1,
    )


@dataclass
class SphereSignalObject:
    global_coeffs: torch.Tensor
    bump_centers: torch.Tensor
    bump_amplitudes: torch.Tensor
    bump_concentrations: torch.Tensor


def _oracle_weights_from_scores(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    weights = scores.clamp_min(eps).reciprocal()
    return weights / weights.sum().clamp_min(eps)


class SphereSignalDataset:
    """
    Sphere signal reconstruction benchmark.

    Each object is a scalar field on the unit sphere, observed through sparse
    samples under different density shifts. The target is the full field on a
    fixed canonical query set.
    """

    def __init__(
        self,
        *,
        n_train: int = 2048,
        n_val: int = 256,
        n_test: int = 512,
        seed: int = 0,
        n_bumps: int = 4,
        query_points: int = 1024,
    ) -> None:
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)
        self.seed = int(seed)
        self.n_bumps = int(n_bumps)
        self.query_points = int(query_points)
        self.total_objects = self.n_train + self.n_val + self.n_test

        self.query_coords = fibonacci_sphere(self.query_points)
        self.refinement_order = nested_fibonacci_prefix_order(self.query_points)
        self.refinement_coords = self.query_coords[self.refinement_order]
        raw_basis = _real_harmonic_basis_raw(self.query_coords)
        self._basis_scale = raw_basis.square().mean(dim=0).sqrt().clamp_min(1e-6)
        self._spectral_projection = torch.linalg.pinv(raw_basis / self._basis_scale.unsqueeze(0))

        degree_scales = torch.tensor(
            [0.45] * 1 + [0.35] * 3 + [0.24] * 5 + [0.16] * 7,
            dtype=torch.float32,
        )

        self.objects: list[SphereSignalObject] = []
        query_values = []
        for global_idx in range(self.total_objects):
            generator = torch.Generator().manual_seed(self.seed * 100_003 + global_idx * 7_919 + 19)
            global_coeffs = degree_scales * torch.randn(16, generator=generator, dtype=torch.float32)
            bump_centers = sample_uniform_sphere(self.n_bumps, generator)
            bump_amplitudes = 0.55 * torch.randn(self.n_bumps, generator=generator, dtype=torch.float32)
            bump_concentrations = torch.empty(self.n_bumps, dtype=torch.float32).uniform_(
                8.0,
                22.0,
                generator=generator,
            )
            obj = SphereSignalObject(
                global_coeffs=global_coeffs,
                bump_centers=bump_centers,
                bump_amplitudes=bump_amplitudes,
                bump_concentrations=bump_concentrations,
            )
            self.objects.append(obj)
            query_values.append(self.evaluate_field_raw(obj, self.query_coords))

        self.query_values_raw = torch.stack(query_values, dim=0).to(dtype=torch.float32)
        train_values = self.query_values_raw[: self.n_train]
        self.value_mean = float(train_values.mean().item())
        self.value_std = float(train_values.std(unbiased=False).clamp_min(1e-6).item())
        self.query_values = self.standardize_values(self.query_values_raw)

        self.split_offsets = {
            "train": (0, self.n_train),
            "val": (self.n_train, self.n_train + self.n_val),
            "test": (self.n_train + self.n_val, self.total_objects),
        }
        self._batch_generators = {
            split: torch.Generator().manual_seed(self.seed * 1009 + i * 37 + 11)
            for i, split in enumerate(SPLITS)
        }

    def split_size(self, split: str) -> int:
        start, end = self.split_offsets[split]
        return end - start

    def get_config(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "seed": self.seed,
            "n_bumps": self.n_bumps,
            "query_points": self.query_points,
        }

    def get_normalization_stats(self) -> dict:
        return {"mean": self.value_mean, "std": self.value_std}

    def _global_index(self, split: str, local_index: int) -> int:
        start, end = self.split_offsets[split]
        if local_index < 0 or start + local_index >= end:
            raise IndexError(f"{local_index=} is out of range for split {split}")
        return start + local_index

    def _harmonic_basis(self, points: torch.Tensor) -> torch.Tensor:
        scale = self._basis_scale.to(device=points.device, dtype=points.dtype)
        return _real_harmonic_basis_raw(points) / scale.unsqueeze(0)

    def evaluate_field_raw(self, obj: SphereSignalObject, points: torch.Tensor) -> torch.Tensor:
        _to = lambda t: t.to(device=points.device, dtype=points.dtype)
        basis = self._harmonic_basis(points)
        smooth = basis @ _to(obj.global_coeffs)
        local = torch.exp(_to(obj.bump_concentrations) * ((points @ _to(obj.bump_centers).T) - 1.0)) @ _to(obj.bump_amplitudes)
        return smooth + local

    def evaluate_split_object_raw(self, split: str, local_index: int, points: torch.Tensor) -> torch.Tensor:
        return self.evaluate_field_raw(self.objects[self._global_index(split, int(local_index))], points)

    def standardize_values(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.value_mean) / self.value_std

    def destandardize_values(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.value_std + self.value_mean

    def get_query_coords(self, device: torch.device | None = None) -> torch.Tensor:
        coords = self.query_coords
        if device is not None:
            coords = coords.to(device)
        return coords

    def get_query_coords_batch(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        coords = self.get_query_coords(device=device)
        return coords.unsqueeze(0).expand(int(batch_size), -1, -1)

    def get_query_targets(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        standardized: bool = True,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        global_indices = torch.tensor(
            [self._global_index(split, int(local_index)) for local_index in local_indices.tolist()],
            dtype=torch.long,
        )
        values = self.query_values[global_indices] if standardized else self.query_values_raw[global_indices]
        if device is not None:
            values = values.to(device)
        return values

    def project_spectral_coeffs(self, values_raw: torch.Tensor) -> torch.Tensor:
        projection = self._spectral_projection.to(device=values_raw.device, dtype=values_raw.dtype)
        return values_raw @ projection.T

    def sample_observation_view(
        self,
        split: str,
        local_index: int,
        *,
        n_points: int,
        sampling_mode: str,
        view_seed: int,
        deterministic_uniform: bool = False,
        standardized: bool = True,
        return_oracle_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if deterministic_uniform:
            points = self.refinement_coords[: int(n_points)].clone()
            oracle_weights = (
                torch.full((int(n_points),), 1.0 / float(n_points), dtype=torch.float32)
                if return_oracle_weights
                else None
            )
        else:
            generator = torch.Generator().manual_seed(int(view_seed))
            if return_oracle_weights:
                points, oracle_scores = sample_surface_points(
                    int(n_points),
                    sampling_mode,
                    generator,
                    return_scores=True,
                )
                oracle_weights = _oracle_weights_from_scores(oracle_scores).to(dtype=torch.float32)
            else:
                points = sample_surface_points(int(n_points), sampling_mode, generator)
                oracle_weights = None
        values_raw = self.evaluate_split_object_raw(split, local_index, points).to(dtype=torch.float32)
        values = self.standardize_values(values_raw) if standardized else values_raw
        if oracle_weights is None:
            return points.to(dtype=torch.float32), values.unsqueeze(-1)
        return points.to(dtype=torch.float32), values.unsqueeze(-1), oracle_weights

    def collate_observations(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        n_points: int,
        sampling_mode: str,
        view_seeds: torch.Tensor,
        deterministic_uniform: bool = False,
        standardized: bool = True,
        device: torch.device | None = None,
        return_oracle_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points_list = []
        values_list = []
        oracle_weight_list = []
        for local_index, view_seed in zip(local_indices.tolist(), view_seeds.tolist()):
            sample = self.sample_observation_view(
                split,
                int(local_index),
                n_points=n_points,
                sampling_mode=sampling_mode,
                view_seed=int(view_seed),
                deterministic_uniform=deterministic_uniform,
                standardized=standardized,
                return_oracle_weights=return_oracle_weights,
            )
            if return_oracle_weights:
                points, values, oracle_weights = sample
                oracle_weight_list.append(oracle_weights)
            else:
                points, values = sample
            points_list.append(points)
            values_list.append(values)

        points_batch = torch.stack(points_list, dim=0)
        values_batch = torch.stack(values_list, dim=0)
        oracle_weights_batch = torch.stack(oracle_weight_list, dim=0) if return_oracle_weights else None
        if device is not None:
            points_batch = points_batch.to(device)
            values_batch = values_batch.to(device)
            if oracle_weights_batch is not None:
                oracle_weights_batch = oracle_weights_batch.to(device)
        if oracle_weights_batch is None:
            return points_batch, values_batch
        return points_batch, values_batch, oracle_weights_batch

    def sample_batch(
        self,
        split: str,
        *,
        batch_size: int,
        n_points: int,
        sampling_mode: str,
        device: torch.device | None = None,
        return_oracle_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = self._batch_generators[split]
        local_indices = torch.randint(0, self.split_size(split), (int(batch_size),), generator=generator)
        view_seeds = torch.randint(0, 2**31 - 1, (int(batch_size),), generator=generator)
        collated = self.collate_observations(
            split,
            local_indices,
            n_points=n_points,
            sampling_mode=sampling_mode,
            view_seeds=view_seeds,
            standardized=True,
            device=device,
            return_oracle_weights=return_oracle_weights,
        )
        if return_oracle_weights:
            obs_coords, obs_values, oracle_weights = collated
        else:
            obs_coords, obs_values = collated
        query_coords = self.get_query_coords_batch(int(batch_size), device=device)
        query_targets = self.get_query_targets(split, local_indices, standardized=True, device=device).unsqueeze(-1)
        if device is not None:
            local_indices = local_indices.to(device)
        if not return_oracle_weights:
            return obs_coords, obs_values, query_coords, query_targets, local_indices
        return obs_coords, obs_values, oracle_weights, query_coords, query_targets, local_indices
