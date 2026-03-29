from __future__ import annotations

from dataclasses import dataclass

import torch

from case_studies.sphere_utils import sample_surface_points, sample_uniform_sphere

SAMPLING_MODES = ("uniform", "polar", "equatorial", "clustered", "hemisphere")
SPLITS = ("train", "val", "test")


def _quadratic_features(points: torch.Tensor) -> torch.Tensor:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return torch.stack(
        [
            x * y,
            y * z,
            z * x,
            x.square() - y.square(),
            3.0 * z.square() - 1.0,
        ],
        dim=1,
    )


@dataclass
class SurfaceSignalObject:
    bias: float
    linear: torch.Tensor
    quadratic: torch.Tensor
    bump_centers: torch.Tensor
    bump_amplitudes: torch.Tensor
    bump_concentrations: torch.Tensor
    integral_estimate: float
    label: int


def evaluate_surface_signal(obj: SurfaceSignalObject, points: torch.Tensor) -> torch.Tensor:
    poly = obj.bias + points @ obj.linear + _quadratic_features(points) @ obj.quadratic
    bumps = torch.exp(obj.bump_concentrations * ((points @ obj.bump_centers.T) - 1.0)) @ obj.bump_amplitudes
    return (poly + bumps).unsqueeze(-1)


class SyntheticSurfaceSignalDataset:
    """
    Same-object benchmark for sampled-set consistency on a sphere surface.

    Each object is a continuous scalar field on the sphere. The repo uses two
    task views over the same objects:

    - binary classification from the sign of the continuum-average field value
    - regression to the continuum-average field value itself

    Training and evaluation then resample the same object under different
    density shifts.
    """

    def __init__(
        self,
        *,
        n_train: int = 2048,
        n_val: int = 256,
        n_test: int = 512,
        seed: int = 0,
        n_bumps: int = 4,
        label_reference_points: int = 4096,
    ) -> None:
        self.n_train = int(n_train)
        self.n_val = int(n_val)
        self.n_test = int(n_test)
        self.seed = int(seed)
        self.n_bumps = int(n_bumps)
        self.label_reference_points = int(label_reference_points)
        self.total_objects = self.n_train + self.n_val + self.n_test

        integrals: list[float] = []
        raw_objects: list[SurfaceSignalObject] = []
        for global_idx in range(self.total_objects):
            obj_seed = self.seed * 100_003 + global_idx * 7919 + 17
            generator = torch.Generator().manual_seed(obj_seed)

            linear = 0.35 * torch.randn(3, generator=generator, dtype=torch.float32)
            quadratic = 0.25 * torch.randn(5, generator=generator, dtype=torch.float32)
            bump_centers = sample_uniform_sphere(self.n_bumps, generator)
            bump_amplitudes = 0.80 * torch.randn(self.n_bumps, generator=generator, dtype=torch.float32)
            bump_concentrations = torch.empty(self.n_bumps, dtype=torch.float32).uniform_(
                8.0,
                22.0,
                generator=generator,
            )
            bias = float(0.25 * torch.randn(1, generator=generator, dtype=torch.float32).item())

            points_ref = sample_surface_points(self.label_reference_points, "uniform", generator)
            provisional = SurfaceSignalObject(
                bias=bias,
                linear=linear,
                quadratic=quadratic,
                bump_centers=bump_centers,
                bump_amplitudes=bump_amplitudes,
                bump_concentrations=bump_concentrations,
                integral_estimate=0.0,
                label=0,
            )
            integral_estimate = float(evaluate_surface_signal(provisional, points_ref).mean().item())
            raw_objects.append(
                SurfaceSignalObject(
                    bias=bias,
                    linear=linear,
                    quadratic=quadratic,
                    bump_centers=bump_centers,
                    bump_amplitudes=bump_amplitudes,
                    bump_concentrations=bump_concentrations,
                    integral_estimate=integral_estimate,
                    label=0,
                )
            )
            integrals.append(integral_estimate)

        threshold = float(torch.tensor(integrals, dtype=torch.float32).median().item())
        self.label_threshold = threshold
        self.objects = [
            SurfaceSignalObject(
                bias=obj.bias,
                linear=obj.linear,
                quadratic=obj.quadratic,
                bump_centers=obj.bump_centers,
                bump_amplitudes=obj.bump_amplitudes,
                bump_concentrations=obj.bump_concentrations,
                integral_estimate=obj.integral_estimate,
                label=int(obj.integral_estimate > threshold),
            )
            for obj in raw_objects
        ]

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
            "label_reference_points": self.label_reference_points,
        }

    def _global_index(self, split: str, local_index: int) -> int:
        start, end = self.split_offsets[split]
        if local_index < 0 or start + local_index >= end:
            raise IndexError(f"{local_index=} is out of range for split {split}")
        return start + local_index

    def sample_view(
        self,
        split: str,
        local_index: int,
        *,
        n_points: int,
        sampling_mode: str,
        view_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        global_index = self._global_index(split, int(local_index))
        obj = self.objects[global_index]
        generator = torch.Generator().manual_seed(int(view_seed))
        points = sample_surface_points(int(n_points), sampling_mode, generator)
        values = evaluate_surface_signal(obj, points).to(dtype=torch.float32)
        return points.to(dtype=torch.float32), values, int(obj.label)

    def get_labels(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        labels = [self.objects[self._global_index(split, int(local_index))].label for local_index in local_indices.tolist()]
        labels_batch = torch.tensor(labels, dtype=torch.long)
        if device is not None:
            labels_batch = labels_batch.to(device)
        return labels_batch

    def get_integral_targets(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        targets = [
            self.objects[self._global_index(split, int(local_index))].integral_estimate
            for local_index in local_indices.tolist()
        ]
        targets_batch = torch.tensor(targets, dtype=torch.float32)
        if device is not None:
            targets_batch = targets_batch.to(device)
        return targets_batch

    def collate_views(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        n_points: int,
        sampling_mode: str,
        view_seeds: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points_list = []
        values_list = []
        for local_index, view_seed in zip(local_indices.tolist(), view_seeds.tolist()):
            points, values, _ = self.sample_view(
                split,
                local_index,
                n_points=n_points,
                sampling_mode=sampling_mode,
                view_seed=int(view_seed),
            )
            points_list.append(points)
            values_list.append(values)

        points_batch = torch.stack(points_list)
        values_batch = torch.stack(values_list)
        labels_batch = self.get_labels(split, local_indices)
        if device is not None:
            points_batch = points_batch.to(device)
            values_batch = values_batch.to(device)
            labels_batch = labels_batch.to(device)
        return points_batch, values_batch, labels_batch

    def sample_batch(
        self,
        split: str,
        *,
        batch_size: int,
        n_points: int,
        sampling_mode: str,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = self._batch_generators[split]
        local_indices = torch.randint(0, self.split_size(split), (int(batch_size),), generator=generator)
        view_seeds = torch.randint(0, 2**31 - 1, (int(batch_size),), generator=generator)
        return self.collate_views(
            split,
            local_indices,
            n_points=n_points,
            sampling_mode=sampling_mode,
            view_seeds=view_seeds,
            device=device,
        )

    def sample_batch_with_indices(
        self,
        split: str,
        *,
        batch_size: int,
        n_points: int,
        sampling_mode: str,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = self._batch_generators[split]
        local_indices = torch.randint(0, self.split_size(split), (int(batch_size),), generator=generator)
        view_seeds = torch.randint(0, 2**31 - 1, (int(batch_size),), generator=generator)
        points_batch, values_batch, labels_batch = self.collate_views(
            split,
            local_indices,
            n_points=n_points,
            sampling_mode=sampling_mode,
            view_seeds=view_seeds,
            device=device,
        )
        if device is not None:
            local_indices = local_indices.to(device)
        return points_batch, values_batch, labels_batch, local_indices
