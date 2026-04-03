from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SAMPLING_MODES = (
    "uniform",
    "leading_edge",
    "trailing_edge",
    "upper_surface",
    "lower_surface",
    "clustered",
    "occluded",
)


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return np.full_like(weights, 1.0 / max(len(weights), 1), dtype=np.float64)
    return weights / total


def _center_and_scale(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    centered = (coords - mins) / spans
    return centered.astype(np.float32), mins.astype(np.float32), spans.astype(np.float32)


def _random_centers(coords: np.ndarray, rng: np.random.Generator, n_centers: int) -> np.ndarray:
    count = min(max(int(n_centers), 1), coords.shape[0])
    idx = rng.choice(coords.shape[0], size=count, replace=False)
    return coords[idx]


def _sampling_bias(coords_unit: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    mode = str(mode).lower()
    x = coords_unit[:, 0]
    y = coords_unit[:, 1]
    y_scale = max(float(y.max() - y.min()), 1e-6)
    y_centered = (y - float(np.median(y))) / y_scale
    if mode == "uniform":
        return np.ones(coords_unit.shape[0], dtype=np.float64)
    if mode == "leading_edge":
        return 0.05 + np.exp(-0.5 * np.square(x / 0.12))
    if mode == "trailing_edge":
        return 0.05 + np.exp(-0.5 * np.square((x - 1.0) / 0.14))
    if mode == "upper_surface":
        return 0.05 + 1.0 / (1.0 + np.exp(-8.0 * y_centered))
    if mode == "lower_surface":
        return 0.05 + 1.0 / (1.0 + np.exp(8.0 * y_centered))
    if mode == "clustered":
        centers = _random_centers(coords_unit, rng, 2)
        dists = np.stack([np.linalg.norm(coords_unit - center[None, :], axis=1) for center in centers], axis=0)
        return 0.15 + np.exp(-28.0 * np.square(dists)).sum(axis=0)
    if mode == "occluded":
        center = _random_centers(coords_unit, rng, 1)[0]
        dist = np.linalg.norm(coords_unit - center[None, :], axis=1)
        return 0.05 + 1.0 - np.exp(-32.0 * np.square(dist))
    raise ValueError(f"Unsupported sampling mode: {mode}")


def _sample_indices(probs: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    n_total = probs.shape[0]
    n_points = int(n_points)
    replace = n_points > n_total
    return rng.choice(n_total, size=n_points, replace=replace, p=probs)


class AirfRANSForceDataset:
    def __init__(
        self,
        processed_root: str | Path,
        *,
        task: str = "full",
        seed: int = 0,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        **_unused,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.task = str(task)
        self.seed = int(seed)

        task_root = self.processed_root / self.task
        self.root = task_root if task_root.exists() else self.processed_root
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No AirfRANS metadata found under {self.root}. "
                "Run case_studies/airfrans_field_prediction/prepare_dataset.py first."
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.target_names = list(self.metadata.get("target_names", ["Cd", "Cl"]))
        self.value_feature_names = list(
            self.metadata.get(
                "value_feature_names",
                ["pressure", "wall_shear_x", "wall_shear_y", "normal_x", "normal_y"],
            )
        )

        self.samples: dict[str, list[dict[str, np.ndarray]]] = {}
        split_limits = {"train": n_train, "val": n_val, "test": n_test}
        for split in ("train", "val", "test"):
            split_dir = self.root / split
            files = sorted(split_dir.glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"No processed AirfRANS samples found under {split_dir}.")
            limit = split_limits[split]
            if limit is not None:
                files = files[: int(limit)]
            split_samples: list[dict[str, np.ndarray]] = []
            for file_path in files:
                with np.load(file_path, allow_pickle=False) as data:
                    coords = np.asarray(data["coords"], dtype=np.float32)
                    values = np.asarray(data["values"], dtype=np.float32)
                    base_weights = _normalized_weights(np.asarray(data["base_weights"], dtype=np.float64)).astype(
                        np.float32
                    )
                    targets = np.asarray(data["targets"], dtype=np.float32)
                coords_unit, _, _ = _center_and_scale(coords[:, :2])
                split_samples.append(
                    {
                        "coords": coords,
                        "coords_unit": coords_unit,
                        "values": values,
                        "base_weights": base_weights,
                        "targets": targets,
                    }
                )
            self.samples[split] = split_samples

        example = self.samples["train"][0]
        self.value_input_dim = int(example["values"].shape[-1])
        self.target_dim = int(example["targets"].shape[-1])
        self.min_points = min(sample["coords"].shape[0] for split in self.samples.values() for sample in split)

        train_targets = np.stack([sample["targets"] for sample in self.samples["train"]], axis=0)
        self.target_mean = torch.from_numpy(train_targets.mean(axis=0).astype(np.float32))
        self.target_std = torch.from_numpy(np.clip(train_targets.std(axis=0), 1e-6, None).astype(np.float32))

    def split_size(self, split: str) -> int:
        return len(self.samples[split])

    def get_config(self) -> dict:
        return {
            "processed_root": str(self.processed_root),
            "task": self.task,
            "seed": self.seed,
            "target_names": list(self.target_names),
            "value_feature_names": list(self.value_feature_names),
            "n_train": self.split_size("train"),
            "n_val": self.split_size("val"),
            "n_test": self.split_size("test"),
            "min_points": self.min_points,
        }

    def get_normalization_stats(self) -> dict:
        return {
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        mean = self.target_mean.to(device=targets.device, dtype=targets.dtype)
        std = self.target_std.to(device=targets.device, dtype=targets.dtype)
        return (targets - mean) / std

    def denormalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        mean = self.target_mean.to(device=targets.device, dtype=targets.dtype)
        std = self.target_std.to(device=targets.device, dtype=targets.dtype)
        return targets * std + mean

    def get_targets(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        device: torch.device | None = None,
        normalized: bool = False,
    ) -> torch.Tensor:
        rows = [self.samples[split][int(idx)]["targets"] for idx in local_indices.tolist()]
        targets = torch.from_numpy(np.stack(rows, axis=0))
        if normalized:
            targets = self.normalize_targets(targets)
        if device is not None:
            targets = targets.to(device=device)
        return targets

    def sample_view(
        self,
        split: str,
        local_index: int,
        *,
        n_points: int,
        sampling_mode: str,
        view_seed: int | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[split][int(local_index)]
        rng = np.random.default_rng(int(view_seed) if view_seed is not None else None)
        probs = _normalized_weights(sample["base_weights"] * _sampling_bias(sample["coords_unit"], sampling_mode, rng))
        sampled_idx = _sample_indices(probs, n_points=n_points, rng=rng)
        coords = torch.from_numpy(sample["coords"][sampled_idx])
        values = torch.from_numpy(sample["values"][sampled_idx])
        if device is not None:
            coords = coords.to(device=device)
            values = values.to(device=device)
        return coords, values

    def collate_views(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        n_points: int,
        sampling_mode: str,
        view_seeds: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords_list = []
        values_list = []
        for batch_idx, local_index in enumerate(local_indices.tolist()):
            seed = None if view_seeds is None else int(view_seeds[batch_idx].item())
            coords, values = self.sample_view(
                split,
                local_index,
                n_points=n_points,
                sampling_mode=sampling_mode,
                view_seed=seed,
                device=device,
            )
            coords_list.append(coords)
            values_list.append(values)
        return torch.stack(coords_list, dim=0), torch.stack(values_list, dim=0)

    def sample_batch(
        self,
        split: str,
        *,
        batch_size: int,
        n_points: int,
        sampling_mode: str,
        device: torch.device | None = None,
        normalized_targets: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = self.split_size(split)
        idx = torch.randint(0, size, (int(batch_size),))
        coords, values = self.collate_views(
            split,
            idx,
            n_points=n_points,
            sampling_mode=sampling_mode,
            device=device,
        )
        targets = self.get_targets(split, idx, device=device, normalized=normalized_targets)
        return coords, values, targets
