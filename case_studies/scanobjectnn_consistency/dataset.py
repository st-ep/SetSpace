from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

DEFAULT_LABEL_NAMES = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]

SAMPLING_MODES = ("uniform_object", "clustered_object", "occluded_object", "background_heavy")
SPLITS = ("train", "val", "test")
TRAIN_FILE = "training_objectdataset_augmentedrot_scale75.h5"
TEST_FILE = "test_objectdataset_augmentedrot_scale75.h5"


@dataclass
class ScanObjectRecord:
    coords: torch.Tensor
    foreground_indices: torch.Tensor
    background_indices: torch.Tensor
    label: int


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _random_direction(generator: torch.Generator) -> torch.Tensor:
    return _normalize(torch.randn(3, generator=generator, dtype=torch.float32))


def _as_long_vector(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values)
    if out.ndim == 2 and out.shape[1] == 1:
        out = out[:, 0]
    return out.astype(np.int64, copy=False)


def _stratified_train_val_split(labels: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if val_fraction <= 0.0:
        all_idx = np.arange(labels.shape[0], dtype=np.int64)
        return all_idx, np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(int(seed))
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []
    for label in np.unique(labels):
        cls_idx = np.flatnonzero(labels == label)
        perm = rng.permutation(cls_idx)
        if perm.size <= 1:
            val_count = 0
        else:
            val_count = int(round(float(perm.size) * float(val_fraction)))
            val_count = max(1, val_count)
            val_count = min(val_count, perm.size - 1)
        val_indices.append(np.sort(perm[:val_count]))
        train_indices.append(np.sort(perm[val_count:]))

    train_idx = np.sort(np.concatenate(train_indices)) if train_indices else np.empty(0, dtype=np.int64)
    val_idx = np.sort(np.concatenate(val_indices)) if val_indices else np.empty(0, dtype=np.int64)
    return train_idx, val_idx


def _sample_indices_from_pool(
    pool_indices: torch.Tensor,
    *,
    n_points: int,
    generator: torch.Generator,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    n_points = int(n_points)
    pool_size = int(pool_indices.numel())
    if n_points <= 0:
        return pool_indices[:0], {
            "replacement_points": 0,
            "used_replacement": 0,
            "empty_pool_points": 0,
        }
    if pool_size == 0:
        return torch.empty((0,), dtype=torch.long), {
            "replacement_points": 0,
            "used_replacement": 0,
            "empty_pool_points": n_points,
        }

    if weights is not None:
        weights = weights.to(dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-8)

    unique_count = min(pool_size, n_points)
    if weights is None:
        unique_local = torch.randperm(pool_size, generator=generator)[:unique_count]
    else:
        unique_local = torch.multinomial(weights, unique_count, replacement=False, generator=generator)
    sampled_local = [unique_local]

    deficit = n_points - unique_count
    if deficit > 0:
        if weights is None:
            extra_local = torch.randint(pool_size, (deficit,), generator=generator)
        else:
            extra_local = torch.multinomial(weights, deficit, replacement=True, generator=generator)
        sampled_local.append(extra_local)

    sampled_local_idx = torch.cat(sampled_local, dim=0)
    shuffle = torch.randperm(sampled_local_idx.numel(), generator=generator)
    sampled = pool_indices[sampled_local_idx[shuffle]]
    return sampled, {
        "replacement_points": max(0, deficit),
        "used_replacement": int(deficit > 0),
        "empty_pool_points": 0,
    }


def _normalize_record(points: np.ndarray, mask: np.ndarray) -> ScanObjectRecord:
    coords = torch.as_tensor(points, dtype=torch.float32)
    mask_bool = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    if mask_bool.numel() != coords.shape[0]:
        raise ValueError("Mask length must match the number of points.")
    if not mask_bool.any():
        mask_bool = torch.ones_like(mask_bool)

    foreground_indices = torch.nonzero(mask_bool, as_tuple=False).squeeze(-1)
    background_indices = torch.nonzero(~mask_bool, as_tuple=False).squeeze(-1)

    foreground_coords = coords[foreground_indices]
    centroid = foreground_coords.mean(dim=0)
    centered = coords - centroid
    radius = centered[foreground_indices].norm(dim=1).max().clamp_min(1e-6)
    normalized = centered / radius
    return ScanObjectRecord(
        coords=normalized.contiguous(),
        foreground_indices=foreground_indices.to(dtype=torch.long),
        background_indices=background_indices.to(dtype=torch.long),
        label=-1,
    )


def _load_h5_records(path: Path) -> list[ScanObjectRecord]:
    with h5py.File(path, "r") as f:
        points = np.asarray(f["data"], dtype=np.float32)
        labels = _as_long_vector(np.asarray(f["label"]))
        masks = np.asarray(f["mask"])

    records = []
    for idx in range(points.shape[0]):
        record = _normalize_record(points[idx], masks[idx])
        record.label = int(labels[idx])
        records.append(record)
    return records


def _make_synthetic_record(label: int, *, n_points_total: int, generator: torch.Generator, n_classes: int) -> tuple[np.ndarray, int, np.ndarray]:
    fg_count = int(round(0.78 * n_points_total))
    bg_count = int(n_points_total) - fg_count

    base_scales = torch.tensor(
        [
            [1.0, 0.6, 0.4],
            [0.9, 0.4, 0.7],
            [0.5, 1.0, 0.6],
            [0.4, 0.8, 1.0],
            [1.1, 0.5, 0.5],
            [0.6, 1.1, 0.5],
        ],
        dtype=torch.float32,
    )
    scale = base_scales[int(label) % base_scales.shape[0]].clone()
    scale = scale * (0.85 + 0.15 * torch.randn(3, generator=generator).abs())
    obj = torch.randn((fg_count, 3), generator=generator, dtype=torch.float32) * scale.unsqueeze(0)
    obj = obj + 0.05 * torch.randn(obj.shape, generator=generator, dtype=torch.float32)

    background = torch.empty((bg_count, 3), dtype=torch.float32).uniform_(-1.4, 1.4, generator=generator)
    points = torch.cat([obj, background], dim=0)
    mask = torch.cat([torch.ones(fg_count, dtype=torch.float32), torch.zeros(bg_count, dtype=torch.float32)], dim=0)

    perm = torch.randperm(n_points_total, generator=generator)
    return points[perm].cpu().numpy(), int(label % n_classes), mask[perm].cpu().numpy()


def create_synthetic_scanobjectnn_fixture(
    data_root: Path | str,
    *,
    n_train: int = 45,
    n_test: int = 15,
    n_points_total: int = 2048,
    n_classes: int = 5,
    seed: int = 0,
) -> Path:
    data_root = Path(data_root)
    split_root = data_root / "main_split"
    split_root.mkdir(parents=True, exist_ok=True)

    def _write_file(path: Path, n_samples: int, offset: int) -> None:
        data = np.zeros((n_samples, n_points_total, 3), dtype=np.float32)
        label = np.zeros((n_samples, 1), dtype=np.int64)
        mask = np.zeros((n_samples, n_points_total), dtype=np.float32)
        for idx in range(n_samples):
            generator = torch.Generator().manual_seed(int(seed) * 100_003 + (offset + idx) * 7_919 + 17)
            points, target, point_mask = _make_synthetic_record(
                idx % n_classes,
                n_points_total=n_points_total,
                generator=generator,
                n_classes=n_classes,
            )
            data[idx] = points
            label[idx, 0] = target
            mask[idx] = point_mask

        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("label", data=label)
            f.create_dataset("mask", data=mask)

    _write_file(split_root / TRAIN_FILE, int(n_train), 0)
    _write_file(split_root / TEST_FILE, int(n_test), int(n_train))
    return data_root


class ScanObjectNNConsistencyDataset:
    def __init__(
        self,
        *,
        data_root: str = "data/ScanObjectNN/h5_files",
        variant: str = "pb_t50_rs",
        seed: int = 0,
        val_fraction: float = 0.1,
        label_names: list[str] | None = None,
    ) -> None:
        self.data_root = str(data_root)
        self.variant = str(variant).lower()
        self.seed = int(seed)
        self.val_fraction = float(val_fraction)
        self.label_names = list(label_names) if label_names is not None else list(DEFAULT_LABEL_NAMES)

        if self.variant != "pb_t50_rs":
            raise ValueError(f"Unsupported ScanObjectNN variant: {variant}")

        root = Path(self.data_root)
        train_path = root / "main_split" / TRAIN_FILE
        test_path = root / "main_split" / TEST_FILE
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "Expected ScanObjectNN files were not found. "
                f"Looked for {train_path} and {test_path}."
            )

        trainval_records = _load_h5_records(train_path)
        test_records = _load_h5_records(test_path)

        trainval_labels = np.array([record.label for record in trainval_records], dtype=np.int64)
        train_idx, val_idx = _stratified_train_val_split(trainval_labels, self.val_fraction, self.seed)
        self.records = {
            "train": [trainval_records[int(idx)] for idx in train_idx.tolist()],
            "val": [trainval_records[int(idx)] for idx in val_idx.tolist()],
            "test": test_records,
        }
        self.num_classes = int(max((record.label for split in self.records.values() for record in split), default=-1) + 1)
        self._batch_generators = {
            split: torch.Generator().manual_seed(self.seed * 1009 + i * 37 + 11)
            for i, split in enumerate(SPLITS)
        }

    def split_size(self, split: str) -> int:
        return len(self.records[split])

    def get_config(self) -> dict:
        return {
            "data_root": self.data_root,
            "variant": self.variant,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "label_names": self.label_names,
        }

    def get_metadata(self) -> dict:
        return {
            "data_root": self.data_root,
            "variant": self.variant,
            "train_size": self.split_size("train"),
            "val_size": self.split_size("val"),
            "test_size": self.split_size("test"),
            "num_classes": self.num_classes,
            "label_names": self.label_names[: self.num_classes],
        }

    def get_label_name(self, label: int) -> str:
        label = int(label)
        if 0 <= label < len(self.label_names):
            return self.label_names[label]
        return f"class_{label}"

    def _record(self, split: str, local_index: int) -> ScanObjectRecord:
        records = self.records[split]
        if local_index < 0 or local_index >= len(records):
            raise IndexError(f"{local_index=} is out of range for split={split}")
        return records[int(local_index)]

    def get_labels(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        labels = torch.tensor(
            [self._record(split, int(local_index)).label for local_index in local_indices.tolist()],
            dtype=torch.long,
        )
        if device is not None:
            labels = labels.to(device)
        return labels

    def _sample_uniform_object(
        self,
        record: ScanObjectRecord,
        *,
        n_points: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        selected, stats = _sample_indices_from_pool(
            record.foreground_indices,
            n_points=n_points,
            generator=generator,
        )
        point_is_foreground = torch.ones(selected.numel(), dtype=torch.bool)
        return selected, point_is_foreground, {
            **stats,
            "background_replacement_points": 0,
            "background_used_replacement": 0,
            "empty_background_points": 0,
            "mode": "uniform_object",
        }

    def _sample_clustered_object(
        self,
        record: ScanObjectRecord,
        *,
        n_points: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        pool = record.foreground_indices
        pool_coords = record.coords[pool]
        if pool.numel() == 0:
            return pool, torch.zeros(0, dtype=torch.bool), {
                "replacement_points": 0,
                "used_replacement": 0,
                "background_replacement_points": 0,
                "background_used_replacement": 0,
                "empty_background_points": n_points,
                "mode": "clustered_object",
            }

        anchor_idx = int(torch.randint(pool.numel(), (1,), generator=generator).item())
        anchor = pool_coords[anchor_idx]
        distances = (pool_coords - anchor.unsqueeze(0)).norm(dim=1)
        positive = distances[distances > 0]
        sigma = positive.median() if positive.numel() > 0 else torch.tensor(1.0, dtype=torch.float32)
        sigma = sigma.clamp_min(1e-3)
        weights = torch.exp(-0.5 * distances.square() / sigma.square()) + 1e-4
        selected, stats = _sample_indices_from_pool(
            pool,
            n_points=n_points,
            generator=generator,
            weights=weights,
        )
        point_is_foreground = torch.ones(selected.numel(), dtype=torch.bool)
        return selected, point_is_foreground, {
            **stats,
            "background_replacement_points": 0,
            "background_used_replacement": 0,
            "empty_background_points": 0,
            "mode": "clustered_object",
        }

    def _sample_occluded_object(
        self,
        record: ScanObjectRecord,
        *,
        n_points: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        pool = record.foreground_indices
        pool_coords = record.coords[pool]
        direction = _random_direction(generator)
        depth = pool_coords @ direction
        if pool.numel() == 0:
            return pool, torch.zeros(0, dtype=torch.bool), {
                "replacement_points": 0,
                "used_replacement": 0,
                "background_replacement_points": 0,
                "background_used_replacement": 0,
                "empty_background_points": n_points,
                "mode": "occluded_object",
                "occlusion_direction": direction,
            }

        unique_count = min(int(n_points), int(pool.numel()))
        keep = torch.argsort(depth, descending=True)[:unique_count]
        selected = pool[keep]
        deficit = int(n_points) - unique_count
        if deficit > 0:
            weights = torch.softmax(depth, dim=0)
            extra = torch.multinomial(weights, deficit, replacement=True, generator=generator)
            selected = torch.cat([selected, pool[extra]], dim=0)
            shuffle = torch.randperm(selected.numel(), generator=generator)
            selected = selected[shuffle]
        point_is_foreground = torch.ones(selected.numel(), dtype=torch.bool)
        return selected, point_is_foreground, {
            "replacement_points": max(0, deficit),
            "used_replacement": int(deficit > 0),
            "background_replacement_points": 0,
            "background_used_replacement": 0,
            "empty_background_points": 0,
            "mode": "occluded_object",
            "occlusion_direction": direction,
            "selected_depth": (record.coords[selected] @ direction).to(dtype=torch.float32),
        }

    def _sample_background_heavy(
        self,
        record: ScanObjectRecord,
        *,
        n_points: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        fg_count = int(math.ceil(0.6 * int(n_points)))
        bg_count = int(n_points) - fg_count

        fg_selected, fg_stats = _sample_indices_from_pool(
            record.foreground_indices,
            n_points=fg_count,
            generator=generator,
        )
        bg_selected, bg_stats = _sample_indices_from_pool(
            record.background_indices,
            n_points=bg_count,
            generator=generator,
        )
        empty_background_points = int(bg_stats["empty_pool_points"])
        if empty_background_points > 0:
            fallback, fallback_stats = _sample_indices_from_pool(
                record.foreground_indices,
                n_points=empty_background_points,
                generator=generator,
            )
            bg_selected = torch.cat([bg_selected, fallback], dim=0)
            bg_replacement_points = int(bg_stats["replacement_points"]) + int(fallback_stats["replacement_points"])
        else:
            bg_replacement_points = int(bg_stats["replacement_points"])

        selected = torch.cat([fg_selected, bg_selected], dim=0)
        point_is_foreground = torch.cat(
            [
                torch.ones(fg_selected.numel(), dtype=torch.bool),
                torch.zeros(bg_selected.numel(), dtype=torch.bool),
            ],
            dim=0,
        )
        if selected.numel() > 0:
            shuffle = torch.randperm(selected.numel(), generator=generator)
            selected = selected[shuffle]
            point_is_foreground = point_is_foreground[shuffle]

        return selected, point_is_foreground, {
            "replacement_points": int(fg_stats["replacement_points"]),
            "used_replacement": int(fg_stats["used_replacement"]),
            "background_replacement_points": bg_replacement_points,
            "background_used_replacement": int(bg_replacement_points > 0),
            "empty_background_points": empty_background_points,
            "mode": "background_heavy",
        }

    def sample_view(
        self,
        split: str,
        local_index: int,
        *,
        n_points: int,
        sampling_mode: str,
        view_seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, dict]:
        record = self._record(split, int(local_index))
        generator = torch.Generator().manual_seed(int(view_seed))

        if sampling_mode == "uniform_object":
            selected, point_is_foreground, metadata = self._sample_uniform_object(record, n_points=n_points, generator=generator)
        elif sampling_mode == "clustered_object":
            selected, point_is_foreground, metadata = self._sample_clustered_object(record, n_points=n_points, generator=generator)
        elif sampling_mode == "occluded_object":
            selected, point_is_foreground, metadata = self._sample_occluded_object(record, n_points=n_points, generator=generator)
        elif sampling_mode == "background_heavy":
            selected, point_is_foreground, metadata = self._sample_background_heavy(record, n_points=n_points, generator=generator)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        coords = record.coords[selected]
        values = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        metadata = {
            **metadata,
            "point_is_foreground": point_is_foreground,
            "num_points": int(coords.shape[0]),
        }
        return coords.to(dtype=torch.float32), values, int(record.label), metadata

    def collate_views(
        self,
        split: str,
        local_indices: torch.Tensor,
        *,
        n_points: int,
        sampling_mode: str,
        view_seeds: torch.Tensor,
        device: torch.device | None = None,
        return_metadata: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        coords_list = []
        values_list = []
        metadata_list = []
        for local_index, view_seed in zip(local_indices.tolist(), view_seeds.tolist()):
            coords, values, _label, metadata = self.sample_view(
                split,
                int(local_index),
                n_points=n_points,
                sampling_mode=sampling_mode,
                view_seed=int(view_seed),
            )
            coords_list.append(coords)
            values_list.append(values)
            metadata_list.append(metadata)

        coords_batch = torch.stack(coords_list, dim=0)
        values_batch = torch.stack(values_list, dim=0)
        labels_batch = self.get_labels(split, local_indices)
        if device is not None:
            coords_batch = coords_batch.to(device)
            values_batch = values_batch.to(device)
            labels_batch = labels_batch.to(device)
        if return_metadata:
            return coords_batch, values_batch, labels_batch, metadata_list
        return coords_batch, values_batch, labels_batch

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
