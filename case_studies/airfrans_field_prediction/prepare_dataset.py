#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

VALUE_FEATURE_NAMES = [
    "pressure",
    "wall_shear_x",
    "wall_shear_y",
    "normal_x",
    "normal_y",
]
TARGET_NAMES = ["Cd", "Cl"]


def _load_airfrans_modules():
    try:
        from airfrans import dataset as airfrans_dataset
        from airfrans.simulation import Simulation
    except ImportError as exc:  # pragma: no cover - exercised in real envs
        raise ImportError(
            "AirfRANS is not installed. Run `pip install -e .[airfrans]` first."
        ) from exc
    return airfrans_dataset, Simulation


def _resolve_airfrans_root(raw_root: Path) -> Path:
    candidates = [raw_root, raw_root / "Dataset"]
    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find AirfRANS manifest.json under {raw_root}. "
        "Expected either raw_root/manifest.json or raw_root/Dataset/manifest.json."
    )


def _task_split_names(manifest: dict[str, list[str]], task: str, *, train: bool) -> list[str]:
    task_key = "full" if task == "scarce" and not train else task
    split_key = "train" if train else "test"
    return list(manifest[f"{task_key}_{split_key}"])


def _line_point_weights(lines: np.ndarray, lengths: np.ndarray, n_points: int) -> np.ndarray:
    weights = np.zeros(int(n_points), dtype=np.float64)
    line_indices = np.asarray(lines, dtype=np.int64).reshape(-1, 3)[:, 1:]
    line_lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    for length, (left_idx, right_idx) in zip(line_lengths, line_indices.tolist()):
        half = 0.5 * float(length)
        weights[int(left_idx)] += half
        weights[int(right_idx)] += half
    total = float(weights.sum())
    if total <= 0.0:
        return np.full(int(n_points), 1.0 / max(int(n_points), 1), dtype=np.float64)
    return weights / total


def _write_split(
    names: list[str],
    split_dir: Path,
    *,
    raw_root: Path,
    Simulation,
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        sim = Simulation(str(raw_root), name)
        coords_xy = np.asarray(sim.airfoil_position, dtype=np.float32)
        zeros = np.zeros((coords_xy.shape[0], 1), dtype=np.float32)
        coords = np.concatenate([coords_xy, zeros], axis=1)

        normals = np.asarray(sim.airfoil_normals, dtype=np.float32)
        pressure = np.asarray(sim.airfoil.point_data["p"], dtype=np.float32).reshape(-1, 1)
        wall_shear = np.asarray(sim.wallshearstress(over_airfoil=True, reference=True), dtype=np.float32)
        values = np.concatenate([pressure, wall_shear, normals], axis=1).astype(np.float32)

        base_weights = _line_point_weights(
            np.asarray(sim.airfoil.lines),
            np.asarray(sim.airfoil.cell_data["Length"]),
            coords.shape[0],
        ).astype(np.float32)

        (cd_pack, cl_pack) = sim.force_coefficient(reference=True)
        cd = float(cd_pack[0])
        cl = float(cl_pack[0])
        targets = np.asarray([cd, cl], dtype=np.float32)

        np.savez_compressed(
            split_dir / f"{name}.npz",
            coords=coords,
            values=values,
            base_weights=base_weights,
            targets=targets,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess AirfRANS into sparse boundary force-regression .npz files."
    )
    parser.add_argument("--raw_root", default="data/airfrans_raw")
    parser.add_argument("--output_dir", default="data/airfrans_processed")
    parser.add_argument("--task", default="full", choices=["full", "scarce", "reynolds", "aoa"])
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--download_if_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download the official preprocessed AirfRANS data if raw_root is missing or empty.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    airfrans_dataset, Simulation = _load_airfrans_modules()
    if args.download_if_missing and (not raw_root.exists() or not any(raw_root.iterdir())):
        raw_root.mkdir(parents=True, exist_ok=True)
        airfrans_dataset.download(str(raw_root))

    dataset_root = _resolve_airfrans_root(raw_root)
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))

    train_names = _task_split_names(manifest, args.task, train=True)
    test_names = _task_split_names(manifest, args.task, train=False)

    rng = np.random.default_rng(int(args.seed))
    train_perm = np.arange(len(train_names), dtype=np.int64)
    rng.shuffle(train_perm)
    n_val = max(1, int(round(float(args.val_fraction) * len(train_names))))
    val_idx = np.sort(train_perm[:n_val])
    fit_idx = np.sort(train_perm[n_val:])
    fit_names = [train_names[int(idx)] for idx in fit_idx.tolist()]
    val_names = [train_names[int(idx)] for idx in val_idx.tolist()]

    _write_split(fit_names, output_dir / "train", raw_root=dataset_root, Simulation=Simulation)
    _write_split(val_names, output_dir / "val", raw_root=dataset_root, Simulation=Simulation)
    _write_split(test_names, output_dir / "test", raw_root=dataset_root, Simulation=Simulation)

    metadata = {
        "task": args.task,
        "raw_dataset_root": str(dataset_root),
        "seed": int(args.seed),
        "val_fraction": float(args.val_fraction),
        "value_feature_names": VALUE_FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "n_train": len(fit_names),
        "n_val": len(val_names),
        "n_test": len(test_names),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared AirfRANS force-regression task '{args.task}' into {output_dir}")


if __name__ == "__main__":
    main()
