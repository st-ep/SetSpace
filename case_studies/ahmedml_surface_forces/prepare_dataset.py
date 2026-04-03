#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def _require_pyvista():
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "prepare_dataset.py requires the optional 'pyvista' dependency. "
            "Install it with `pip install pyvista` before preparing AhmedML."
        ) from exc
    return pv


PRESSURE_CANDIDATES = ["CpT", "Cp", "cp", "pMean", "p", "pressure"]
SHEAR_CANDIDATES = ["wallShearStress", "wallShearStressMean", "Cf", "cf", "skinFriction"]
NORMAL_CANDIDATES = ["Normals", "normals", "Normal", "normal"]
TARGET_ALIASES = {
    "Cd": ["Cd", "cd", "drag", "drag_coeff", "Cx"],
    "Cl": ["Cl", "cl", "lift", "lift_coeff", "Cz"],
    "Cs": ["Cs", "cs", "sideforce", "Cy"],
    "Cmz": ["Cmz", "cmz", "yaw_moment", "Mz"],
    "Cmy": ["Cmy", "cmy", "pitch_moment", "My"],
}


def _read_force_targets(csv_path: Path, target_names: list[str]) -> np.ndarray:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    normalized = {str(k).strip(): v for k, v in row.items()}
    lower_map = {k.lower(): v for k, v in normalized.items()}
    out = []
    for target_name in target_names:
        candidates = TARGET_ALIASES.get(target_name, [target_name])
        value = None
        for candidate in candidates:
            if candidate in normalized:
                value = normalized[candidate]
                break
            candidate_lower = candidate.lower()
            if candidate_lower in lower_map:
                value = lower_map[candidate_lower]
                break
        if value is None:
            raise KeyError(f"Could not find target '{target_name}' in {csv_path.name}. Available columns: {list(normalized)}")
        out.append(float(value))
    return np.asarray(out, dtype=np.float32)


def _select_array(data_map, candidates: list[str], *, ndim: int | None = None) -> tuple[str, np.ndarray] | None:
    for name in candidates:
        if name not in data_map:
            continue
        arr = np.asarray(data_map[name])
        if arr.ndim == 1:
            arr = arr[:, None]
        if ndim is not None and arr.shape[1] != int(ndim):
            continue
        return name, arr.astype(np.float32)
    return None


def _compute_cell_areas(mesh) -> np.ndarray:
    with_areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)
    return np.clip(np.asarray(with_areas.cell_data["Area"], dtype=np.float64), 1e-12, None)


def _downsample_indices(weights: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if weights.shape[0] <= max_points:
        return np.arange(weights.shape[0], dtype=np.int64)
    probs = weights / weights.sum()
    return rng.choice(weights.shape[0], size=int(max_points), replace=False, p=probs)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a compact AhmedML surface-force benchmark dataset.")
    parser.add_argument("--raw_root", required=True, help="Root directory containing run*/boundary_*.vtp and force_mom*.csv")
    parser.add_argument("--output_dir", required=True, help="Directory where processed .npz samples will be written")
    parser.add_argument("--target_names", nargs="+", default=["Cd", "Cl"])
    parser.add_argument("--max_points_per_sample", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_constref_targets", action="store_true")
    return parser.parse_args()


def _run_index_from_name(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    if match is None:
        raise ValueError(f"Could not parse run index from directory name: {name}")
    return int(match.group(1))


def main():
    args = parse_args()
    pv = _require_pyvista()
    raw_root = Path(args.raw_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    run_dirs = sorted([path for path in raw_root.iterdir() if path.is_dir() and path.name.startswith("run")])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {raw_root}")

    value_feature_names: list[str] | None = None
    processed_paths: list[str] = []
    for run_dir in run_dirs:
        run_id = run_dir.name
        run_index = _run_index_from_name(run_id)
        boundary_path = run_dir / f"boundary_{run_index}.vtp"
        if not boundary_path.exists():
            boundary_path = run_dir / f"boundary_{run_index}.vtk"
        if not boundary_path.exists():
            raise FileNotFoundError(f"Could not find boundary file for {run_id}")

        target_file = "force_mom_constref_" if args.use_constref_targets else "force_mom_"
        force_path = run_dir / f"{target_file}{run_index}.csv"
        if not force_path.exists():
            raise FileNotFoundError(f"Could not find force/moment CSV for {run_id}")

        surface = pv.read(boundary_path).compute_normals(point_normals=False, cell_normals=True, inplace=False)
        cell_areas = _compute_cell_areas(surface)
        centers = surface.cell_centers()

        normals_result = _select_array(surface.cell_data, NORMAL_CANDIDATES, ndim=3)
        if normals_result is None:
            raise KeyError(f"Could not find or compute cell normals for {boundary_path}")
        _, normals = normals_result

        pressure_result = _select_array(surface.cell_data, PRESSURE_CANDIDATES, ndim=1)
        if pressure_result is None:
            raise KeyError(
                f"Could not find a pressure-like cell array in {boundary_path}. "
                f"Tried {PRESSURE_CANDIDATES}."
            )
        pressure_name, pressure = pressure_result

        shear_result = _select_array(surface.cell_data, SHEAR_CANDIDATES, ndim=3)
        shear_name = None
        shear = None
        if shear_result is not None:
            shear_name, shear = shear_result

        values_list = [pressure, normals]
        feature_names = [pressure_name, "normal_x", "normal_y", "normal_z"]
        if shear is not None:
            values_list.append(shear)
            feature_names.extend([f"{shear_name}_x", f"{shear_name}_y", f"{shear_name}_z"])
        values = np.concatenate(values_list, axis=1).astype(np.float32)

        points = np.asarray(centers.points, dtype=np.float32)
        keep_idx = _downsample_indices(cell_areas, max_points=args.max_points_per_sample, rng=rng)
        keep_idx = np.sort(keep_idx)
        points = points[keep_idx]
        values = values[keep_idx]
        cell_areas = cell_areas[keep_idx]
        cell_areas = cell_areas / cell_areas.sum()
        targets = _read_force_targets(force_path, list(args.target_names))

        sample_path = output_dir / f"{run_id}.npz"
        np.savez_compressed(
            sample_path,
            coords=points.astype(np.float32),
            values=values.astype(np.float32),
            base_weights=cell_areas.astype(np.float32),
            targets=targets.astype(np.float32),
        )
        processed_paths.append(sample_path.name)
        if value_feature_names is None:
            value_feature_names = feature_names

    metadata = {
        "target_names": list(args.target_names),
        "value_feature_names": list(value_feature_names or []),
        "files": processed_paths,
        "seed": int(args.seed),
        "max_points_per_sample": int(args.max_points_per_sample),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Prepared {len(processed_paths)} AhmedML samples under {output_dir}")


if __name__ == "__main__":
    main()
