from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _read_models(metrics_path: Path) -> set[str]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return set(payload["models"].keys())


def _read_experiment_config(checkpoint_dir: Path) -> dict:
    return json.loads((checkpoint_dir / "experiment_config.json").read_text(encoding="utf-8"))


class BenchmarkVariantTests(unittest.TestCase):
    @staticmethod
    def _write_ahmed_fixture(root: Path, *, n_samples: int = 12, n_points: int = 24) -> None:
        rng = np.random.default_rng(0)
        for sample_idx in range(n_samples):
            coords = rng.normal(size=(n_points, 3)).astype(np.float32)
            normals = rng.normal(size=(n_points, 3)).astype(np.float32)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-6)
            pressure = rng.normal(size=(n_points, 1)).astype(np.float32)
            base_weights = rng.random(n_points).astype(np.float32)
            base_weights /= base_weights.sum()
            values = np.concatenate([pressure, normals], axis=1).astype(np.float32)
            cd = float((base_weights * pressure[:, 0]).sum())
            cl = float((base_weights * (pressure[:, 0] * coords[:, 2])).sum())
            np.savez_compressed(
                root / f"run{sample_idx:04d}.npz",
                coords=coords,
                values=values,
                base_weights=base_weights,
                targets=np.asarray([cd, cl], dtype=np.float32),
            )
        (root / "metadata.json").write_text(
            json.dumps(
                {
                    "target_names": ["Cd", "Cl"],
                    "value_feature_names": ["pressure", "normal_x", "normal_y", "normal_z"],
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _write_airfrans_fixture(root: Path, *, n_train: int = 8, n_val: int = 2, n_test: int = 2, n_points: int = 32) -> None:
        rng = np.random.default_rng(1)
        metadata = {
            "task": "full",
            "value_feature_names": [
                "pressure",
                "wall_shear_x",
                "wall_shear_y",
                "normal_x",
                "normal_y",
            ],
            "target_names": ["Cd", "Cl"],
        }
        (root / "full").mkdir(parents=True, exist_ok=True)
        for split, count in [("train", n_train), ("val", n_val), ("test", n_test)]:
            split_dir = root / "full" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for sample_idx in range(count):
                theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False, dtype=np.float32)
                x = (0.5 * np.cos(theta) + 0.5).astype(np.float32)
                y = (0.2 * np.sin(theta)).astype(np.float32)
                coords = np.stack([x, y, np.zeros_like(x)], axis=1).astype(np.float32)
                normals = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
                pressure = (0.3 * np.cos(theta) + 0.05 * rng.normal(size=n_points)).astype(np.float32)
                wall_shear = np.stack(
                    [
                        (0.08 * np.sin(theta)).astype(np.float32),
                        (0.04 * np.cos(theta)).astype(np.float32),
                    ],
                    axis=1,
                )
                values = np.concatenate([pressure[:, None], wall_shear, normals], axis=1).astype(np.float32)
                base_weights = rng.random(n_points).astype(np.float32)
                base_weights /= base_weights.sum()
                cd = float((base_weights * (pressure * normals[:, 0] + wall_shear[:, 0])).sum())
                cl = float((base_weights * (pressure * normals[:, 1] + wall_shear[:, 1])).sum())
                np.savez_compressed(
                    split_dir / f"sim_{split}_{sample_idx:04d}.npz",
                    coords=coords,
                    values=values,
                    base_weights=base_weights,
                    targets=np.asarray([cd, cl], dtype=np.float32),
                )
        ((root / "full") / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_point_cloud_benchmark_models(self):
        import case_studies.point_cloud_consistency.run_benchmark as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "point_cloud_benchmark"
            argv = [
                "run_benchmark.py",
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "4",
                "--n_test",
                "4",
                "--label_reference_points",
                "64",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--n_tokens",
                "4",
                "--token_dim",
                "8",
                "--key_dim",
                "8",
                "--hidden_dim",
                "16",
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.object(benchmark, "plot_metrics", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_benchmark_overview", lambda *args, **kwargs: None))
                stack.enter_context(
                    patch.object(benchmark, "plot_qualitative_responses", lambda *args, **kwargs: None)
                )
                stack.enter_context(patch.object(sys, "argv", argv))
                benchmark.main()

            self.assertEqual(
                _read_models(output_dir / "metrics.json"),
                {"uniform", "geometry_aware", "oracle_density", "voronoi"},
            )

    def test_ahmed_benchmark_models(self):
        import case_studies.ahmedml_surface_forces.run_benchmark as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_root = Path(tmp_dir) / "processed_ahmed"
            processed_root.mkdir(parents=True, exist_ok=True)
            self._write_ahmed_fixture(processed_root)
            output_dir = Path(tmp_dir) / "ahmed_benchmark"
            argv = [
                "run_benchmark.py",
                "--processed_root",
                str(processed_root),
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "2",
                "--n_test",
                "2",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--n_tokens",
                "4",
                "--token_dim",
                "8",
                "--key_dim",
                "8",
                "--hidden_dim",
                "16",
            ]
            with patch.object(sys, "argv", argv):
                benchmark.main()

            self.assertEqual(
                _read_models(output_dir / "metrics.json"),
                {"uniform", "geometry_aware"},
            )

    def test_ahmed_pointnext_benchmark_runs_pointnext_only(self):
        import case_studies.ahmedml_surface_forces.run_benchmark as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_root = Path(tmp_dir) / "processed_ahmed"
            processed_root.mkdir(parents=True, exist_ok=True)
            self._write_ahmed_fixture(processed_root)
            output_dir = Path(tmp_dir) / "ahmed_pointnext_benchmark"
            argv = [
                "run_benchmark.py",
                "--processed_root",
                str(processed_root),
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "2",
                "--n_test",
                "2",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--backbone",
                "pointnext",
            ]
            with patch.object(sys, "argv", argv):
                benchmark.main()

            self.assertEqual(_read_models(output_dir / "metrics.json"), {"pointnext"})

    def test_point_cloud_regression_benchmark_models(self):
        import case_studies.point_cloud_consistency.run_mean_regression as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "point_cloud_regression_benchmark"
            argv = [
                "run_mean_regression.py",
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "4",
                "--n_test",
                "4",
                "--label_reference_points",
                "64",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--n_tokens",
                "4",
                "--token_dim",
                "8",
                "--key_dim",
                "8",
                "--hidden_dim",
                "16",
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.object(benchmark, "plot_metrics", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_convergence_metrics", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_benchmark_overview", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(sys, "argv", argv))
                benchmark.main()

            self.assertEqual(
                _read_models(output_dir / "metrics.json"),
                {"uniform", "geometry_aware", "oracle_density", "voronoi"},
            )

    def test_point_cloud_regression_pointnext_runs_pointnext_only(self):
        import case_studies.point_cloud_consistency.run_mean_regression as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "point_cloud_regression_pointnext"
            argv = [
                "run_mean_regression.py",
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "4",
                "--n_test",
                "4",
                "--label_reference_points",
                "64",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--backbone",
                "pointnext",
                "--pointnext_width",
                "8",
                "--pointnext_nsample",
                "8",
                "--pointnext_expansion",
                "2",
                "--pointnext_sa_layers",
                "1",
                "--pointnext_head_hidden_dim",
                "16",
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.object(benchmark, "plot_metrics", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_convergence_metrics", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_benchmark_overview", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(sys, "argv", argv))
                benchmark.main()

            self.assertEqual(_read_models(output_dir / "metrics.json"), {"pointnext"})

    def test_sphere_benchmark_models(self):
        import case_studies.sphere_signal_reconstruction.run_benchmark as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "sphere_benchmark"
            argv = [
                "run_benchmark.py",
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "4",
                "--n_test",
                "4",
                "--query_points",
                "16",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--n_basis",
                "4",
                "--key_dim",
                "8",
                "--value_dim",
                "8",
                "--encoder_hidden_dim",
                "16",
                "--trunk_hidden_dim",
                "16",
                "--n_trunk_layers",
                "3",
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.object(benchmark, "plot_summary", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_convergence", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_qualitative", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(sys, "argv", argv))
                benchmark.main()

            self.assertEqual(
                _read_models(output_dir / "metrics.json"),
                {"uniform", "geometry_aware", "oracle_density", "voronoi_oracle"},
            )

    def test_airfrans_benchmark_models(self):
        import case_studies.airfrans_field_prediction.run_benchmark as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_root = Path(tmp_dir) / "processed_airfrans"
            self._write_airfrans_fixture(processed_root)
            output_dir = Path(tmp_dir) / "airfrans_benchmark"
            argv = [
                "run_benchmark.py",
                "--processed_root",
                str(processed_root),
                "--task",
                "full",
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "2",
                "--n_test",
                "2",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--n_tokens",
                "4",
                "--token_dim",
                "8",
                "--key_dim",
                "8",
                "--hidden_dim",
                "16",
            ]
            with patch.object(sys, "argv", argv):
                benchmark.main()

            self.assertEqual(_read_models(output_dir / "metrics.json"), {"uniform", "geometry_aware"})

    def test_airfrans_pointnext_benchmark_runs_pointnext_only(self):
        import case_studies.airfrans_field_prediction.run_benchmark as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_root = Path(tmp_dir) / "processed_airfrans"
            self._write_airfrans_fixture(processed_root)
            output_dir = Path(tmp_dir) / "airfrans_pointnext_benchmark"
            argv = [
                "run_benchmark.py",
                "--processed_root",
                str(processed_root),
                "--task",
                "full",
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "2",
                "--n_test",
                "2",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--backbone",
                "pointnext",
                "--pointnext_width",
                "8",
                "--pointnext_nsample",
                "8",
                "--pointnext_expansion",
                "2",
                "--pointnext_sa_layers",
                "1",
                "--pointnext_head_hidden_dim",
                "16",
            ]
            with patch.object(sys, "argv", argv):
                benchmark.main()

            self.assertEqual(_read_models(output_dir / "metrics.json"), {"pointnext"})

    def test_pointnext_benchmark_runs_uniform_only(self):
        import case_studies.point_cloud_consistency.run_benchmark as benchmark

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "point_cloud_pointnext_benchmark"
            argv = [
                "run_benchmark.py",
                "--output_dir",
                str(output_dir),
                "--device",
                "cpu",
                "--seed",
                "0",
                "--n_train",
                "8",
                "--n_val",
                "4",
                "--n_test",
                "4",
                "--label_reference_points",
                "64",
                "--train_points",
                "8",
                "--steps",
                "1",
                "--batch_size",
                "2",
                "--eval_every",
                "1",
                "--val_objects",
                "2",
                "--test_objects",
                "2",
                "--reference_points",
                "16",
                "--n_resamples",
                "1",
                "--point_counts",
                "8",
                "--sampling_modes",
                "uniform",
                "--backbone",
                "pointnext",
            ]
            with ExitStack() as stack:
                stack.enter_context(patch.object(benchmark, "plot_metrics", lambda *args, **kwargs: None))
                stack.enter_context(patch.object(benchmark, "plot_benchmark_overview", lambda *args, **kwargs: None))
                stack.enter_context(
                    patch.object(benchmark, "plot_qualitative_responses", lambda *args, **kwargs: None)
                )
                stack.enter_context(patch.object(sys, "argv", argv))
                benchmark.main()

            self.assertEqual(_read_models(output_dir / "metrics.json"), {"uniform"})


if __name__ == "__main__":
    unittest.main()
