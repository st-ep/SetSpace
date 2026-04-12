from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from case_studies.ahmedml_surface_forces.benchmark import (
    load_model_checkpoint as load_ahmed_model_checkpoint,
    save_training_artifacts as save_ahmed_artifacts,
)
from case_studies.ahmedml_surface_forces.dataset import AhmedMLSurfaceForceDataset
from case_studies.ahmedml_surface_forces.models import build_force_regressor
from case_studies.point_cloud_consistency.benchmark import (
    load_model_checkpoint as load_point_model_checkpoint,
    save_training_artifacts as save_point_artifacts,
)
from case_studies.point_cloud_consistency.dataset import SyntheticSurfaceSignalDataset
from case_studies.point_cloud_consistency.models import (
    PointCloudMeanRegressor,
    PointCloudWeightedMeanRegressor,
    build_point_cloud_classifier,
    build_point_cloud_regressor,
)
from case_studies.sphere_signal_reconstruction.benchmark import (
    load_model_checkpoint as load_sphere_model_checkpoint,
    save_training_artifacts as save_sphere_artifacts,
)
from case_studies.sphere_signal_reconstruction.dataset import SphereSignalDataset
from case_studies.sphere_signal_reconstruction.models import SphereSignalReconstructor


POINT_WEIGHT_MODES = ["uniform", "knn", "oracle_density", "voronoi"]


def _point_model_config(weight_mode: str) -> dict:
    return {
        "task": "classification",
        "value_input_dim": 1,
        "num_classes": 2,
        "backbone": "set_encoder",
        "n_tokens": 4,
        "token_dim": 8,
        "key_dim": 8,
        "hidden_dim": 16,
        "activation_fn": "gelu",
        "basis_activation": "softplus",
        "value_mode": "mlp_xu",
        "normalize": "total",
        "weight_mode": weight_mode,
        "knn_k": 4,
        "intrinsic_dim": 2,
    }


class CaseStudyModelTests(unittest.TestCase):
    @staticmethod
    def _write_ahmed_fixture(root: Path, *, n_samples: int = 12, n_points: int = 24) -> None:
        rng = np.random.default_rng(0)
        target_names = ["Cd", "Cl"]
        value_feature_names = ["pressure", "normal_x", "normal_y", "normal_z"]
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
            __import__("json").dumps(
                {
                    "target_names": target_names,
                    "value_feature_names": value_feature_names,
                }
            ),
            encoding="utf-8",
        )

    def test_point_cloud_classifier_roundtrip(self):
        dataset = SyntheticSurfaceSignalDataset(n_train=4, n_val=2, n_test=2, seed=0, label_reference_points=64)
        coords = torch.randn(2, 8, 3)
        values = torch.randn(2, 8, 1)
        oracle_weights = torch.full((2, 8), 1.0 / 8.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for weight_mode in POINT_WEIGHT_MODES:
                with self.subTest(weight_mode=weight_mode):
                    model_config = _point_model_config(weight_mode)
                    model = build_point_cloud_classifier(
                        backbone="set_encoder",
                        activation_fn=torch.nn.GELU,
                        **{k: v for k, v in model_config.items() if k not in {"task", "activation_fn", "backbone"}},
                    )
                    out = (
                        model(coords, values, point_weights=oracle_weights)
                        if weight_mode == "oracle_density"
                        else model(coords, values)
                    )
                    self.assertEqual(out.shape, (2, 2))
                    out.sum().backward()

                    checkpoint_dir = Path(tmp_dir) / f"point_{weight_mode}"
                    save_point_artifacts(
                        checkpoint_dir,
                        model=model,
                        dataset=dataset,
                        model_config=model_config,
                        training_config={},
                        training_summary={},
                    )
                    loaded_model, _cfg = load_point_model_checkpoint(checkpoint_dir, torch.device("cpu"))
                    with torch.no_grad():
                        reloaded = (
                            loaded_model(coords, values, point_weights=oracle_weights)
                            if weight_mode == "oracle_density"
                            else loaded_model(coords, values)
                        )
                    self.assertTrue(torch.allclose(out.detach(), reloaded, atol=1e-5, rtol=1e-5))

    def test_point_cloud_regressor_forward_backward(self):
        coords = torch.randn(2, 8, 3)
        values = torch.randn(2, 8, 1)
        oracle_weights = torch.full((2, 8), 1.0 / 8.0)
        for weight_mode in POINT_WEIGHT_MODES:
            with self.subTest(weight_mode=weight_mode):
                model = PointCloudMeanRegressor(
                    value_input_dim=1,
                    n_tokens=4,
                    token_dim=8,
                    key_dim=8,
                    hidden_dim=16,
                    basis_activation="softplus",
                    value_mode="mlp_xu",
                    normalize="total",
                    weight_mode=weight_mode,
                    knn_k=4,
                    intrinsic_dim=2,
                )
                preds = (
                    model(coords, values, point_weights=oracle_weights)
                    if weight_mode == "oracle_density"
                    else model(coords, values)
                )
                self.assertEqual(preds.shape, (2,))
                preds.sum().backward()

    def test_weighted_mean_regressor_roundtrip(self):
        dataset = SyntheticSurfaceSignalDataset(n_train=4, n_val=2, n_test=2, seed=0, label_reference_points=64)
        coords = torch.randn(2, 8, 3)
        values = torch.randn(2, 8, 1)
        point_weights = torch.tensor(
            [[0.10, 0.15, 0.05, 0.20, 0.10, 0.10, 0.15, 0.15], [0.20, 0.10, 0.10, 0.05, 0.15, 0.10, 0.20, 0.10]],
            dtype=torch.float32,
        )

        model = PointCloudWeightedMeanRegressor(value_input_dim=1, weight_mode="oracle_density", knn_k=4, intrinsic_dim=2)
        expected = (point_weights * values.squeeze(-1)).sum(dim=1) / point_weights.sum(dim=1)
        preds = model(coords, values, point_weights=point_weights)
        self.assertTrue(torch.allclose(preds, expected))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "point_weighted_mean"
            save_point_artifacts(
                checkpoint_dir,
                model=model,
                dataset=dataset,
                model_config={
                    "task": "regression",
                    "backbone": "weighted_mean",
                    "output_dim": 1,
                    "value_input_dim": 1,
                    "activation_fn": "gelu",
                    "weight_mode": "oracle_density",
                    "knn_k": 4,
                    "intrinsic_dim": 2,
                },
                training_config={},
                training_summary={},
            )
            loaded_model, _cfg = load_point_model_checkpoint(checkpoint_dir, torch.device("cpu"))
            with torch.no_grad():
                reloaded = loaded_model(coords, values, point_weights=point_weights)
            self.assertTrue(torch.allclose(preds, reloaded))

    def test_pointnext_regressor_roundtrip(self):
        dataset = SyntheticSurfaceSignalDataset(n_train=4, n_val=2, n_test=2, seed=0, label_reference_points=64)
        coords = torch.randn(2, 16, 3)
        values = torch.randn(2, 16, 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_config = {
                "task": "regression",
                "backbone": "pointnext",
                "output_dim": 1,
                "value_input_dim": 1,
                "activation_fn": "gelu",
                "weight_mode": "uniform",
                "knn_k": 4,
                "intrinsic_dim": 2,
                "pointnext_width": 8,
                "pointnext_blocks": [1, 1, 1, 1, 1, 1],
                "pointnext_strides": [1, 2, 2, 2, 2, 1],
                "pointnext_radius": 0.15,
                "pointnext_radius_scaling": 1.5,
                "pointnext_nsample": 8,
                "pointnext_expansion": 2,
                "pointnext_sa_layers": 1,
                "pointnext_sa_use_res": True,
                "pointnext_normalize_dp": True,
                "pointnext_head_hidden_dim": 16,
            }
            model = build_point_cloud_regressor(
                backbone="pointnext",
                output_dim=1,
                value_input_dim=1,
                activation_fn=torch.nn.GELU,
                weight_mode="uniform",
                knn_k=4,
                intrinsic_dim=2,
                pointnext_width=8,
                pointnext_blocks=(1, 1, 1, 1, 1, 1),
                pointnext_strides=(1, 2, 2, 2, 2, 1),
                pointnext_radius=0.15,
                pointnext_radius_scaling=1.5,
                pointnext_nsample=8,
                pointnext_expansion=2,
                pointnext_sa_layers=1,
                pointnext_sa_use_res=True,
                pointnext_normalize_dp=True,
                pointnext_head_hidden_dim=16,
            )
            preds = model(coords, values)
            self.assertEqual(preds.shape, (2,))
            preds.sum().backward()

            checkpoint_dir = Path(tmp_dir) / "pointnext_regression"
            save_point_artifacts(
                checkpoint_dir,
                model=model,
                dataset=dataset,
                model_config=model_config,
                training_config={},
                training_summary={},
            )
            loaded_model, _cfg = load_point_model_checkpoint(checkpoint_dir, torch.device("cpu"))
            model.eval()
            loaded_model.eval()
            with torch.no_grad():
                original = model(coords, values)
                reloaded = loaded_model(coords, values)
            self.assertTrue(torch.allclose(original, reloaded, atol=1e-5, rtol=1e-5))

    def test_sphere_reconstructor_roundtrip(self):
        dataset = SphereSignalDataset(n_train=4, n_val=2, n_test=2, seed=0, query_points=16)
        obs_coords = torch.randn(2, 8, 3)
        obs_values = torch.randn(2, 8, 1)
        query_coords = torch.randn(2, 16, 3)
        oracle_weights = torch.full((2, 8), 1.0 / 8.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            for weight_mode in POINT_WEIGHT_MODES:
                with self.subTest(weight_mode=weight_mode):
                    model_config = {
                        "weight_mode": weight_mode,
                        "n_basis": 4,
                        "key_dim": 8,
                        "value_dim": 8,
                        "encoder_hidden_dim": 16,
                        "trunk_hidden_dim": 16,
                        "n_trunk_layers": 3,
                        "activation_fn": "gelu",
                        "basis_activation": "softplus",
                        "value_mode": "mlp_xu",
                        "encoder_normalize": "total",
                        "use_deeponet_bias": True,
                        "knn_k": 4,
                        "intrinsic_dim": 2,
                    }
                    model = SphereSignalReconstructor(**{k: v for k, v in model_config.items() if k != "activation_fn"})
                    preds = (
                        model(obs_coords, obs_values, query_coords, sensor_weights=oracle_weights)
                        if weight_mode == "oracle_density"
                        else model(obs_coords, obs_values, query_coords)
                    )
                    self.assertEqual(preds.shape, (2, 16))
                    preds.sum().backward()

                    checkpoint_dir = Path(tmp_dir) / f"sphere_{weight_mode}"
                    save_sphere_artifacts(
                        checkpoint_dir,
                        model=model,
                        dataset=dataset,
                        model_config=model_config,
                        training_config={},
                        training_summary={},
                    )
                    loaded_model, _cfg = load_sphere_model_checkpoint(checkpoint_dir, torch.device("cpu"))
                    with torch.no_grad():
                        reloaded = (
                            loaded_model(obs_coords, obs_values, query_coords, sensor_weights=oracle_weights)
                            if weight_mode == "oracle_density"
                            else loaded_model(obs_coords, obs_values, query_coords)
                        )
                    self.assertTrue(torch.allclose(preds.detach(), reloaded, atol=1e-5, rtol=1e-5))

    def test_ahmed_force_regressor_roundtrip(self):
        coords = torch.randn(2, 8, 3)
        values = torch.randn(2, 8, 4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "processed_ahmed"
            root.mkdir(parents=True, exist_ok=True)
            self._write_ahmed_fixture(root)
            dataset = AhmedMLSurfaceForceDataset(root, seed=0, n_train=8, n_val=2, n_test=2)

            model_config = {
                "value_input_dim": dataset.value_input_dim,
                "output_dim": dataset.target_dim,
                "n_tokens": 4,
                "token_dim": 8,
                "key_dim": 8,
                "hidden_dim": 16,
                "activation_fn": "gelu",
                "basis_activation": "softplus",
                "value_mode": "mlp_xu",
                "normalize": "total",
                "weight_mode": "knn",
                "knn_k": 4,
                "intrinsic_dim": 2,
            }
            model = build_force_regressor(
                value_input_dim=dataset.value_input_dim,
                output_dim=dataset.target_dim,
                activation_fn=torch.nn.GELU,
                n_tokens=4,
                token_dim=8,
                key_dim=8,
                hidden_dim=16,
                basis_activation="softplus",
                value_mode="mlp_xu",
                normalize="total",
                weight_mode="knn",
                knn_k=4,
                intrinsic_dim=2,
            )
            preds = model(coords, values)
            self.assertEqual(preds.shape, (2, dataset.target_dim))
            preds.sum().backward()

            checkpoint_dir = Path(tmp_dir) / "ahmed_knn"
            save_ahmed_artifacts(
                checkpoint_dir,
                model=model,
                dataset=dataset,
                model_config=model_config,
                training_config={},
                training_summary={},
            )
            loaded_model, _cfg = load_ahmed_model_checkpoint(checkpoint_dir, torch.device("cpu"))
            with torch.no_grad():
                reloaded = loaded_model(coords, values)
            self.assertTrue(torch.allclose(preds.detach(), reloaded, atol=1e-5, rtol=1e-5))

    def test_oracle_density_requires_explicit_weights(self):
        coords = torch.randn(2, 8, 3)
        values = torch.randn(2, 8, 1)
        queries = torch.randn(2, 16, 3)

        point_model = PointCloudMeanRegressor(
            value_input_dim=1,
            n_tokens=4,
            token_dim=8,
            key_dim=8,
            hidden_dim=16,
            basis_activation="softplus",
            value_mode="mlp_xu",
            normalize="total",
            weight_mode="oracle_density",
            knn_k=4,
            intrinsic_dim=2,
        )
        with self.assertRaises(ValueError):
            point_model(coords, values)

        sphere_model = SphereSignalReconstructor(
            weight_mode="oracle_density",
            n_basis=4,
            key_dim=8,
            value_dim=8,
            encoder_hidden_dim=16,
            trunk_hidden_dim=16,
            n_trunk_layers=3,
            basis_activation="softplus",
            value_mode="mlp_xu",
            encoder_normalize="total",
            use_deeponet_bias=True,
            knn_k=4,
            intrinsic_dim=2,
        )
        with self.assertRaises(ValueError):
            sphere_model(coords, values, queries)


if __name__ == "__main__":
    unittest.main()
