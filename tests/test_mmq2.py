from __future__ import annotations

import unittest

import torch

from case_studies.point_cloud_consistency.models import build_point_cloud_classifier
from set_encoders import WeightedSetEncoder, infer_knn_density_weights, infer_moment2_weights, infer_uniform_weights
from set_encoders.mmq import _local_chart, solve_local_moment2_weights


def _sample_anisotropic_disk(n_points: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    radius = torch.sqrt(torch.rand(n_points, generator=generator))
    theta = 2.0 * torch.pi * torch.rand(n_points, generator=generator)
    x = radius * torch.cos(theta)
    y = 0.25 * radius * torch.sin(theta)
    return torch.stack([x, y, torch.zeros_like(x)], dim=-1)


class MMQ2Tests(unittest.TestCase):
    def test_local_moment2_weights_match_degree2_constraints(self) -> None:
        center = torch.zeros((1, 1, 3), dtype=torch.float32)
        generator = torch.Generator().manual_seed(0)
        xy = torch.randn((1, 1, 16, 2), generator=generator, dtype=torch.float32) * torch.tensor([0.2, 0.1])
        grouped = torch.cat([xy, torch.zeros((1, 1, 16, 1), dtype=torch.float32)], dim=-1)

        weights = solve_local_moment2_weights(center, grouped)
        patch_xy = grouped[0, 0, :, :2]
        u = patch_xy[:, 0]
        v = patch_xy[:, 1]
        w = weights[0, 0]
        local_coords, _ = _local_chart(center[0, 0], grouped[0, 0], tangent_k=16, eps=1e-8)
        lu = local_coords[:, 0]
        lv = local_coords[:, 1]

        self.assertAlmostEqual(float(w.sum().item()), 1.0, places=5)
        self.assertLess(abs(float((w * u).sum().item())), 1e-5)
        self.assertLess(abs(float((w * v).sum().item())), 1e-5)
        self.assertLess(abs(float((w * u * v).sum().item())), 1e-5)
        self.assertLess(abs(float((w * (u.square() - v.square())).sum().item())), 1e-5)
        self.assertLess(abs(float((w * (lu.square() + lv.square())).sum().item()) - 0.5), 1e-5)

    def test_local_chart_recovers_sphere_patch(self) -> None:
        generator = torch.Generator().manual_seed(1)
        xy = 0.08 * torch.randn((32, 2), generator=generator, dtype=torch.float32)
        z = torch.sqrt((1.0 - xy.square().sum(dim=1)).clamp_min(0.0))
        patch = torch.cat([xy, z.unsqueeze(-1)], dim=-1)
        center = patch[0]

        local_coords, rel = _local_chart(center, patch, tangent_k=16, eps=1e-8)
        tangent_ref = rel[:, :2] / rel.norm(dim=-1).max().clamp_min(1e-8)
        distance_error = torch.mean(torch.abs(torch.cdist(local_coords, local_coords) - torch.cdist(tangent_ref, tangent_ref)))

        self.assertEqual(tuple(local_coords.shape), (32, 2))
        self.assertLess(float(distance_error.item()), 5e-3)

    def test_global_moment2_improves_quadratic_anisotropy(self) -> None:
        moment2_errors = []
        uniform_errors = []
        knn_errors = []
        for seed in range(4):
            coords = _sample_anisotropic_disk(96, seed=seed).unsqueeze(0)
            moment2 = infer_moment2_weights(coords)[0]
            uniform = infer_uniform_weights(coords)[0]
            uniform = uniform / uniform.sum()
            knn = infer_knn_density_weights(coords, k=8, intrinsic_dim=2, normalize=True)[0]

            x = coords[0, :, 0]
            y = coords[0, :, 1]
            moment = x * y
            anis = x.square() - y.square()

            def error(weights: torch.Tensor) -> float:
                return abs(float((weights * moment).sum().item())) + abs(float((weights * anis).sum().item()))

            moment2_errors.append(error(moment2))
            uniform_errors.append(error(uniform))
            knn_errors.append(error(knn))

        self.assertLess(sum(moment2_errors) / len(moment2_errors), sum(uniform_errors) / len(uniform_errors))
        self.assertLess(sum(moment2_errors) / len(moment2_errors), sum(knn_errors) / len(knn_errors))

    def test_weighted_set_encoder_accepts_signed_moment2_weights(self) -> None:
        coords = _sample_anisotropic_disk(48, seed=7).unsqueeze(0)
        values = torch.randn((1, 48, 1), generator=torch.Generator().manual_seed(7))
        weights = infer_moment2_weights(coords)

        encoder = WeightedSetEncoder(
            n_tokens=8,
            coord_dim=3,
            value_input_dim=1,
            output_dim=16,
            key_dim=16,
            value_dim=16,
            hidden_dim=32,
            activation_fn=torch.nn.GELU,
            normalize="total",
        )
        out = encoder(coords, values, element_weights=weights)
        self.assertEqual(tuple(out.shape), (1, 8, 16))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_pointnext_moment2_forward(self) -> None:
        generator = torch.Generator().manual_seed(3)
        coords = torch.randn((2, 128, 3), generator=generator)
        values = torch.randn((2, 128, 1), generator=generator)
        model = build_point_cloud_classifier(
            backbone="pointnext",
            num_classes=4,
            value_input_dim=1,
            weight_mode="moment2",
        )
        logits = model(coords, values)
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
