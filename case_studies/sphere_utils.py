from __future__ import annotations

import torch


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def sample_uniform_sphere(n_points: int, generator: torch.Generator) -> torch.Tensor:
    return normalize(torch.randn((int(n_points), 3), generator=generator, dtype=torch.float32))


def score_candidates(points: torch.Tensor, mode: str, generator: torch.Generator) -> torch.Tensor:
    z = points[:, 2]

    if mode == "uniform":
        return torch.ones(points.shape[0], dtype=points.dtype)
    if mode == "polar":
        return 0.05 + torch.exp(2.8 * z)
    if mode == "equatorial":
        return 0.05 + torch.exp(-10.0 * z.square())
    if mode == "clustered":
        centers = sample_uniform_sphere(2, generator)
        scores = torch.exp(7.0 * (points @ centers.T))
        return 0.05 + scores.sum(dim=1)
    if mode == "hemisphere":
        view_dir = sample_uniform_sphere(1, generator).squeeze(0)
        visibility = points @ view_dir
        return 0.05 + torch.sigmoid(8.0 * visibility)

    raise ValueError(f"Unknown sampling mode: {mode}")


def sample_surface_points(
    n_points: int,
    sampling_mode: str,
    generator: torch.Generator,
    candidate_multiplier: int = 10,
) -> torch.Tensor:
    candidate_count = max(int(n_points) * int(candidate_multiplier), 2048)
    candidates = sample_uniform_sphere(candidate_count, generator)
    scores = score_candidates(candidates, sampling_mode, generator).clamp_min(1e-6)
    sample_idx = torch.multinomial(scores, int(n_points), replacement=False, generator=generator)
    return candidates[sample_idx]
