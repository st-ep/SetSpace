from __future__ import annotations

import torch

from .mmq import solve_global_moment2_weights


def _coerce_2d(
    xs: torch.Tensor,
    tensor: torch.Tensor | None,
    *,
    as_bool: bool = False,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.dim() == 3 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)
    expected_shape = xs.shape[:2]
    if tensor.shape != expected_shape:
        raise ValueError(f"{tensor.shape=} must be (B, N) = {expected_shape}")
    if as_bool:
        return tensor.to(device=xs.device).bool()
    return tensor.to(device=xs.device, dtype=xs.dtype)


def _coerce_mask(
    xs: torch.Tensor,
    sensor_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    return _coerce_2d(xs, sensor_mask, as_bool=True)


def _coerce_weights(
    xs: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor | None:
    return _coerce_2d(xs, weights)


def infer_uniform_weights(
    xs: torch.Tensor,
    sensor_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return uniform weights, zeroing out masked elements."""
    if xs.dim() != 3:
        raise ValueError(f"xs must be 3D (B, N, dx), got {xs.shape=}")
    mask = _coerce_mask(xs, sensor_mask)
    w = torch.ones((xs.shape[0], xs.shape[1]), device=xs.device, dtype=xs.dtype)
    if mask is not None:
        w = w * mask.to(dtype=xs.dtype)
    return w


def infer_quadrature_weights(
    xs: torch.Tensor,
    sensor_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Infer geometry-aware aggregation weights from element coordinates.

    - 1D: trapezoidal / Voronoi-style cell widths from sorted coordinates.
    - Higher-D: uniform weights over valid elements.
    """
    if xs.dim() != 3:
        raise ValueError(f"xs must be 3D (B, N, dx), got {xs.shape=}")

    batch_size, n_sensors, dx = xs.shape
    device, dtype = xs.device, xs.dtype

    mask = _coerce_mask(xs, sensor_mask)

    if dx != 1:
        w = torch.ones((batch_size, n_sensors), device=device, dtype=dtype)
        if mask is not None:
            w = w * mask.to(dtype)
        return w

    x = xs[..., 0].detach()
    w = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)

    if mask is None:
        if n_sensors == 1:
            return torch.ones((batch_size, n_sensors), device=device, dtype=dtype)

        x_sorted, sort_idx = torch.sort(x, dim=1)
        dx_sorted = (x_sorted[:, 1:] - x_sorted[:, :-1]).clamp_min(0.0)

        w_sorted = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)
        w_sorted[:, 0] = dx_sorted[:, 0] / 2.0
        w_sorted[:, -1] = dx_sorted[:, -1] / 2.0
        if n_sensors > 2:
            w_sorted[:, 1:-1] = (dx_sorted[:, :-1] + dx_sorted[:, 1:]) / 2.0

        w.scatter_(1, sort_idx, w_sorted)
        return w.clamp_min(eps)

    # Vectorized masked trapezoidal weights: push invalid elements to +inf
    # so they sort to the end, compute widths on the padded tensor, then
    # zero out invalid positions.
    mask_f = mask.to(dtype)
    sentinel = x.max() + 1.0
    x_padded = torch.where(mask, x, sentinel)
    x_sorted, sort_idx = torch.sort(x_padded, dim=1)
    dx_sorted = (x_sorted[:, 1:] - x_sorted[:, :-1]).clamp_min(0.0)

    w_sorted = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)
    w_sorted[:, 0] = dx_sorted[:, 0] / 2.0
    if n_sensors > 1:
        w_sorted[:, -1] = dx_sorted[:, -1] / 2.0
    if n_sensors > 2:
        w_sorted[:, 1:-1] = (dx_sorted[:, :-1] + dx_sorted[:, 1:]) / 2.0

    # Map sorted weights back and zero out invalid positions
    w.scatter_(1, sort_idx, w_sorted)
    w = w * mask_f

    # Handle single-valid-element rows: set weight to 1.0
    n_valid = mask_f.sum(dim=1)
    single_mask = (n_valid == 1).unsqueeze(1) & mask
    w = torch.where(single_mask, torch.ones_like(w), w)

    return w.clamp_min(0.0)


def infer_knn_density_weights(
    xs: torch.Tensor,
    sensor_mask: torch.Tensor | None = None,
    *,
    k: int = 8,
    intrinsic_dim: int = 2,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Estimate measure-aware weights from local point density via kNN radii.

    For sampled manifolds, the k-th nearest-neighbor radius scales like
    density^(-1 / intrinsic_dim), so r_k^intrinsic_dim provides a simple local
    cell-volume proxy.
    """
    if xs.dim() != 3:
        raise ValueError(f"xs must be 3D (B, N, dx), got {xs.shape=}")
    if intrinsic_dim < 1:
        raise ValueError(f"intrinsic_dim must be >= 1, got {intrinsic_dim}")

    batch_size, n_sensors, _ = xs.shape
    device, dtype = xs.device, xs.dtype
    mask = _coerce_mask(xs, sensor_mask)

    if mask is None and n_sensors > 1:
        # Fully vectorized unmasked path
        dists = torch.cdist(xs, xs)
        dists.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        neighbor_rank = min(max(int(k), 1), n_sensors - 1)
        kth_dists = torch.topk(dists, k=neighbor_rank, largest=False).values[:, :, -1]
        weights = kth_dists.clamp_min(eps).pow(int(intrinsic_dim))
        if normalize:
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
        return weights

    weights = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)

    if mask is None:
        # n_sensors <= 1
        if n_sensors == 1:
            weights.fill_(1.0)
        return weights

    for b in range(batch_size):
        idx_valid = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        n_valid = int(idx_valid.numel())
        if n_valid == 0:
            continue
        if n_valid == 1:
            weights[b, idx_valid[0]] = 1.0
            continue

        x_valid = xs[b, idx_valid]
        dists = torch.cdist(x_valid, x_valid)
        dists.fill_diagonal_(float("inf"))

        neighbor_rank = min(max(int(k), 1), n_valid - 1)
        kth_dists = torch.topk(dists, k=neighbor_rank, largest=False).values[:, -1]
        raw = kth_dists.clamp_min(eps).pow(int(intrinsic_dim))

        if normalize:
            raw = raw / raw.sum().clamp_min(eps)

        weights[b, idx_valid] = raw

    return weights


def infer_moment2_weights(
    xs: torch.Tensor,
    sensor_mask: torch.Tensor | None = None,
    *,
    anchor_ratio: float = 0.125,
    max_anchors: int = 32,
    patch_k: int = 16,
    tangent_k: int = 16,
    rank_tol: float = 1e-6,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict]]:
    if xs.dim() != 3:
        raise ValueError(f"xs must be 3D (B, N, dx), got {xs.shape=}")

    batch_size, n_sensors, _ = xs.shape
    device, dtype = xs.device, xs.dtype
    mask = _coerce_mask(xs, sensor_mask)

    weights = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)
    diagnostics: list[dict] = []

    for b in range(batch_size):
        if mask is None:
            idx_valid = torch.arange(n_sensors, device=device)
        else:
            idx_valid = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        n_valid = int(idx_valid.numel())
        if n_valid == 0:
            diagnostics.append(
                {
                    "batch_index": int(b),
                    "anchor_count": 0,
                    "fallback": True,
                    "residual": 0.0,
                    "full_row_rank": False,
                    "n_valid": 0,
                }
            )
            continue

        sample_coords = xs[b, idx_valid].detach()
        sample_weights, sample_diag = solve_global_moment2_weights(
            sample_coords,
            anchor_ratio=anchor_ratio,
            max_anchors=max_anchors,
            patch_k=patch_k,
            tangent_k=tangent_k,
            rank_tol=rank_tol,
            eps=eps,
            return_diagnostics=True,
        )
        weights[b, idx_valid] = sample_weights.to(device=device, dtype=dtype)
        sample_diag["batch_index"] = int(b)
        sample_diag["n_valid"] = int(n_valid)
        diagnostics.append(sample_diag)

    if not return_diagnostics:
        return weights
    return weights, diagnostics
