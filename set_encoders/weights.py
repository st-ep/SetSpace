from __future__ import annotations

import torch


def _coerce_mask(
    xs: torch.Tensor,
    sensor_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if sensor_mask is None:
        return None
    if sensor_mask.dim() == 3 and sensor_mask.shape[-1] == 1:
        sensor_mask = sensor_mask.squeeze(-1)
    expected_shape = xs.shape[:2]
    if sensor_mask.shape != expected_shape:
        raise ValueError(f"{sensor_mask.shape=} must be (B, N) = {expected_shape}")
    return sensor_mask.to(device=xs.device).bool()


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

    for b in range(batch_size):
        idx_valid = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
        n_valid = int(idx_valid.numel())
        if n_valid == 0:
            continue
        if n_valid == 1:
            w[b, idx_valid[0]] = 1.0
            continue

        x_valid = x[b, idx_valid]
        x_sorted, perm = torch.sort(x_valid)
        idx_sorted = idx_valid[perm]

        dx_sorted = (x_sorted[1:] - x_sorted[:-1]).clamp_min(0.0)
        w_sorted = torch.zeros((n_valid,), device=device, dtype=dtype)
        w_sorted[0] = dx_sorted[0] / 2.0
        w_sorted[-1] = dx_sorted[-1] / 2.0
        if n_valid > 2:
            w_sorted[1:-1] = (dx_sorted[:-1] + dx_sorted[1:]) / 2.0

        w[b, idx_sorted] = w_sorted

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
    weights = torch.zeros((batch_size, n_sensors), device=device, dtype=dtype)

    for b in range(batch_size):
        if mask is None:
            idx_valid = torch.arange(n_sensors, device=device)
        else:
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
