from __future__ import annotations

import math

import torch


def _normalized_uniform_weights(
    n_points: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if int(n_points) <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    return torch.full((int(n_points),), 1.0 / float(n_points), device=device, dtype=dtype)


def farthest_point_sample(coords: torch.Tensor, n_samples: int) -> torch.Tensor:
    if coords.dim() != 2:
        raise ValueError(f"coords must be 2D (N, D), got {coords.shape=}")

    n_points = coords.shape[0]
    n_samples = int(min(max(int(n_samples), 1), n_points))
    device = coords.device

    centroids = torch.zeros((n_samples,), dtype=torch.long, device=device)
    min_dist = torch.full((n_points,), float("inf"), device=device)
    centroid_guess = coords.mean(dim=0, keepdim=True)
    farthest = ((coords - centroid_guess).square().sum(dim=-1)).max(dim=0).indices

    for i in range(n_samples):
        centroids[i] = farthest
        centroid = coords[farthest].unsqueeze(0)
        dists = (coords - centroid).square().sum(dim=-1)
        min_dist = torch.minimum(min_dist, dists)
        farthest = min_dist.max(dim=0).indices

    return centroids


def knn_indices(query_coords: torch.Tensor, support_coords: torch.Tensor, k: int) -> torch.Tensor:
    if query_coords.dim() != 2 or support_coords.dim() != 2:
        raise ValueError(
            f"query_coords and support_coords must be 2D, got {query_coords.shape=} and {support_coords.shape=}"
        )
    k = int(min(max(int(k), 1), support_coords.shape[0]))
    dists = torch.cdist(query_coords, support_coords)
    return torch.topk(dists, k=k, largest=False).indices


def _matrix_rank(matrix: torch.Tensor, rank_tol: float) -> int:
    if matrix.numel() == 0:
        return 0
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0:
        return 0
    threshold = float(rank_tol) * float(singular_values.max().item())
    return int((singular_values > threshold).sum().item())


def _local_chart(
    center: torch.Tensor,
    patch_coords: torch.Tensor,
    *,
    tangent_k: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel = patch_coords - center.unsqueeze(0)
    if rel.shape[0] == 0:
        return torch.zeros((0, 2), device=patch_coords.device, dtype=patch_coords.dtype), rel

    tangent_k = int(min(max(int(tangent_k), 3), rel.shape[0]))
    dists = rel.square().sum(dim=-1)
    tan_idx = torch.topk(dists, k=tangent_k, largest=False).indices
    pca_points = rel[tan_idx]
    cov = pca_points.transpose(0, 1) @ pca_points / float(max(tangent_k, 1))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)[:2]
    basis = eigvecs[:, order]
    local = rel @ basis
    radius = rel.norm(dim=-1).max().clamp_min(float(eps))
    return local / radius, rel


def _moment_rows(local_coords: torch.Tensor) -> torch.Tensor:
    u = local_coords[:, 0]
    v = local_coords[:, 1]
    return torch.stack(
        [
            torch.ones_like(u),
            u,
            v,
            u * v,
            u.square() - v.square(),
            u.square() + v.square(),
        ],
        dim=0,
    )


def _solve_min_norm(
    rows: torch.Tensor,
    rhs: torch.Tensor,
    *,
    rank_tol: float,
    eps: float,
) -> tuple[torch.Tensor, dict]:
    rows64 = rows.to(dtype=torch.float64)
    rhs64 = rhs.to(device=rows.device, dtype=torch.float64)
    gram = rows64 @ rows64.transpose(0, 1)
    rank = _matrix_rank(gram, rank_tol)
    full_row_rank = rank == rows.shape[0]

    if full_row_rank:
        try:
            y = torch.linalg.solve(gram, rhs64)
            sol = rows64.transpose(0, 1) @ y
        except RuntimeError:
            full_row_rank = False
            sol = torch.linalg.pinv(rows64, rtol=float(rank_tol)) @ rhs64
    else:
        sol = torch.linalg.pinv(rows64, rtol=float(rank_tol)) @ rhs64

    residual = float(torch.linalg.vector_norm(rows64 @ sol - rhs64).item())
    return sol.to(dtype=rows.dtype), {
        "rank": int(rank),
        "full_row_rank": bool(full_row_rank),
        "residual": residual,
    }


def solve_local_moment2_weights(
    centers: torch.Tensor,
    grouped_coords: torch.Tensor,
    *,
    tangent_k: int = 16,
    rank_tol: float = 1e-6,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    if centers.dim() != 3 or grouped_coords.dim() != 4:
        raise ValueError(
            f"centers must be (B, M, D) and grouped_coords must be (B, M, K, D), got {centers.shape=} and {grouped_coords.shape=}"
        )
    if centers.shape[0] != grouped_coords.shape[0] or centers.shape[1] != grouped_coords.shape[1]:
        raise ValueError(f"centers and grouped_coords must share (B, M), got {centers.shape=} and {grouped_coords.shape=}")

    batch_size, n_query, n_neighbors, _ = grouped_coords.shape
    weights = torch.zeros((batch_size, n_query, n_neighbors), device=grouped_coords.device, dtype=grouped_coords.dtype)
    fallback_mask = torch.zeros((batch_size, n_query), device=grouped_coords.device, dtype=torch.bool)
    residuals = torch.zeros((batch_size, n_query), device=grouped_coords.device, dtype=grouped_coords.dtype)

    rhs_template = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.5], device=grouped_coords.device, dtype=grouped_coords.dtype)

    for b in range(batch_size):
        for q in range(n_query):
            if n_neighbors < 6 or centers.shape[-1] < 2:
                weights[b, q] = _normalized_uniform_weights(n_neighbors, device=grouped_coords.device, dtype=grouped_coords.dtype)
                fallback_mask[b, q] = True
                continue

            center = centers[b, q]
            patch = grouped_coords[b, q]
            local_coords, _ = _local_chart(center, patch, tangent_k=tangent_k, eps=eps)
            rows = _moment_rows(local_coords)
            rank = _matrix_rank(rows, rank_tol)
            if rank < rows.shape[0]:
                weights[b, q] = _normalized_uniform_weights(n_neighbors, device=grouped_coords.device, dtype=grouped_coords.dtype)
                fallback_mask[b, q] = True
                continue

            sol, diagnostics = _solve_min_norm(rows, rhs_template, rank_tol=rank_tol, eps=eps)
            weight_sum = float(sol.sum().item())
            if math.isfinite(weight_sum) and abs(weight_sum) > float(eps):
                sol = sol / weight_sum
            weights[b, q] = sol
            residuals[b, q] = float(diagnostics["residual"])

    if not return_diagnostics:
        return weights
    return weights, {
        "fallback_mask": fallback_mask,
        "residuals": residuals,
    }


def solve_global_moment2_weights(
    coords: torch.Tensor,
    *,
    anchor_ratio: float = 0.125,
    max_anchors: int = 32,
    patch_k: int = 16,
    tangent_k: int = 16,
    rank_tol: float = 1e-6,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    if coords.dim() != 2:
        raise ValueError(f"coords must be 2D (N, D), got {coords.shape=}")

    n_points, coord_dim = coords.shape
    if n_points == 0:
        weights = torch.zeros((0,), device=coords.device, dtype=coords.dtype)
        diagnostics = {"anchor_count": 0, "fallback": True, "residual": 0.0, "full_row_rank": False}
        return (weights, diagnostics) if return_diagnostics else weights
    if n_points < 6 or coord_dim < 2 or patch_k < 6:
        weights = _normalized_uniform_weights(n_points, device=coords.device, dtype=coords.dtype)
        diagnostics = {"anchor_count": 0, "fallback": True, "residual": 0.0, "full_row_rank": False}
        return (weights, diagnostics) if return_diagnostics else weights

    requested_anchors = min(
        max(int(round(float(anchor_ratio) * float(n_points))), 4),
        int(max_anchors),
        int(n_points),
    )
    requested_anchors = min(requested_anchors, max((n_points - 1) // 5, 1))
    anchor_idx_all = farthest_point_sample(coords, requested_anchors)

    best_solution = None
    best_diag = None
    patch_k = int(min(max(int(patch_k), 6), n_points))

    for anchor_count in range(int(requested_anchors), 0, -1):
        anchor_idx = anchor_idx_all[:anchor_count]
        anchor_coords = coords[anchor_idx]
        neighbor_idx = knn_indices(anchor_coords, coords, patch_k)

        n_rows = 1 + 6 * anchor_count
        n_cols = n_points + anchor_count
        system = torch.zeros((n_rows, n_cols), device=coords.device, dtype=coords.dtype)
        rhs = torch.zeros((n_rows,), device=coords.device, dtype=coords.dtype)
        system[0, :n_points] = 1.0
        rhs[0] = 1.0

        row_offset = 1
        for anchor_offset in range(anchor_count):
            patch = coords[neighbor_idx[anchor_offset]]
            center = anchor_coords[anchor_offset]
            local_coords, _ = _local_chart(center, patch, tangent_k=tangent_k, eps=eps)
            u = local_coords[:, 0]
            v = local_coords[:, 1]
            patch_cols = neighbor_idx[anchor_offset]
            mass_col = n_points + anchor_offset

            system[row_offset, patch_cols] = 1.0
            system[row_offset, mass_col] = -1.0
            system[row_offset + 1, patch_cols] = u
            system[row_offset + 2, patch_cols] = v
            system[row_offset + 3, patch_cols] = u * v
            system[row_offset + 4, patch_cols] = u.square() - v.square()
            system[row_offset + 5, patch_cols] = u.square() + v.square()
            system[row_offset + 5, mass_col] = -0.5
            row_offset += 6

        rank = _matrix_rank(system, rank_tol)
        if rank == n_rows:
            solution, solve_diag = _solve_min_norm(system, rhs, rank_tol=rank_tol, eps=eps)
            best_solution = solution[:n_points]
            weight_sum = float(best_solution.sum().item())
            if math.isfinite(weight_sum) and abs(weight_sum) > float(eps):
                best_solution = best_solution / weight_sum
            best_diag = {
                "anchor_count": int(anchor_count),
                "fallback": False,
                "residual": float(solve_diag["residual"]),
                "full_row_rank": True,
            }
            break

        if anchor_count == 1:
            solution, solve_diag = _solve_min_norm(system, rhs, rank_tol=rank_tol, eps=eps)
            best_solution = solution[:n_points]
            weight_sum = float(best_solution.sum().item())
            if math.isfinite(weight_sum) and abs(weight_sum) > float(eps):
                best_solution = best_solution / weight_sum
            best_diag = {
                "anchor_count": int(anchor_count),
                "fallback": True,
                "residual": float(solve_diag["residual"]),
                "full_row_rank": False,
            }

    if best_solution is None or not torch.isfinite(best_solution).all():
        best_solution = _normalized_uniform_weights(n_points, device=coords.device, dtype=coords.dtype)
        best_diag = {"anchor_count": 0, "fallback": True, "residual": float("inf"), "full_row_rank": False}

    if not return_diagnostics:
        return best_solution
    return best_solution, best_diag
