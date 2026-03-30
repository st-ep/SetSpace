from __future__ import annotations

import torch
import torch.nn as nn


def _gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if points.dim() != 3:
        raise ValueError(f"points must be (B, N, C), got {points.shape=}")
    if idx.dim() == 2:
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        return points.gather(1, gather_idx)
    if idx.dim() == 3:
        batch_size, n_query, n_neighbors = idx.shape
        flat = idx.reshape(batch_size, -1)
        gathered = _gather_points(points, flat)
        return gathered.reshape(batch_size, n_query, n_neighbors, points.shape[-1])
    raise ValueError(f"idx must be 2D or 3D, got {idx.shape=}")


def _gather_features(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if features.dim() != 3:
        raise ValueError(f"features must be (B, C, N), got {features.shape=}")
    if idx.dim() == 2:
        gather_idx = idx.unsqueeze(1).expand(-1, features.shape[1], -1)
        return features.gather(2, gather_idx)
    if idx.dim() == 3:
        batch_size, n_query, n_neighbors = idx.shape
        flat = idx.reshape(batch_size, -1)
        gathered = _gather_features(features, flat)
        return gathered.reshape(batch_size, features.shape[1], n_query, n_neighbors)
    raise ValueError(f"idx must be 2D or 3D, got {idx.shape=}")


def _farthest_point_sample(coords: torch.Tensor, n_samples: int) -> torch.Tensor:
    if coords.dim() != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must be (B, N, 3), got {coords.shape=}")

    batch_size, n_points, _ = coords.shape
    n_samples = int(min(max(int(n_samples), 1), n_points))
    device = coords.device
    batch_idx = torch.arange(batch_size, device=device)

    centroids = torch.zeros((batch_size, n_samples), dtype=torch.long, device=device)
    min_dist = torch.full((batch_size, n_points), float("inf"), device=device)
    centroid_guess = coords.mean(dim=1, keepdim=True)
    farthest = ((coords - centroid_guess).square().sum(dim=-1)).max(dim=1).indices

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = coords[batch_idx, farthest].unsqueeze(1)
        dists = (coords - centroid).square().sum(dim=-1)
        min_dist = torch.minimum(min_dist, dists)
        farthest = min_dist.max(dim=1).indices

    return centroids


def _knn_indices(query_coords: torch.Tensor, support_coords: torch.Tensor, k: int) -> torch.Tensor:
    if query_coords.dim() != 3 or support_coords.dim() != 3:
        raise ValueError(
            f"query_coords and support_coords must be 3D, got {query_coords.shape=} and {support_coords.shape=}"
        )
    k = int(min(max(int(k), 1), support_coords.shape[1]))
    dists = torch.cdist(query_coords, support_coords)
    return torch.topk(dists, k=k, largest=False).indices


def _ball_query_indices(
    query_coords: torch.Tensor,
    support_coords: torch.Tensor,
    *,
    radius: float | None,
    nsample: int,
) -> torch.Tensor:
    if radius is None:
        return _knn_indices(query_coords, support_coords, nsample)

    dists = torch.cdist(query_coords, support_coords)
    nsample = int(min(max(int(nsample), 1), support_coords.shape[1]))
    knn_idx = torch.topk(dists, k=nsample, largest=False).indices

    masked = dists.masked_fill(dists > float(radius), float("inf"))
    topk = torch.topk(masked, k=nsample, largest=False)
    valid = torch.isfinite(topk.values)
    return torch.where(valid, topk.indices, knn_idx)


def _group_density_weights(
    grouped_coords: torch.Tensor,
    *,
    density_k: int,
    intrinsic_dim: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    if grouped_coords.dim() != 4:
        raise ValueError(f"grouped_coords must be (B, M, K, 3), got {grouped_coords.shape=}")

    batch_size, n_query, n_neighbors, _ = grouped_coords.shape
    if n_neighbors == 1:
        return torch.ones((batch_size, n_query, n_neighbors), device=grouped_coords.device, dtype=grouped_coords.dtype)

    flat = grouped_coords.reshape(batch_size * n_query, n_neighbors, grouped_coords.shape[-1])
    dists = torch.cdist(flat, flat)
    dists.diagonal(dim1=1, dim2=2).fill_(float("inf"))
    neighbor_rank = min(max(int(density_k), 1), n_neighbors - 1)
    kth = torch.topk(dists, k=neighbor_rank, largest=False).values[:, :, -1]
    raw = kth.clamp_min(eps).pow(int(intrinsic_dim))
    weights = raw / raw.sum(dim=1, keepdim=True).clamp_min(eps)
    return weights.reshape(batch_size, n_query, n_neighbors)


def _make_conv1d_block(in_dim: int, out_dim: int, *, activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False), nn.BatchNorm1d(out_dim)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _make_conv2d_block(in_dim: int, out_dim: int, *, activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False), nn.BatchNorm2d(out_dim)]
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _make_linear_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True))


class PointNeXtLocalAggregation(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        radius: float,
        nsample: int,
        weight_mode: str,
        density_k: int,
        intrinsic_dim: int,
        normalize_dp: bool,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.nsample = int(nsample)
        self.weight_mode = weight_mode.lower()
        self.density_k = int(density_k)
        self.intrinsic_dim = int(intrinsic_dim)
        self.normalize_dp = bool(normalize_dp)
        self.message_net = _make_conv2d_block(channels + 3, channels, activation=True)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        idx = _ball_query_indices(coords, coords, radius=self.radius, nsample=self.nsample)
        grouped_coords = _gather_points(coords, idx)
        rel_coords = grouped_coords - coords.unsqueeze(2)
        if self.normalize_dp and self.radius > 0:
            rel_coords = rel_coords / self.radius
        grouped_feats = _gather_features(feats, idx)
        message_input = torch.cat([rel_coords.permute(0, 3, 1, 2), grouped_feats], dim=1)
        messages = self.message_net(message_input)

        if self.weight_mode == "uniform":
            return messages.max(dim=-1).values
        if self.weight_mode == "knn":
            weights = _group_density_weights(
                grouped_coords,
                density_k=self.density_k,
                intrinsic_dim=self.intrinsic_dim,
            )
            return torch.sum(messages * weights.unsqueeze(1), dim=-1)
        raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")


class PointNeXtInvResBlock(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        radius: float,
        nsample: int,
        expansion: int,
        weight_mode: str,
        density_k: int,
        intrinsic_dim: int,
        normalize_dp: bool,
    ) -> None:
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.local = PointNeXtLocalAggregation(
            channels=channels,
            radius=radius,
            nsample=nsample,
            weight_mode=weight_mode,
            density_k=density_k,
            intrinsic_dim=intrinsic_dim,
            normalize_dp=normalize_dp,
        )
        self.pointwise = nn.Sequential(
            _make_conv1d_block(channels, hidden_channels, activation=True),
            _make_conv1d_block(hidden_channels, channels, activation=False),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = self.local(coords, feats)
        x = self.pointwise(x)
        return self.activation(x + feats)


class PointNeXtSetAbstraction(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: int,
        radius: float | None,
        nsample: int,
        layers: int,
        use_res: bool,
        weight_mode: str,
        density_k: int,
        intrinsic_dim: int,
        normalize_dp: bool,
        is_head: bool = False,
        global_pool: bool = False,
    ) -> None:
        super().__init__()
        self.stride = int(stride)
        self.radius = None if radius is None else float(radius)
        self.nsample = int(nsample)
        self.use_res = bool(use_res)
        self.weight_mode = weight_mode.lower()
        self.density_k = int(density_k)
        self.intrinsic_dim = int(intrinsic_dim)
        self.normalize_dp = bool(normalize_dp)
        self.is_head = bool(is_head)
        self.global_pool = bool(global_pool)

        if self.is_head:
            mlp_layers: list[nn.Module] = []
            current_dim = int(in_channels)
            for _ in range(max(int(layers), 1)):
                mlp_layers.append(_make_conv1d_block(current_dim, int(out_channels), activation=True))
                current_dim = int(out_channels)
            self.head_mlp = nn.Sequential(*mlp_layers)
            return

        mid_channels = int(out_channels // 2) if self.stride > 1 else int(out_channels)
        message_layers: list[nn.Module] = []
        current_dim = int(in_channels) + 3
        total_layers = max(int(layers), 1)
        for layer_idx in range(total_layers):
            next_dim = int(out_channels) if layer_idx == total_layers - 1 else mid_channels
            message_layers.append(_make_conv2d_block(current_dim, next_dim, activation=True))
            current_dim = next_dim
        self.message_net = nn.Sequential(*message_layers)
        self.skip = None
        if self.use_res and not self.global_pool:
            self.skip = _make_conv1d_block(int(in_channels), int(out_channels), activation=False)
            self.activation = nn.ReLU(inplace=True)

    def _reduce(self, messages: torch.Tensor, grouped_coords: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        # `uniform` follows PointNeXt's max reduction; `knn` is our research variant.
        if self.weight_mode == "uniform":
            return messages.max(dim=-1).values
        if self.weight_mode == "knn":
            weights = _group_density_weights(
                grouped_coords,
                density_k=self.density_k,
                intrinsic_dim=self.intrinsic_dim,
            )
            return torch.sum(messages * weights.unsqueeze(1), dim=-1)
        raise ValueError(f"Unsupported weight_mode: {self.weight_mode}")

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_head:
            return coords, self.head_mlp(feats)

        if self.global_pool:
            pooled_coords = coords.mean(dim=1, keepdim=True)
            grouped_coords = coords.unsqueeze(1)
            rel_coords = grouped_coords - pooled_coords.unsqueeze(2)
            scale = max(self.radius or 1.0, 1e-8)
            if self.normalize_dp:
                rel_coords = rel_coords / scale
            grouped_feats = feats.unsqueeze(2)
            message_input = torch.cat([rel_coords.permute(0, 3, 1, 2), grouped_feats], dim=1)
            messages = self.message_net(message_input)
            return pooled_coords, self._reduce(messages, grouped_coords, pooled_coords)

        n_points = coords.shape[1]
        n_samples = max(n_points // max(self.stride, 1), 1)
        sample_idx = _farthest_point_sample(coords, n_samples)
        sampled_coords = _gather_points(coords, sample_idx)
        grouped_idx = _ball_query_indices(sampled_coords, coords, radius=self.radius, nsample=self.nsample)
        grouped_coords = _gather_points(coords, grouped_idx)
        rel_coords = grouped_coords - sampled_coords.unsqueeze(2)
        scale = max(self.radius or 1.0, 1e-8)
        if self.normalize_dp:
            rel_coords = rel_coords / scale
        grouped_feats = _gather_features(feats, grouped_idx)
        message_input = torch.cat([rel_coords.permute(0, 3, 1, 2), grouped_feats], dim=1)
        messages = self.message_net(message_input)
        reduced = self._reduce(messages, grouped_coords, sampled_coords)

        if self.skip is None:
            return sampled_coords, reduced

        skip = self.skip(_gather_features(feats, sample_idx))
        return sampled_coords, self.activation(reduced + skip)


class PointNeXtClassifier(nn.Module):
    def __init__(
        self,
        *,
        value_input_dim: int = 1,
        num_classes: int,
        width: int = 32,
        blocks: tuple[int, ...] = (1, 1, 1, 1, 1, 1),
        strides: tuple[int, ...] = (1, 2, 2, 2, 2, 1),
        radius: float = 0.15,
        radius_scaling: float = 1.5,
        nsample: int = 32,
        expansion: int = 4,
        sa_layers: int = 2,
        sa_use_res: bool = True,
        head_hidden_dim: int = 256,
        weight_mode: str = "uniform",
        density_k: int = 8,
        intrinsic_dim: int = 2,
        normalize_dp: bool = True,
    ) -> None:
        super().__init__()
        if weight_mode.lower() not in {"uniform", "knn"}:
            raise ValueError(f"weight_mode must be 'uniform' or 'knn', got {weight_mode}")
        if len(blocks) != len(strides):
            raise ValueError("blocks and strides must have the same length.")
        if len(blocks) < 2:
            raise ValueError("PointNeXtClassifier expects at least a head stage and a global stage.")

        self.weight_mode = weight_mode.lower()
        self.value_input_dim = int(value_input_dim)

        channels: list[int] = []
        current_width = int(width)
        for stride in strides:
            if int(stride) > 1:
                current_width *= 2
            channels.append(current_width)

        # The default `(width=32, blocks=(1,1,1,1,1,1), strides=(1,2,2,2,2,1))`
        # matches the PointNeXt-S classification scaling from the paper/config.
        stages: list[nn.Module] = []
        in_channels = 3 + self.value_input_dim
        current_radius = float(radius)
        current_nsample = int(nsample)
        total_stages = len(blocks)
        for stage_idx, (n_blocks, stride, out_channels) in enumerate(zip(blocks, strides, channels)):
            stage_idx = int(stage_idx)
            stride = int(stride)
            n_blocks = int(n_blocks)
            is_head = stage_idx == 0
            is_global = stage_idx == total_stages - 1
            use_res = bool(sa_use_res) and not is_head and not is_global
            sa_block = PointNeXtSetAbstraction(
                in_channels=in_channels,
                out_channels=int(out_channels),
                stride=stride,
                radius=None if is_global else current_radius,
                nsample=current_nsample,
                layers=1 if (n_blocks > 1 and not is_head) else int(sa_layers),
                use_res=use_res,
                weight_mode=self.weight_mode,
                density_k=int(density_k),
                intrinsic_dim=int(intrinsic_dim),
                normalize_dp=normalize_dp,
                is_head=is_head,
                global_pool=is_global,
            )
            stage_blocks: list[nn.Module] = [sa_block]

            if not is_head and not is_global:
                for _ in range(max(n_blocks - 1, 0)):
                    stage_blocks.append(
                        PointNeXtInvResBlock(
                            channels=int(out_channels),
                            radius=current_radius,
                            nsample=current_nsample,
                            expansion=int(expansion),
                            weight_mode=self.weight_mode,
                            density_k=int(density_k),
                            intrinsic_dim=int(intrinsic_dim),
                            normalize_dp=normalize_dp,
                        )
                    )

            stages.append(nn.ModuleList(stage_blocks))
            in_channels = int(out_channels)
            if not is_head and not is_global and stride > 1:
                current_radius *= float(radius_scaling)

        self.stages = nn.ModuleList(stages)
        final_channels = channels[-1]
        self.head = nn.Sequential(
            _make_linear_block(final_channels, 2 * int(head_hidden_dim)),
            _make_linear_block(2 * int(head_hidden_dim), int(head_hidden_dim)),
            nn.Linear(int(head_hidden_dim), int(num_classes)),
        )

    def embed(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if point_mask is not None:
            raise ValueError("PointNeXtClassifier currently expects fixed-size inputs without point masks.")
        if coords.dim() != 3 or values.dim() != 3:
            raise ValueError(f"coords and values must be 3D, got {coords.shape=} and {values.shape=}")
        if coords.shape[:2] != values.shape[:2]:
            raise ValueError(f"coords and values must share (B, N), got {coords.shape=} and {values.shape=}")

        stage_coords = coords
        stage_feats = torch.cat([coords, values], dim=-1).transpose(1, 2).contiguous()
        for stage in self.stages:
            stage_coords, stage_feats = stage[0](stage_coords, stage_feats)
            for block in stage[1:]:
                stage_feats = block(stage_coords, stage_feats)
        return stage_feats.squeeze(-1)

    def forward(
        self,
        coords: torch.Tensor,
        values: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.head(self.embed(coords, values, point_mask=point_mask))
