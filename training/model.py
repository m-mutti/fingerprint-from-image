"""Embedding network: timm backbone + compact embedding head."""

from __future__ import annotations

import timm
import torch
import torch.nn.functional as F
from torch import nn

from training.config import ModelConfig

__all__ = ["EmbeddingNet", "NormalizedEmbedder"]


class _PoolHead(nn.Module):
    """Concat(avg, max) pooling → LayerNorm → Dropout → Linear → BatchNorm1d."""

    def __init__(self, channels: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(2 * channels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * channels, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3))], dim=1)
        return self.bn(self.fc(self.drop(self.norm(pooled))))


class _FlattenHead(nn.Module):
    """BatchNorm2d → Dropout → Flatten → Linear → BatchNorm1d (ArcFace "E" head)."""

    def __init__(self, channels: int, spatial: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.bn2d = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * spatial * spatial, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bn(self.fc(self.drop(self.bn2d(features)).flatten(1)))


class EmbeddingNet(nn.Module):
    """Backbone feature map → un-normalised ``dim``-d embedding."""

    def __init__(self, cfg: ModelConfig, input_size: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=cfg.pretrained, in_chans=1, num_classes=0,
            global_pool="", drop_path_rate=cfg.drop_path,
        )
        with torch.no_grad():
            probe = self.backbone.forward_features(torch.zeros(1, 1, input_size, input_size))
        channels, spatial = probe.shape[1], probe.shape[2]
        if cfg.head == "flatten":
            self.head: nn.Module = _FlattenHead(channels, spatial, cfg.dim, cfg.dropout)
        elif cfg.head == "gem":
            self.head = _PoolHead(channels, cfg.dim, cfg.dropout)
        else:
            raise ValueError(f"Unknown head {cfg.head!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_features(x))


class NormalizedEmbedder(nn.Module):
    """Export wrapper: ``(N, 1, S, S)`` float32 → ``(N, D)`` unit-norm float32."""

    def __init__(self, net: EmbeddingNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x).float(), dim=1)
