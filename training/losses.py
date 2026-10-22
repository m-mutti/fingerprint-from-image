"""Margin-based identity losses: ArcFace (± sub-centres), CosFace and AdaFace."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.config import LossConfig

__all__ = ["MarginLoss"]


class MarginLoss(nn.Module):
    """Normalised-softmax classifier with an additive margin on the target class.

    ``margin_factor`` (0..1) linearly warms the margin up at the start of training.
    """

    def __init__(self, cfg: LossConfig, dim: int, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.empty(num_classes * cfg.subcenters, dim))
        nn.init.normal_(self.weight, std=0.01)
        self.register_buffer("norm_mean", torch.tensor(20.0))
        self.register_buffer("norm_std", torch.tensor(100.0))
        self.margin_factor = 1.0

    def cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Cosine similarity to each class (max over sub-centres)."""

        weight = F.normalize(self.weight.float(), dim=1)
        cos = F.linear(F.normalize(embeddings.float(), dim=1), weight)
        if self.cfg.subcenters > 1:
            cos = cos.view(-1, self.num_classes, self.cfg.subcenters).amax(dim=2)
        return cos.clamp(-1 + 1e-7, 1 - 1e-7)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cos = self.cosine(embeddings)
        m = self.cfg.margin * self.margin_factor
        onehot = F.one_hot(labels, self.num_classes).bool()
        if self.cfg.kind == "cosface":
            logits = torch.where(onehot, cos - m, cos)
        elif self.cfg.kind == "arcface":
            theta = torch.acos(cos)
            target = torch.cos(theta + m)
            fallback = cos - math.sin(math.pi - m) * m
            target = torch.where(theta + m < math.pi, target, fallback)
            logits = torch.where(onehot, target, cos)
        elif self.cfg.kind == "adaface":
            logits = self._adaface(embeddings, cos, onehot, m)
        else:
            raise ValueError(f"Unknown loss {self.cfg.kind!r}")
        return F.cross_entropy(logits * self.cfg.scale, labels)

    def _adaface(
        self, embeddings: torch.Tensor, cos: torch.Tensor, onehot: torch.Tensor, m: float
    ) -> torch.Tensor:
        norms = embeddings.float().norm(dim=1).clamp(0.001, 100).detach()
        with torch.no_grad():
            self.norm_mean.mul_(0.99).add_(0.01 * norms.mean())
            self.norm_std.mul_(0.99).add_(0.01 * norms.std())
        scaler = ((norms - self.norm_mean) / (self.norm_std + 1e-3) * self.cfg.adaface_h)
        scaler = scaler.clamp(-1, 1)[:, None]
        theta = torch.acos(cos)
        theta_m = (theta - m * scaler).clamp(1e-3, math.pi - 1e-3)
        target = torch.cos(theta_m) - (m + m * scaler)
        return torch.where(onehot, target, cos)
