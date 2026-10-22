"""Identity losses: ArcFace (± sub-centres), CosFace, AdaFace; SupCon / MS pair losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from training.config import LossConfig

__all__ = ["MarginLoss", "aux_pair_loss", "multi_similarity_loss", "supcon_loss"]


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


def supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., arXiv 2004.11362) over one batch.

    Anchors without an in-batch positive are ignored (P×K batches always have K-1).
    """

    z = F.normalize(embeddings.float(), dim=1)
    logits = z @ z.T / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    positives = (labels[:, None] == labels[None, :]) & ~eye
    logits = logits.masked_fill(eye, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = positives.sum(dim=1)
    valid = count > 0
    if not valid.any():
        return z.sum() * 0.0
    mean_pos = (log_prob.masked_fill(~positives, 0.0).sum(dim=1)[valid] / count[valid])
    return -mean_pos.mean()


def multi_similarity_loss(embeddings: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0,
                          beta: float = 50.0, base: float = 0.5,
                          margin: float = 0.1) -> torch.Tensor:
    """Multi-Similarity loss with its pair mining (Wang et al., arXiv 1904.06627)."""

    z = F.normalize(embeddings.float(), dim=1)
    sim = z @ z.T
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos_mask = (labels[:, None] == labels[None, :]) & ~eye
    neg_mask = labels[:, None] != labels[None, :]
    inf = torch.tensor(float("inf"), device=z.device)
    hardest_pos = torch.where(pos_mask, sim, inf).amin(dim=1, keepdim=True)
    hardest_neg = torch.where(neg_mask, sim, -inf).amax(dim=1, keepdim=True)
    keep_neg = neg_mask & (sim + margin > hardest_pos)
    keep_pos = pos_mask & (sim - margin < hardest_neg)
    pos_term = torch.log1p((torch.exp(-alpha * (sim - base)) * keep_pos).sum(dim=1)) / alpha
    neg_term = torch.log1p((torch.exp(beta * (sim - base)) * keep_neg).sum(dim=1)) / beta
    return (pos_term + neg_term).mean()


def aux_pair_loss(kind: str, embeddings: torch.Tensor, labels: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    """Dispatch the auxiliary in-batch pair loss (``supcon`` | ``ms``)."""

    if kind == "supcon":
        return supcon_loss(embeddings, labels, temperature)
    if kind == "ms":
        return multi_similarity_loss(embeddings, labels)
    raise ValueError(f"unknown auxiliary loss {kind!r}")
