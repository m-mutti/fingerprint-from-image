"""Resumable training loop: P×K batches, accumulation, EMA, guards and full-state checkpoints.

``state.pt`` (written atomically after every epoch) holds the model, EMA, classifier,
optimizer, step/epoch counters, best validation score, guard state and every RNG
(python/numpy/torch/CUDA/sampler), so a crashed or pre-empted job resumes exactly.
Guards raise :class:`TrainingAborted` with a reason (``nonfinite`` / ``divergence``).
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import ModelEmaV3

from training.augment import Augmenter
from training.config import TrainConfig
from training.losses import MarginLoss, aux_pair_loss
from training.model import EmbeddingNet
from training.sampler import PKSampler
from training.train_utils import GpuMonitor, cosine_lr, load_model

__all__ = ["TrainData", "Trainer", "TrainingAborted"]

logger = logging.getLogger(__name__)


class TrainingAborted(RuntimeError):
    """A guard stopped training; ``reason`` is recorded in the registry."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


@dataclass
class TrainData:
    """Cached canvases on pinned CPU memory plus labels and source groups."""

    canvases: torch.Tensor  # (N, C, C) uint8
    backgrounds: torch.Tensor  # (N,) float32
    boxes: torch.Tensor  # (N, 4) int64
    labels: torch.Tensor  # (N,) int64
    groups: list[str] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1


def _optimizer(net: EmbeddingNet, criterion: MarginLoss, cfg: TrainConfig,
               lr: float) -> torch.optim.AdamW:
    """Backbone / head(+classifier) groups; norms and biases (ndim <= 1) get no decay."""

    groups: dict[tuple[str, bool], list[torch.nn.Parameter]] = {
        ("backbone", True): [], ("backbone", False): [], ("head", True): [], ("head", False): []}
    named = [("backbone", n, p) for n, p in net.backbone.named_parameters()]
    named += [("head", n, p) for n, p in net.head.named_parameters()]
    named += [("head", n, p) for n, p in criterion.named_parameters()]
    for part, name, param in named:
        decay = param.ndim > 1 and not name.endswith(".bias")
        groups[(part, decay)].append(param)
    wd, head_lr = cfg.optim.weight_decay, lr * cfg.optim.head_lr_mult
    return torch.optim.AdamW([
        {"params": groups[("backbone", True)], "lr": lr, "weight_decay": wd},
        {"params": groups[("backbone", False)], "lr": lr, "weight_decay": 0.0},
        {"params": groups[("head", True)], "lr": head_lr, "weight_decay": wd},
        {"params": groups[("head", False)], "lr": head_lr, "weight_decay": 0.0},
    ], fused=True)


class Trainer:
    """Owns model/EMA/optimizer/sampler for one run directory."""

    def __init__(self, cfg: TrainConfig, run_dir: Path, data: TrainData, batch: int,
                 accum: int, shares: dict[str, float], heartbeat: Path | None = None,
                 device: str | None = None) -> None:
        self.cfg, self.run_dir, self.data = cfg, run_dir, data
        self.batch, self.accum = batch, accum
        self.device = torch.device(device or ("cuda" if torch.cuda.device_count() > 0 else "cpu"))
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            torch.cuda.set_per_process_memory_fraction(0.9)  # never spill into WDDM shared memory
        self.lr = cfg.optim.scaled_lr(batch * accum)
        self.net = EmbeddingNet(cfg.model, cfg.input_size).to(self.device)
        self.net = self.net.to(memory_format=torch.channels_last)
        self.criterion = MarginLoss(cfg.loss, cfg.model.dim, data.num_classes).to(self.device)
        # deep copy BEFORE compiling; warm-up avoids biasing short runs towards init weights
        self.ema = ModelEmaV3(self.net, decay=cfg.optim.ema, use_warmup=True)
        self._bn_buffers = [b for m in self.net.modules()
                            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                            for b in m.buffers()]
        self._bn_buffers += list(self.criterion.buffers())  # AdaFace running norm stats
        if cfg.compile != "none":
            # Backbone only: one Inductor graph over backbone+head gives NaN gradients on
            # torch 2.11 / triton-windows 3.7 / sm_120 (EXPERIMENTS.md).
            self.net.backbone.forward_features = torch.compile(
                self.net.backbone.forward_features, mode=cfg.compile)
        self.optimizer = _optimizer(self.net, self.criterion, cfg, self.lr)
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        self.augment = Augmenter(cfg.aug, cfg.input_size)
        self.sampler = PKSampler(data.labels.numpy(), data.groups, shares,
                                 p=max(1, batch // cfg.optim.k), k=cfg.optim.k, seed=cfg.seed)
        self.steps_per_epoch = max(1, cfg.optim.samples_per_epoch // (batch * accum))
        self.total_steps = self.steps_per_epoch * cfg.optim.epochs
        self.warmup = max(self.steps_per_epoch, int(cfg.optim.warmup_epochs * self.steps_per_epoch))
        self.margin_ramp = cfg.loss.margin_warmup_epochs * self.steps_per_epoch
        self.teacher = None
        if cfg.distill.teacher:
            self.teacher, _ = load_model(Path(cfg.distill.teacher), self.device.type)
        self.step, self.epoch = 0, 0
        self.best: tuple[float, float] = (math.inf, math.inf)  # (pooled FNMR@0.1 %, mean EER)
        self.evals_since_best = 0
        self.nonfinite, self.best_window, self.bad_windows = 0, math.inf, 0
        self.gpu = GpuMonitor()
        self.heartbeat = heartbeat
        self.log_path = run_dir / "metrics.jsonl"

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(), "ema": self.ema.state_dict(),
            "criterion": self.criterion.state_dict(), "optimizer": self.optimizer.state_dict(),
            "step": self.step, "epoch": self.epoch, "best": list(self.best),
            "evals_since_best": self.evals_since_best,
            "guards": [self.nonfinite, self.best_window, self.bad_windows],
            "sampler": self.sampler.state(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.cuda else None,
        }

    def save_state(self) -> None:
        tmp = self.run_dir / "state.pt.tmp"
        torch.save(self.state_dict(), tmp)
        tmp.replace(self.run_dir / "state.pt")

    def load_state(self) -> bool:
        path = self.run_dir / "state.pt"
        if not path.is_file():
            return False
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.net.load_state_dict(state["net"])
        self.ema.load_state_dict(state["ema"])
        self.criterion.load_state_dict(state["criterion"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.epoch = state["step"], state["epoch"]
        self.best = tuple(state["best"])
        self.evals_since_best = state.get("evals_since_best", 0)
        self.nonfinite, self.best_window, self.bad_windows = state["guards"]
        self.sampler.load_state(state["sampler"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if self.cuda and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        self._truncate_log()
        logger.info("resumed from %s at epoch %d step %d", path, self.epoch, self.step)
        return True

    def _truncate_log(self) -> None:
        """Drop metrics rows written after the resumed checkpoint (no duplicates)."""

        if not self.log_path.is_file():
            return
        kept = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            limit = self.epoch if row.get("kind") == "epoch" else self.step
            if (row.get("epoch") if row.get("kind") == "epoch" else row.get("step", 0)) <= limit:
                kept.append(line)
        self.log_path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")

    # ---------------------------------------------------------------- steps
    def _autocast(self) -> torch.autocast:
        """bf16 autocast on CUDA; plain fp32 on CPU (CPU bf16 backward is not portable)."""

        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.cuda)

    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if self.heartbeat is not None:
            self.heartbeat.write_text(str(time.time()), encoding="utf-8")

    def _loss(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        d, dev = self.data, self.device
        index = torch.from_numpy(idx)
        canvas = d.canvases[index]
        canvas = (canvas.pin_memory() if self.cuda else canvas).to(dev, non_blocking=True)[:, None]
        x = self.augment(canvas, d.backgrounds[index].to(dev, non_blocking=True),
                         d.boxes[index].to(dev, non_blocking=True))
        x = x.contiguous(memory_format=torch.channels_last)
        with self._autocast():
            emb = self.net(x)
        labels = d.labels[index].to(dev, non_blocking=True)
        loss = self.criterion(emb.float(), labels)
        if self.cfg.loss.aux != "none":
            loss = loss + self.cfg.loss.aux_weight * aux_pair_loss(
                self.cfg.loss.aux, emb, labels, self.cfg.loss.aux_temperature)
        if self.teacher is not None:
            with torch.no_grad(), self._autocast():
                target = F.normalize(self.teacher(x).float(), dim=1)
            student = F.normalize(emb.float(), dim=1)
            loss = self.cfg.distill.w_arc * loss + self.cfg.distill.w_cos * (
                1 - (student * target).sum(1)).mean() + self.cfg.distill.w_rel * F.mse_loss(
                student @ student.T, target @ target.T)
        return loss, x

    def _check_window(self, mean: float) -> None:
        """Divergence guard, active once warm-up and the margin ramp have finished."""

        if self.step < max(self.warmup, self.margin_ramp) + self.steps_per_epoch:
            return
        self.best_window = min(self.best_window, mean)
        self.bad_windows = self.bad_windows + 1 if mean > (
            self.cfg.optim.divergence_factor * self.best_window) else 0
        if self.bad_windows >= 3:
            raise TrainingAborted("divergence", f"window loss {mean:.3f} vs best "
                                  f"{self.best_window:.3f} for 3 windows at step {self.step}")

    def train_epoch(self) -> dict:
        cfg = self.cfg
        self.net.train()
        params = list(self.net.parameters()) + list(self.criterion.parameters())
        start = window_start = time.time()
        stats = torch.zeros(4, device=self.device)  # sum loss, sum loss^2, sum grad, n
        epoch_losses: list[float] = []
        for _ in range(self.steps_per_epoch):
            factor = cosine_lr(self.step, self.total_steps, self.warmup)
            self.criterion.margin_factor = min(1.0, self.step / max(1.0, self.margin_ramp))
            for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
                group["lr"] = base * factor
            self.optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=self.device)
            snapshot = [b.clone() for b in self._bn_buffers]
            finite = True
            for _ in range(self.accum):
                loss, _x = self._loss(self.sampler.batch())
                if not torch.isfinite(loss):  # check BEFORE backward (no NaN gradients)
                    finite = False
                    total = loss.detach()
                    break
                (loss / self.accum).backward()
                total = total + loss.detach() / self.accum
            grad_norm = (torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip) if finite
                         else torch.tensor(float("nan"), device=self.device))
            if not finite or not torch.isfinite(grad_norm):
                for buffer, saved in zip(self._bn_buffers, snapshot, strict=True):
                    buffer.copy_(saved)  # a bad forward must not poison BN running stats
                self.optimizer.zero_grad(set_to_none=True)
                self.nonfinite += 1
                self._log({"kind": "nonfinite", "step": self.step, "loss": float(total)})
                if self.nonfinite > cfg.optim.max_nonfinite:
                    raise TrainingAborted("nonfinite", f"{self.nonfinite} skipped steps")
                self.step += 1
                continue
            self.optimizer.step()
            self.ema.update(self.net, step=self.step)
            self.step += 1
            stats += torch.stack([total, total * total, grad_norm.float(), torch.ones_like(total)])
            if self.step % cfg.log_every == 0:
                s_loss, s_sq, s_grad, n = stats.tolist()
                mean = s_loss / n
                now = time.time()
                self._log({
                    "kind": "step", "step": self.step, "epoch": self.epoch + 1, "loss": mean,
                    "loss_std": max(0.0, s_sq / n - mean * mean) ** 0.5,
                    "grad_norm": s_grad / n, "lr": self.base_lrs[0] * factor,
                    "margin": cfg.loss.margin * self.criterion.margin_factor,
                    "gpu_util": self.gpu.utilisation(),
                    "img_per_s": round(n * self.batch * self.accum / (now - window_start), 1),
                    "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2)
                    if self.cuda else None,
                })
                epoch_losses.append(mean)
                self._check_window(mean)
                stats.zero_()
                window_start = now
        self.epoch += 1
        seconds = time.time() - start
        return {"kind": "epoch", "epoch": self.epoch, "step": self.step,
                "loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                "sec": round(seconds, 1),
                "img_per_s": round(self.steps_per_epoch * self.batch * self.accum / seconds, 1)}

    @torch.no_grad()
    def embed(self, x: np.ndarray) -> np.ndarray:
        """EMA-model embeddings (unit norm) for validation."""

        model = self.ema.module.eval()
        out = []
        for i in range(0, len(x), 64):
            chunk = torch.from_numpy(x[i : i + 64]).to(self.device)
            with self._autocast():
                emb = model(chunk.contiguous(memory_format=torch.channels_last))
            out.append(F.normalize(emb.float(), dim=1).cpu().numpy())
        return np.concatenate(out)
