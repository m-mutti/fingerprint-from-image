"""Seeding, environment logging, checkpoint I/O and LR schedule helpers."""

from __future__ import annotations

import logging
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.config import TrainConfig, config_from_dict
from training.model import EmbeddingNet, NormalizedEmbedder

__all__ = [
    "GpuMonitor",
    "cosine_lr",
    "environment",
    "load_model",
    "save_checkpoint",
    "set_seed",
    "setup_logging",
]

logger = logging.getLogger(__name__)


class GpuMonitor:
    """Instantaneous GPU utilisation via NVML (returns None if NVML is unavailable)."""

    def __init__(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError, OSError, RuntimeError):
            self._nvml = None

    def utilisation(self) -> int | None:
        if self._nvml is None:
            return None
        return int(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CUDA included)."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def environment() -> dict[str, Any]:
    """Versions and hardware for reproducibility logs."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    import timm

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timm": timm.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": commit,
    }


def setup_logging(run_dir: Path) -> None:
    """Log to stdout and ``run_dir/train.log``."""

    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(run_dir / "train.log", encoding="utf-8")],
        force=True,
    )


def cosine_lr(step: int, total: int, warmup: int) -> float:
    """Linear warm-up then cosine decay to 1 % of the peak."""

    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: TrainConfig,
                    extra: dict[str, Any]) -> None:
    """Save embedding weights (no classifier) with the config needed to rebuild them."""

    tmp = Path(str(path) + ".tmp")
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), **extra}, tmp)
    tmp.replace(path)  # atomic: a crash never leaves a truncated checkpoint


def load_model(path: Path, device: str = "cpu") -> tuple[NormalizedEmbedder, TrainConfig]:
    """Rebuild the normalised embedder from a checkpoint (no network download)."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    raw = dict(state["config"])
    raw["model"] = {**raw["model"], "pretrained": False}
    cfg = config_from_dict(raw)
    net = EmbeddingNet(cfg.model, cfg.input_size)
    net.load_state_dict(state["model"])
    model = NormalizedEmbedder(net).to(device).eval()
    return model, cfg
