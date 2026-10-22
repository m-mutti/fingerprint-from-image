"""Experiment configuration (frozen dataclasses, loadable from YAML)."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AugConfig",
    "DataConfig",
    "LossConfig",
    "ModelConfig",
    "OptimConfig",
    "TrainConfig",
    "config_from_dict",
    "load_config",
]

ROOT = Path(os.environ.get("FFI_ROOT", "E:/Personal_tries/Mutti_Finger_Prints"))


@dataclass(frozen=True)
class DataConfig:
    """Which images to train on and how to split them."""

    data_root: str = str(ROOT / "data")
    cache_dir: str = str(ROOT / "runs" / "embedder" / "cache")
    socofing: bool = True
    socofing_altered: bool = True
    socofing_split_file: str | None = str(
        ROOT / "benchmarks" / "splits" / "socofing_subject_split_seed0.json"
    )
    socofing_val_subjects: int = 60  # used only when socofing_split_file is None
    dedup_threshold: float | None = 0.5  # ridge-band NCC; merges duplicate SOCOFing prints
    split_seed: int = 0
    fvc_train_sets: tuple[str, ...] = (
        "FVC2000_DB1_B", "FVC2000_DB2_B", "FVC2000_DB3_B", "FVC2000_DB4_B",
    )
    fvc_val_sets: tuple[str, ...] = (
        "FVC2002_DB1_B", "FVC2002_DB2_B", "FVC2002_DB3_B", "FVC2002_DB4_B",
    )
    fvc_weight: float = 0.1  # fraction of each epoch drawn from FVC training images
    synthetic_dir: str | None = None  # optional synthetic-print cache (see synth.py)
    synthetic_weight: float = 0.0
    naive_resize: bool = False  # ablation: skip ridge-period normalisation
    canvas: int = 320  # cached canvas side (≥ input size, room for translation)


@dataclass(frozen=True)
class AugConfig:
    """GPU acquisition augmentation strengths (probabilities and ranges)."""

    strength: str = "heavy"  # "light" | "heavy"
    rotation: float = 45.0
    scale: float = 0.3  # log-uniform ± (covers ridge-period estimate spread across impressions)
    translate: float = 0.12  # fraction of input size
    elastic: float = 0.035  # max std of the coarse displacement field (normalised units)
    partial: float = 0.35
    cutout: float = 0.25
    lowres: float = 0.35
    blur: float = 0.3
    sharpen: float = 0.2
    morph: float = 0.35
    noise: float = 0.5
    illumination: float = 0.3
    polarity: float = 0.1
    frame: float = 0.15  # synthetic sensor-frame edges / bars
    min_print: float = 0.35  # occlusion must keep ≥ this fraction of the print
    gamma: float = 0.5


@dataclass(frozen=True)
class ModelConfig:
    """Backbone + embedding head."""

    backbone: str = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    pretrained: bool = True
    head: str = "gem"  # "gem" | "flatten"
    dim: int = 256
    dropout: float = 0.1
    drop_path: float = 0.1


@dataclass(frozen=True)
class LossConfig:
    """Margin-based identity loss."""

    kind: str = "arcface"  # "arcface" | "adaface" | "cosface"
    scale: float = 32.0
    margin: float = 0.4
    subcenters: int = 1
    margin_warmup_epochs: float = 2.0
    adaface_h: float = 0.333


@dataclass(frozen=True)
class OptimConfig:
    """Optimiser and schedule."""

    epochs: int = 30
    samples_per_epoch: int = 40000
    batch_size: int = 128
    lr: float = 5e-4
    head_lr_mult: float = 10.0
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema: float = 0.999
    grad_clip: float = 5.0


@dataclass(frozen=True)
class TrainConfig:
    """Top-level experiment config."""

    name: str = "baseline"
    seed: int = 0
    out_dir: str = str(ROOT / "runs" / "embedder")
    input_size: int = 256
    eval_every: int = 2
    compile: str = "default"  # torch.compile mode or "none"
    log_every: int = 50
    num_threads: int = 8
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view for logging and checkpoints."""

        return dataclasses.asdict(self)


def _build(cls: type, values: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    names = {f.name for f in dataclasses.fields(cls)}
    defaults = cls()
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"Unknown config key {cls.__name__}.{key}")
        default = getattr(defaults, key)
        if isinstance(default, tuple) and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load a YAML config (nested sections) on top of the defaults."""

    raw: dict[str, Any] = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> TrainConfig:
    """Build a :class:`TrainConfig` from nested plain dicts (YAML or checkpoint)."""

    nested = {"data": DataConfig, "aug": AugConfig, "model": ModelConfig,
              "loss": LossConfig, "optim": OptimConfig}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in nested:
            kwargs[key] = _build(nested[key], value or {})
        else:
            kwargs[key] = value
    return TrainConfig(**kwargs)
