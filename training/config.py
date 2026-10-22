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
    "DistillConfig",
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
    manifest_dir: str = str(ROOT / "data" / "manifests")
    # manifest mode: {manifest group: share of each epoch}, e.g. {"SOCOFing": 0.5, "FVC2000": 0.25}
    manifest_train: dict[str, float] | None = None
    cache_dir: str = str(ROOT / "runs" / "cache")
    socofing: bool = True
    socofing_altered: bool = True
    socofing_drop_variants: tuple[str, ...] = ("CR",)  # alteration types excluded (manifest)
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
    anisotropy: float = 0.05  # log-uniform ± axis-scale difference
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
    aux: str = "none"  # in-batch pair loss on P×K batches: "none" | "supcon" | "ms"
    aux_weight: float = 0.5
    aux_temperature: float = 0.1


@dataclass(frozen=True)
class OptimConfig:
    """Optimiser and schedule."""

    epochs: int = 30
    samples_per_epoch: int = 40000
    batch_size: int = 96  # requested P*K batch; the VRAM probe lowers it if > 80 % peak
    skip_probe: bool = False  # trust batch_size without probing (tests / known hardware)
    k: int = 4  # images per identity in a P*K batch
    effective_batch: int = 0  # > batch_size -> gradient accumulation; 0 = no accumulation
    lr: float = 5e-4  # reference LR at lr_ref_batch, rescaled to the effective batch
    lr_ref_batch: int = 128
    lr_scaling: str = "sqrt"  # "sqrt" | "linear" | "none"
    head_lr_mult: float = 10.0
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema: float = 0.9995  # EMA with warm-up (timm ModelEmaV3 use_warmup)
    grad_clip: float = 5.0
    divergence_factor: float = 2.0  # abort when the windowed loss exceeds this x its best
    max_nonfinite: int = 20
    early_stop_patience: int = 3  # evaluations without pooled-val improvement ...
    early_stop_after: int = 40  # ... counted only after this epoch (0 = never stop early)

    def scaled_lr(self, effective_batch: int) -> float:
        """Reference LR rescaled from ``lr_ref_batch`` to ``effective_batch``."""

        ratio = effective_batch / self.lr_ref_batch
        if self.lr_scaling == "linear":
            return self.lr * ratio
        if self.lr_scaling == "sqrt":
            return self.lr * ratio ** 0.5
        return self.lr


@dataclass(frozen=True)
class DistillConfig:
    """Optional knowledge distillation from a trained teacher checkpoint."""

    teacher: str | None = None  # path to teacher best.pt (same embedding dim)
    w_arc: float = 1.0  # identity margin loss
    w_cos: float = 4.0  # 1 - cos(student, teacher)
    w_rel: float = 4.0  # MSE between in-batch similarity matrices


@dataclass(frozen=True)
class TrainConfig:
    """Top-level experiment config."""

    name: str = "baseline"
    seed: int = 0
    out_dir: str = str(ROOT / "runs" / "embedder")
    input_size: int = 256
    eval_every: int = 4
    compile: str = "default"  # torch.compile mode or "none"
    log_every: int = 50
    num_threads: int = 8
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)

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
              "loss": LossConfig, "optim": OptimConfig, "distill": DistillConfig}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in nested:
            kwargs[key] = _build(nested[key], value or {})
        else:
            kwargs[key] = value
    return TrainConfig(**kwargs)
