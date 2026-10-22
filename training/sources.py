"""Training/validation sample gathering and per-source sampling weights.

Two modes:

* **manifest** (``data.manifest_train`` set): train on the shared manifests' ``train``
  split of the listed groups (``exclude: true`` rows dropped), each group drawn with
  the given share of every epoch;
* **legacy** (first experiments): SOCOFing via the seed-0 split + local duplicate
  merging, plus ``data.fvc_train_sets`` (FVC DB_B folders).

Validation always uses the manifest ``val`` identities (:mod:`training.validate`).
FVC2004 (``split == "heldout"``) is never loaded here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from training.config import TrainConfig
from training.data import Sample, fvc_samples, socofing_samples, split_from_file, split_socofing
from training.dedup import dedup_socofing, duplicate_groups
from training.manifest import group_of, load_rows, rows_to_samples, val_groups
from training.synth import synthetic_samples

__all__ = ["gather", "sampling_weights"]

logger = logging.getLogger(__name__)
_VAL_MANIFESTS = ("FVC2000", "FVC2002", "neurotech", "l3sf", "SOCOFing")


def _legacy(cfg: TrainConfig) -> tuple[list[Sample], dict]:
    root = Path(cfg.data.data_root)
    train: list[Sample] = []
    stats: dict = {}
    if cfg.data.socofing:
        soco = socofing_samples(root, altered=cfg.data.socofing_altered)
        if cfg.data.socofing_split_file:
            parts = split_from_file(soco, Path(cfg.data.socofing_split_file))
        else:
            soco_train, soco_val = split_socofing(soco, cfg.data.socofing_val_subjects,
                                                  cfg.data.split_seed)
            parts = {"train": soco_train, "val": soco_val, "test": []}
        if cfg.data.dedup_threshold is not None:
            groups = duplicate_groups(soco, Path(cfg.data.cache_dir), cfg.data.dedup_threshold)
            parts, stats = dedup_socofing(parts, groups)
        train += parts["train"]
    train += fvc_samples(root, cfg.data.fvc_train_sets)
    if cfg.data.fvc_train_sets:
        logger.warning("legacy FVC training sets overlap the manifest FVC2000/2002 val fingers")
    return train, stats


def gather(cfg: TrainConfig) -> tuple[list[Sample], dict[str, list[Sample]], dict]:
    """(train samples, val samples per set, stats)."""

    manifest_dir = Path(cfg.data.manifest_dir)
    if cfg.data.manifest_train:
        rows = load_rows(manifest_dir, cfg.data.manifest_train)
        train = rows_to_samples(rows, "train")
        stats: dict = {"manifest_groups": dict(cfg.data.manifest_train)}
    else:
        train, stats = _legacy(cfg)
    if cfg.data.synthetic_dir:
        train += synthetic_samples(Path(cfg.data.synthetic_dir))
    val = val_groups(load_rows(manifest_dir, _VAL_MANIFESTS))
    train_ids = {s.label for s in train}
    leaked = sum(1 for samples in val.values() for s in samples if s.label in train_ids)
    stats["val_images_in_train_identities"] = leaked
    return train, val, stats


def _group(sample: Sample) -> str:
    if sample.source == "synthetic":
        return "synthetic"
    if sample.source.lower() == "socofing":
        return "SOCOFing"
    return group_of(sample.source)


def sampling_weights(samples: list[Sample], cfg: TrainConfig) -> torch.Tensor:
    """Per-image weights giving each source group its configured share of an epoch."""

    groups = np.array([_group(s) for s in samples])
    if cfg.data.manifest_train:
        shares = dict(cfg.data.manifest_train)
    else:  # legacy: FVC DB_B folders share "fvc_weight", SOCOFing the rest
        groups = np.array(["fvc" if g.startswith("FVC") else g for g in groups])
        rest = 1.0 - cfg.data.fvc_weight - cfg.data.synthetic_weight
        shares = {"fvc": cfg.data.fvc_weight, "SOCOFing": rest}
    shares["synthetic"] = cfg.data.synthetic_weight
    present = sorted(set(groups.tolist()))
    total = sum(shares.get(g, 0.0) for g in present)
    weights = np.zeros(len(samples))
    for g in present:
        weights[groups == g] = shares.get(g, 0.0) / total / (groups == g).sum()
    return torch.from_numpy(weights)
