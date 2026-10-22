"""Training/validation sample gathering, source groups/shares and the held-out guard.

Two modes:

* **manifest** (``data.manifest_train`` set): train on the shared manifests' ``train``
  split of the listed groups (``exclude: true`` rows dropped), each group drawn with
  the given share of every epoch;
* **legacy** (first experiments): SOCOFing via the seed-0 split + local duplicate
  merging, plus ``data.fvc_train_sets`` (FVC DB_B folders).

Validation always uses the manifest ``val`` identities (:mod:`training.validate`).
Every training sample is checked against the held-out set (FVC2004, ``split ==
"heldout"``) before anything is cached or trained (:mod:`training.guards`).
"""

from __future__ import annotations

import logging
from pathlib import Path

from training.config import TrainConfig
from training.data import Sample, fvc_samples, socofing_samples, split_from_file, split_socofing
from training.dedup import dedup_socofing, duplicate_groups
from training.guards import check_train_samples, heldout_from_rows
from training.manifest import drop_variants, group_of, load_rows, rows_to_samples, val_groups
from training.synth import synthetic_samples

__all__ = ["HELDOUT_MANIFESTS", "gather", "source_group", "train_shares"]

logger = logging.getLogger(__name__)
VAL_MANIFESTS = ("FVC2000", "FVC2002", "neurotech", "l3sf", "SOCOFing")
HELDOUT_MANIFESTS = ("FVC2004",)


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


def source_group(sample: Sample) -> str:
    """Sampling group: manifest file name (``FVC2002_DB3`` → ``FVC2002``), or synthetic."""

    if sample.source == "synthetic":
        return "synthetic"
    if sample.source.lower() == "socofing":
        return "SOCOFing"
    return group_of(sample.source)


def train_shares(cfg: TrainConfig) -> dict[str, float]:
    """Per-group share of every epoch (P×K sampler draws identities by these shares)."""

    if cfg.data.manifest_train:
        shares = dict(cfg.data.manifest_train)
    else:  # legacy FVC DB_B folders are named FVC2000 etc. after grouping
        rest = 1.0 - cfg.data.fvc_weight - cfg.data.synthetic_weight
        shares = {"SOCOFing": rest}
        for name in cfg.data.fvc_train_sets:
            shares[group_of(name.removesuffix("_B"))] = cfg.data.fvc_weight
    if cfg.data.synthetic_dir:
        shares["synthetic"] = cfg.data.synthetic_weight
    return shares


def gather(cfg: TrainConfig) -> tuple[list[Sample], dict[str, list[Sample]], dict]:
    """(train samples, val samples per set, stats incl. the held-out guard summary)."""

    manifest_dir = Path(cfg.data.manifest_dir)
    if cfg.data.manifest_train:
        rows = load_rows(manifest_dir, [g for g, share in cfg.data.manifest_train.items()
                                        if share > 0])
        rows = drop_variants(rows, cfg.data.socofing_drop_variants)
        train = rows_to_samples(rows, "train")
        stats: dict = {"manifest_groups": dict(cfg.data.manifest_train)}
    else:
        train, stats = _legacy(cfg)
    if cfg.data.synthetic_dir:
        train += synthetic_samples(Path(cfg.data.synthetic_dir))
    heldout = heldout_from_rows(load_rows(manifest_dir, HELDOUT_MANIFESTS))
    sha_of = {s.path: s.sha256 for s in train if s.sha256}
    stats["heldout_guard"] = check_train_samples(train, heldout, sha_of)
    val = val_groups(load_rows(manifest_dir, VAL_MANIFESTS))
    train_ids = {s.label for s in train}
    stats["val_images_in_train_identities"] = sum(
        1 for samples in val.values() for s in samples if s.label in train_ids)
    if stats["val_images_in_train_identities"]:
        raise ValueError("validation identities appear in the training set")
    return train, val, stats
