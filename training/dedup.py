"""SOCOFing duplicate-print detection and leakage-safe relabelling.

SOCOFing files some identical (or near-identical) prints under different
subject IDs / fingers. Real images are compared by normalised cross-correlation
(NCC) of high-passed (ridge-band) 90x96 images; pairs above ``threshold`` are merged with
union-find into one identity. Each duplicate group is then owned by its *most
held-out* split (test > val > train); members filed in other splits are dropped,
so no print appears in two splits and no print carries two class labels.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from training.data import Sample, load_gray

__all__ = ["dedup_socofing", "duplicate_groups"]

logger = logging.getLogger(__name__)
_PRIORITY = {"test": 0, "val": 1, "train": 2}


def _ncc_matrix(paths: Sequence[str]) -> np.ndarray:
    feats = []
    for path in paths:
        gray = Image.fromarray(load_gray(path, "socofing")).resize((90, 96))
        image = np.asarray(gray, dtype=np.float32)
        low = np.asarray(gray.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
        band = image - low  # ridge detail only; silhouette/background removed
        flat = band.ravel() - band.mean()
        feats.append(flat / (np.linalg.norm(flat) + 1e-6))
    matrix = np.stack(feats)
    return matrix @ matrix.T


def duplicate_groups(
    samples: Sequence[Sample], cache_dir: Path, threshold: float = 0.7
) -> list[list[str]]:
    """Groups (≥ 2) of SOCOFing finger labels whose Real prints have NCC > ``threshold``."""

    cache = cache_dir / f"socofing_dup_groups_{threshold:.2f}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    real = sorted({s.label: s.path for s in samples if "/Real/" in s.path.replace("\\", "/")}
                  .items())
    labels = [label for label, _ in real]
    ncc = _ncc_matrix([path for _, path in real])
    parent = list(range(len(labels)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows, cols = np.nonzero(np.triu(ncc, k=1) > threshold)
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        parent[find(i)] = find(j)
    members: dict[int, list[str]] = {}
    for i, label in enumerate(labels):
        members.setdefault(find(i), []).append(label)
    groups = sorted(sorted(g) for g in members.values() if len(g) > 1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(groups, indent=1), encoding="utf-8")
    logger.info("SOCOFing duplicate pairs > %.2f: %d -> %d groups", threshold, len(rows),
                len(groups))
    return groups


def dedup_socofing(
    parts: dict[str, list[Sample]], groups: Sequence[Sequence[str]]
) -> tuple[dict[str, list[Sample]], dict[str, int]]:
    """Relabel duplicate groups to one identity owned by one split; drop the other copies."""

    split_of = {s.label: part for part, items in parts.items() for s in items}
    canonical: dict[str, str] = {}
    drop: set[str] = set()
    for group in groups:
        owner = min((split_of[label] for label in group), key=_PRIORITY.__getitem__)
        keep = [label for label in group if split_of[label] == owner]
        for label in group:
            if split_of[label] != owner:
                drop.add(label)
            else:
                canonical[label] = keep[0]
    out: dict[str, list[Sample]] = {}
    for part, items in parts.items():
        out[part] = [
            Sample(s.path, canonical.get(s.label, s.label), s.source, s.subject)
            for s in items if s.label not in drop
        ]
    before = sum(len({s.label for s in items}) for items in parts.values())
    after = sum(len({s.label for s in items}) for items in out.values())
    stats = {"groups": len(groups), "labels_dropped_cross_split": len(drop),
             "classes_before": before, "classes_after": after,
             "classes_removed": before - after}
    return out, stats
