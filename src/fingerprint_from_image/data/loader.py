"""Framework-free manifest loader with identity labels and P x K batch sampling.

Example::

    from fingerprint_from_image.data import ManifestDataset
    train = ManifestDataset.from_files("data/manifests/all.jsonl", splits={"train"},
                                       exclude_datasets={"mendeley_family"})
    image, label = train[0]            # (H, W) uint8, int identity label
    for batch in train.pk_batches(p=32, k=4, seed=0):
        ...                            # list of row indices, 32 identities x 4 images
"""

from __future__ import annotations

import glob
import random
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .quality import to_gray_uint8
from .schema import read_manifest

__all__ = ["ManifestDataset", "load_gray"]


def load_gray(path: str | Path) -> np.ndarray:
    """Decode an image file to ``(H, W) uint8`` (16-bit and colour handled)."""

    with Image.open(path) as image:
        image.load()
        return to_gray_uint8(image)


class ManifestDataset:
    """Rows of one or more manifests after filtering, with contiguous int labels.

    Excluded rows (duplicates) are dropped unless ``include_excluded``; FVC2004
    (``heldout``) rows are only returned when ``heldout`` is in ``splits``.
    """

    def __init__(self, rows: Iterable[Mapping[str, Any]], *,
                 splits: Collection[str] | None = None,
                 datasets: Collection[str] | None = None,
                 exclude_datasets: Collection[str] = (),
                 modalities: Collection[str] | None = None,
                 include_variants: bool = True,
                 include_excluded: bool = False,
                 min_impressions: int = 1,
                 where: Callable[[Mapping[str, Any]], bool] | None = None) -> None:
        wanted_splits = set(splits) if splits is not None else {"train", "val", "test"}
        kept = []
        for row in rows:
            if row["split"] not in wanted_splits:
                continue
            if datasets is not None and row["dataset"] not in datasets:
                continue
            if row["dataset"] in exclude_datasets:
                continue
            if modalities is not None and row["modality"] not in modalities:
                continue
            if row.get("exclude") and not include_excluded:
                continue
            if row.get("variant") and not include_variants:
                continue
            if where is not None and not where(row):
                continue
            kept.append(dict(row))
        counts: dict[str, int] = defaultdict(int)
        for row in kept:
            counts[row["identity"]] += 1
        self.rows = [r for r in kept if counts[r["identity"]] >= min_impressions]
        self.identities = sorted({r["identity"] for r in self.rows})
        self.label_of = {identity: i for i, identity in enumerate(self.identities)}
        self.labels = np.array([self.label_of[r["identity"]] for r in self.rows], dtype=np.int64)
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels.tolist()):
            self.by_label[label].append(index)

    @classmethod
    def from_files(cls, *patterns: str | Path, **filters: Any) -> ManifestDataset:
        """Load every JSONL matching the glob ``patterns`` (deduplicated by path)."""

        paths = sorted({p for pattern in patterns for p in glob.glob(str(pattern))})
        if not paths:
            raise FileNotFoundError(f"no manifest matches {patterns}")
        seen: set[str] = set()
        rows = []
        for manifest in paths:
            for row in read_manifest(manifest):
                if row["path"] not in seen:
                    seen.add(row["path"])
                    rows.append(row)
        return cls(rows, **filters)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return load_gray(self.rows[index]["path"]), int(self.labels[index])

    @property
    def num_identities(self) -> int:
        return len(self.identities)

    def pk_batches(self, p: int, k: int, *, seed: int = 0) -> Iterator[list[int]]:
        """Yield batches of ``p`` identities x ``k`` row indices for one epoch.

        Identities with fewer than ``k`` images are sampled with replacement;
        identities with a single image are skipped (no positive pair).
        """

        rng = random.Random(seed)
        order = [label for label, idx in self.by_label.items() if len(idx) >= 2]
        rng.shuffle(order)
        for start in range(0, len(order) - p + 1, p):
            batch: list[int] = []
            for label in order[start:start + p]:
                members = self.by_label[label]
                if len(members) >= k:
                    batch.extend(rng.sample(members, k))
                else:
                    batch.extend(rng.choice(members) for _ in range(k))
            yield batch
