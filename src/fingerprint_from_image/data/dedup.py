"""Exact and near-duplicate detection across all manifests, and their resolution.

* Exact duplicates: identical file SHA-256.
* Near duplicates: cosine of :func:`~.quality.near_dup_descriptor` >= threshold
  between images of *different* identities (same-identity impressions are
  expected to look alike and are not flagged).

Connected components (union-find) become ``duplicate_group`` ids.  In each
group the copy in the most protected split is kept (``heldout`` > ``test`` >
``val`` > ``train``; ties by identity then path); other members are marked
``exclude`` so no print can appear in two splits, and FVC2004 copies always
win over training copies.  Derived variants (``variant`` != "") of an
excluded original are excluded too.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import MutableMapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "SPLIT_PRIORITY",
    "find_near_duplicate_pairs",
    "resolve_duplicates",
    "verify_pairs",
]

SPLIT_PRIORITY = {"heldout": 0, "test": 1, "val": 2, "train": 3}


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_near_duplicate_pairs(descriptors: np.ndarray, identities: Sequence[str], *,
                              threshold: float = 0.6, block: int = 1024,
                              max_pairs: int = 200_000) -> list[tuple[int, int, float]]:
    """Return ``(i, j, cosine)`` with ``i < j``, different identities, cosine >= threshold."""

    matrix = np.ascontiguousarray(descriptors, dtype=np.float32)
    codes: dict[str, int] = {}
    ids = np.asarray([codes.setdefault(i, len(codes)) for i in identities])
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, len(matrix), block):
        stop = min(start + block, len(matrix))
        scores = matrix[start:stop] @ matrix[start:].T
        rows, cols = np.nonzero(scores >= threshold)
        keep = (cols > rows) & (ids[start + rows] != ids[start + cols])
        for r, c in zip(rows[keep].tolist(), cols[keep].tolist(), strict=True):
            pairs.append((start + r, start + c, float(scores[r, c])))
        if len(pairs) > max_pairs:
            raise RuntimeError(f"more than {max_pairs} near-duplicate pairs; threshold too low?")
    return pairs


def _fine_task(item: tuple[str, str]) -> float:
    from .loader import load_gray
    from .quality import fine_similarity

    return fine_similarity(load_gray(item[0]), load_gray(item[1]))


def verify_pairs(records: Sequence[MutableMapping[str, Any]],
                 candidates: Sequence[tuple[int, int, float]], *, threshold: float = 0.6,
                 workers: int = 1
                 ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int, float]]]:
    """Score candidates by 160x160 ridge-crop cosine; return (kept, all scored).

    A pair is kept when its fine cosine reaches ``threshold`` or the files are
    byte-identical.
    """

    items = [(records[i]["path"], records[j]["path"]) for i, j, _ in candidates]
    if workers > 1 and len(items) > 64:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            scores = list(pool.map(_fine_task, items, chunksize=32))
    else:
        scores = [_fine_task(item) for item in items]
    scored = [(i, j, s) for (i, j, _), s in zip(candidates, scores, strict=True)]
    return [(i, j, s) for i, j, s in scored
            if s >= threshold or records[i]["sha256"] == records[j]["sha256"]], scored


def _group_id(members: list[str]) -> str:
    return "dup-" + hashlib.sha1("|".join(sorted(members)).encode()).hexdigest()[:12]


def resolve_duplicates(records: Sequence[MutableMapping[str, Any]],
                       near_pairs: Sequence[tuple[int, int, float]] = ()) -> dict[str, int]:
    """Assign ``duplicate_group`` / ``exclude`` in place; return counters.

    ``near_pairs`` index into ``records``.  Rows already excluded (corrupt
    files) keep their reason.
    """

    union = _UnionFind(len(records))
    by_hash: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_hash[row["sha256"]].append(index)
    for members in by_hash.values():
        for other in members[1:]:
            union.union(members[0], other)
    for i, j, _ in near_pairs:
        union.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        groups[union.find(index)].append(index)

    stats = {"groups": 0, "excluded_cross_identity": 0, "excluded_exact_same_identity": 0,
             "excluded_variants": 0}
    for members in groups.values():
        if len(members) == 1:
            continue
        stats["groups"] += 1
        group = _group_id([records[m]["path"] for m in members])
        for m in members:
            records[m]["duplicate_group"] = group
        ranked = sorted(members, key=lambda m: (
            bool(records[m].get("exclude")),
            SPLIT_PRIORITY.get(records[m]["split"], 9),
            records[m]["identity"],
            records[m]["path"],
        ))
        keep = ranked[0]
        kept_identity = records[keep]["identity"]
        seen_hash_in_identity: set[tuple[str, str]] = {(kept_identity, records[keep]["sha256"])}
        for m in ranked[1:]:
            row = records[m]
            if row.get("exclude"):
                continue
            if row["identity"] == kept_identity:
                key = (row["identity"], row["sha256"])
                if key in seen_hash_in_identity:
                    _exclude(row, f"exact duplicate within identity of {records[keep]['path']}")
                    stats["excluded_exact_same_identity"] += 1
                continue
            _exclude(row, f"duplicate of {records[keep]['path']} ({records[keep]['split']})")
            stats["excluded_cross_identity"] += 1
    stats["excluded_variants"] = _exclude_orphaned_variants(records)
    return stats


def _exclude(row: MutableMapping[str, Any], reason: str) -> None:
    row["exclude"] = True
    row["exclude_reason"] = reason


def _exclude_orphaned_variants(records: Sequence[MutableMapping[str, Any]]) -> int:
    """Exclude derived variants of identities whose originals are all excluded."""

    originals: dict[str, list[bool]] = defaultdict(list)
    for row in records:
        if not row.get("variant"):
            originals[row["identity"]].append(bool(row.get("exclude")))
    dead = {identity for identity, flags in originals.items() if flags and all(flags)}
    count = 0
    for row in records:
        if row.get("variant") and row["identity"] in dead and not row.get("exclude"):
            _exclude(row, "variant of an excluded duplicate identity")
            count += 1
    return count
