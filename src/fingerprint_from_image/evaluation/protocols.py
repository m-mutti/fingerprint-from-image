"""Comparison protocols: FVC genuine/impostor pairs and subject-disjoint splits.

Sample indices are flat: ``index = finger * n_impressions + impression`` with
zero-based ``finger`` and ``impression`` (FVC file ``101_1.tif`` is index 0).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np

__all__ = [
    "Pairs",
    "SubjectSplit",
    "check_subject_overlap",
    "fvc_pairs",
    "socofing_subject_split",
    "split_subjects",
]

ImpostorProtocol = Literal["fvc", "all"]


@dataclass(frozen=True)
class Pairs:
    """Unordered comparison pairs as ``(M, 2)`` int64 arrays of flat sample indices.

    Each row ``(i, j)`` has ``i < j``; ``i`` is the enrolled/reference sample and
    ``j`` the probe, matching the FVC convention of never running the symmetric
    comparison.
    """

    genuine: np.ndarray = field(repr=False)
    impostor: np.ndarray = field(repr=False)
    n_fingers: int
    n_impressions: int
    impostor_protocol: ImpostorProtocol

    @property
    def n_samples(self) -> int:
        return self.n_fingers * self.n_impressions

    def all_pairs(self) -> np.ndarray:
        """Genuine then impostor rows, stacked."""

        return np.concatenate([self.genuine, self.impostor], axis=0)


def _as_pairs(rows: Iterable[tuple[int, int]]) -> np.ndarray:
    array = np.asarray(list(rows), dtype=np.int64)
    return array.reshape(-1, 2)


def fvc_pairs(
    n_fingers: int, n_impressions: int, impostor: ImpostorProtocol = "fvc"
) -> Pairs:
    """FVC2000/2002/2004 verification pairs.

    Genuine: every unordered pair of impressions of the same finger
    (``n_fingers * C(n_impressions, 2)``; 280 for a DB_B set, 2,800 for DB_A).
    Impostor ``"fvc"``: the first impression of each finger against the first
    impression of every other finger (``C(n_fingers, 2)``; 45 for DB_B, 4,950
    for DB_A).  Impostor ``"all"``: every cross-finger unordered pair, which
    gives the FMR resolution needed for FMR1000 on the small DB_B sets.
    """

    if n_fingers < 1 or n_impressions < 1:
        raise ValueError("n_fingers and n_impressions must be positive")
    if impostor not in ("fvc", "all"):
        raise ValueError("impostor must be 'fvc' or 'all'")
    m = n_impressions
    genuine = _as_pairs(
        (f * m + a, f * m + b) for f in range(n_fingers) for a, b in combinations(range(m), 2)
    )
    if impostor == "fvc":
        impostor_rows = _as_pairs((f * m, g * m) for f, g in combinations(range(n_fingers), 2))
    else:
        impostor_rows = _as_pairs(
            (f * m + a, g * m + b)
            for f, g in combinations(range(n_fingers), 2)
            for a in range(m)
            for b in range(m)
        )
    return Pairs(genuine, impostor_rows, n_fingers, n_impressions, impostor)


@dataclass(frozen=True)
class SubjectSplit:
    """Subject identifiers per split plus the seed and fractions that produced them."""

    splits: dict[str, tuple[str, ...]]
    seed: int
    fractions: dict[str, float]

    def split_of(self, subject: str) -> str:
        for name, members in self.splits.items():
            if subject in members:
                return name
        raise KeyError(subject)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "fractions": self.fractions,
            "counts": {name: len(members) for name, members in self.splits.items()},
            "subjects": {name: list(members) for name, members in self.splits.items()},
        }


def _subject_sort_key(subject: str) -> tuple[int, int | str]:
    return (0, int(subject)) if subject.isdigit() else (1, subject)


def split_subjects(
    subjects: Iterable[str],
    fractions: Sequence[tuple[str, float]] = (("train", 0.8), ("val", 0.1), ("test", 0.1)),
    *,
    seed: int = 0,
) -> SubjectSplit:
    """Deterministically partition subjects (not samples) into named splits.

    Subjects are sorted, shuffled with ``random.Random(seed)``, and cut by the
    cumulative fractions (rounded), so the result is reproducible across
    platforms and Python versions.
    """

    ordered = sorted(set(subjects), key=_subject_sort_key)
    if not ordered:
        raise ValueError("no subjects to split")
    total = sum(weight for _name, weight in fractions)
    if total <= 0 or any(weight < 0 for _name, weight in fractions):
        raise ValueError("fractions must be non-negative with a positive sum")
    random.Random(seed).shuffle(ordered)
    splits: dict[str, tuple[str, ...]] = {}
    start = 0
    cumulative = 0.0
    for position, (name, weight) in enumerate(fractions):
        cumulative += weight
        stop = len(ordered) if position == len(fractions) - 1 else round(
            len(ordered) * cumulative / total
        )
        splits[name] = tuple(sorted(ordered[start:stop], key=_subject_sort_key))
        start = stop
    return SubjectSplit(
        splits=splits,
        seed=seed,
        fractions={name: weight / total for name, weight in fractions},
    )


def check_subject_overlap(groups: Mapping[str, Iterable[str]]) -> dict[str, int]:
    """Count shared subjects for every pair of groups (``"a&b" -> count``)."""

    sets = {name: set(members) for name, members in groups.items()}
    return {
        f"{a}&{b}": len(sets[a] & sets[b]) for a, b in combinations(sorted(sets), 2)
    }


def socofing_subject_split(
    root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    seed: int = 0,
    fractions: Sequence[tuple[str, float]] = (("train", 0.8), ("val", 0.1), ("test", 0.1)),
    include_files: bool = True,
) -> dict[str, Any]:
    """Build a subject-disjoint SOCOFing split and optionally write it as JSON.

    The overlap of the shipped ``train/val/test`` folders is measured and
    recorded under ``"shipped_overlap"``.  When the shipped folders share
    subjects (the HF mirror does: every subject appears in ``train``), all
    images are pooled and re-split by subject id with :func:`split_subjects`.
    With ``include_files`` the manifest lists every image (path relative to
    ``root``) with its split and parsed attributes.
    """

    from .datasets import load_socofing  # local import keeps module import light

    records = load_socofing(root)
    if not records:
        raise ValueError(f"no SOCOFing images found under {root}")
    shipped: dict[str, set[str]] = {}
    for record in records:
        shipped.setdefault(record.attributes["shipped_split"], set()).add(record.subject)
    overlap = check_subject_overlap(shipped)
    disjoint = not any(overlap.values())
    if disjoint:
        split = SubjectSplit(
            splits={
                name: tuple(sorted(members, key=_subject_sort_key))
                for name, members in shipped.items()
            },
            seed=seed,
            fractions={},
        )
    else:
        split = split_subjects((r.subject for r in records), fractions, seed=seed)
    manifest: dict[str, Any] = {
        "dataset": "SOCOFing",
        "unit": "subject",
        "shipped_overlap": overlap,
        "resplit": not disjoint,
        **split.to_dict(),
        "image_counts": {},
    }
    root_path = Path(root)
    counts: dict[str, dict[str, int]] = {}
    files: list[dict[str, Any]] = []
    for record in records:
        name = split.split_of(record.subject)
        kind = record.attributes["alteration"] or "Real"
        counts.setdefault(name, {}).setdefault(kind, 0)
        counts[name][kind] += 1
        if include_files:
            files.append(
                {
                    "path": record.path.relative_to(root_path).as_posix(),
                    "split": name,
                    "subject": record.subject,
                    "finger_id": record.finger_id,
                    **{k: v for k, v in record.attributes.items() if k != "shipped_split"},
                }
            )
    manifest["image_counts"] = counts
    if include_files:
        manifest["files"] = files
    if manifest_path is not None:
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest
