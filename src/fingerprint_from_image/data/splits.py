"""Subject-disjoint split assignment and leakage checks for manifests."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_FRACTIONS",
    "assign_subject_splits",
    "check_splits",
    "load_subject_split_file",
]

DEFAULT_FRACTIONS: Mapping[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def _subject_key(subject: str) -> tuple[int, str]:
    return (0, f"{int(subject):012d}") if subject.isdigit() else (1, subject)


def assign_subject_splits(
    subjects: Iterable[str],
    *,
    seed: int = 0,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
) -> dict[str, str]:
    """Map each subject to a split with a seeded shuffle (deterministic).

    Subjects are sorted (numerically when possible) before shuffling, so the
    result depends only on the subject set and ``seed``.  Split sizes are the
    rounded fractions; ``train`` (or the first split) absorbs the remainder.
    Every non-zero fraction gets at least one subject when there are enough.
    """

    unique = sorted(set(subjects), key=_subject_key)
    if not unique:
        return {}
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("fractions must sum to a positive number")
    rng = random.Random(seed)
    order = list(unique)
    rng.shuffle(order)
    names = list(fractions)
    sizes = {name: int(round(len(order) * fractions[name] / total)) for name in names}
    for name in names:
        if fractions[name] > 0 and sizes[name] == 0 and len(order) >= len(names):
            sizes[name] = 1
    absorber = "train" if "train" in sizes else names[0]
    sizes[absorber] = len(order) - sum(v for k, v in sizes.items() if k != absorber)
    if sizes[absorber] < 0:
        raise ValueError("too few subjects for the requested fractions")
    mapping: dict[str, str] = {}
    start = 0
    for name in [n for n in names if n != absorber] + [absorber]:
        for subject in order[start:start + sizes[name]]:
            mapping[subject] = name
        start += sizes[name]
    return mapping


def load_subject_split_file(path: str | Path) -> dict[str, str]:
    """Read a ``{"subjects": {split: [ids]}}`` JSON (e.g. the SOCOFing seed-0 split)."""

    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(s): split for split, ids in spec["subjects"].items() for s in ids}


def check_splits(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return human-readable leakage violations (empty list = clean).

    Checks, over rows that are not ``exclude``-d:

    * every identity lives in exactly one split;
    * every ``(dataset, subject)`` lives in exactly one split;
    * every ``duplicate_group`` lives in exactly one split;
    * every image hash (``sha256``) lives in exactly one split;
    * datasets whose name starts with ``FVC2004`` are only in ``heldout``.
    """

    by_key: dict[str, dict[str, set[str]]] = {
        "identity": defaultdict(set),
        "subject": defaultdict(set),
        "duplicate_group": defaultdict(set),
        "sha256": defaultdict(set),
    }
    problems: list[str] = []
    for row in records:
        if row.get("exclude"):
            continue
        split = row["split"]
        by_key["identity"][row["identity"]].add(split)
        by_key["subject"][f"{row['dataset']}:{row['subject']}"].add(split)
        by_key["sha256"][row["sha256"]].add(split)
        group = row.get("duplicate_group")
        if group:
            by_key["duplicate_group"][group].add(split)
        if row["dataset"].upper().startswith("FVC2004") and split != "heldout":
            problems.append(f"FVC2004 row outside heldout: {row['path']} ({split})")
    for kind, mapping in by_key.items():
        for key, splits in sorted(mapping.items()):
            if len(splits) > 1:
                problems.append(f"{kind} {key} spans splits {sorted(splits)}")
    return problems
