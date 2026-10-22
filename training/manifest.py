"""Shared data manifests (``data/manifests/*.jsonl``) → training samples and val/test sets.

Each row has ``path``, ``dataset`` (e.g. ``FVC2002_DB1``), ``identity``, ``subject``,
``impression``, ``split`` (train / val / test / heldout) and ``exclude``. Rows with
``exclude: true`` are dropped (duplicate resolution by the data agent); rows with
``split == "heldout"`` (FVC2004) are only ever read by the test evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from training.data import Sample
from training.guards import check_selection_rows

__all__ = [
    "SELECTION_SETS",
    "VAL_SETS",
    "all_pair_scores",
    "group_of",
    "load_rows",
    "rows_to_samples",
    "val_groups",
]

# Per-sensor validation sets reported every evaluation; selection = mean over SELECTION_SETS.
VAL_SETS: tuple[str, ...] = (
    "FVC2000_DB1", "FVC2000_DB2", "FVC2000_DB3", "FVC2000_DB4",
    "FVC2002_DB1", "FVC2002_DB2", "FVC2002_DB3", "FVC2002_DB4",
    "neurotech", "l3sf", "SOCOFing",
)
SELECTION_SETS: tuple[str, ...] = (
    "FVC2000_DB1", "FVC2000_DB2", "FVC2000_DB3",
    "FVC2002_DB1", "FVC2002_DB2", "FVC2002_DB3", "neurotech",
)


def group_of(dataset: str) -> str:
    """Manifest file / sampling group of a dataset name (``FVC2002_DB3`` → ``FVC2002``)."""

    return dataset.split("_DB")[0]


def load_rows(manifest_dir: Path, groups: Iterable[str]) -> list[dict]:
    """Non-excluded rows of the given manifest files."""

    rows: list[dict] = []
    for group in groups:
        with (manifest_dir / f"{group}.jsonl").open(encoding="utf-8") as stream:
            rows += [r for r in map(json.loads, stream) if not r.get("exclude")]
    return rows


def rows_to_samples(rows: Iterable[dict], split: str) -> list[Sample]:
    """Rows of one split as :class:`Sample` (label = manifest identity)."""

    if split == "heldout":
        raise ValueError("held-out rows are for the final test only")
    return [
        Sample(r["path"], r["identity"], r["dataset"], f"{r['dataset']}:{r['subject']}",
               r.get("sha256"))
        for r in rows if r["split"] == split
    ]


def val_groups(rows: Sequence[dict], sets: Sequence[str] = VAL_SETS,
               socofing_variants: tuple[str, ...] = ("", "Hard")) -> dict[str, list[Sample]]:
    """Validation samples per sensor/dataset.

    SOCOFing val keeps Real images and Altered-Hard variants only (as before), so its
    all-pairs EER stays cheap; every other set uses all of its val images.
    """

    out: dict[str, list[Sample]] = {}
    for name in sets:
        chosen = [r for r in rows if r["split"] == "val" and r["dataset"] == name]
        if name == "SOCOFing":
            chosen = [r for r in chosen
                      if r.get("variant", "") == "" or "Altered-Hard" in r["path"]]
        if chosen:
            check_selection_rows(chosen)  # selection guard: val rows only
            out[name] = rows_to_samples(chosen, "val")
    return out


def all_pair_scores(embeddings: np.ndarray, labels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(genuine, impostor) cosine scores over every unordered pair."""

    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = emb @ emb.T
    lab = np.asarray(labels)
    upper = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    same = lab[:, None] == lab[None, :]
    return scores[upper & same], scores[upper & ~same]
