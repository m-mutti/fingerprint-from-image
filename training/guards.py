"""Protocol guards enforced in code: held-out isolation and val-only selection.

* **Held-out guard** — the set of held-out (``split == "heldout"``) identities, image
  paths and image SHA-256 values is hashed; training refuses any sample that hits it,
  the hash is stored in every ``train`` record, and evaluation refuses checkpoints whose
  record reports an overlap or a different held-out definition.
* **Selection guard** — selection inputs must come from ``val`` rows only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from training.data import Sample
from training.keys import canonical_json

__all__ = [
    "ContaminationError",
    "HeldOut",
    "check_selection_rows",
    "check_train_samples",
    "heldout_from_rows",
]


class ContaminationError(RuntimeError):
    """Training or selection data overlaps the held-out test set."""


@dataclass(frozen=True)
class HeldOut:
    """Identifiers of the held-out test set."""

    identities: frozenset[str]
    paths: frozenset[str]
    sha256: frozenset[str]

    @property
    def digest(self) -> str:
        payload = canonical_json({"identities": sorted(self.identities),
                                  "sha256": sorted(self.sha256)})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _norm(path: str) -> str:
    return Path(path).as_posix().lower()


def heldout_from_rows(rows: Iterable[dict]) -> HeldOut:
    """Collect held-out identities/paths/hashes from manifest rows (all splits passed in)."""

    held = [r for r in rows if r.get("split") == "heldout"]
    return HeldOut(
        identities=frozenset(r["identity"] for r in held),
        paths=frozenset(_norm(r["path"]) for r in held),
        sha256=frozenset(r["sha256"] for r in held if r.get("sha256")),
    )


def check_train_samples(samples: Sequence[Sample], heldout: HeldOut,
                        sha_of: dict[str, str] | None = None) -> dict:
    """Raise :class:`ContaminationError` if any training sample is held-out data.

    ``sha_of`` maps sample paths to image SHA-256 (from the manifests) when known.
    Returns a summary stored in the train record.
    """

    hits = [s.path for s in samples if s.label in heldout.identities
            or _norm(s.path) in heldout.paths
            or (sha_of is not None and sha_of.get(s.path) in heldout.sha256)]
    if hits:
        raise ContaminationError(f"{len(hits)} training images are held-out data, e.g. {hits[0]}")
    return {"heldout_digest": heldout.digest, "heldout_overlap": 0,
            "heldout_identities": len(heldout.identities)}


def check_selection_rows(rows: Iterable[dict]) -> None:
    """Selection (checkpoint/threshold choice) may only read ``val`` rows."""

    bad = [r["path"] for r in rows if r.get("split") != "val"]
    if bad:
        raise ContaminationError(f"selection data contains non-val rows, e.g. {bad[0]}")
