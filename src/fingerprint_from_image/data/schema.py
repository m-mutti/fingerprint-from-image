"""Manifest schema: one JSON object per image, one JSONL file per dataset.

A manifest row is a plain ``dict`` (JSON-serialisable) with the fields in
:data:`REQUIRED_FIELDS` plus the optional fields in :data:`OPTIONAL_FIELDS`.
:func:`validate_record` checks types and vocabularies; :func:`read_manifest`
and :func:`write_manifest` round-trip JSONL files.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = [
    "FINGER_LABELS",
    "MODALITIES",
    "OPTIONAL_FIELDS",
    "REQUIRED_FIELDS",
    "SPLITS",
    "ManifestError",
    "canonical_finger",
    "make_identity",
    "read_manifest",
    "validate_record",
    "write_manifest",
]

SPLITS: tuple[str, ...] = ("train", "val", "test", "heldout")

MODALITIES: tuple[str, ...] = (
    "contact_optical",
    "contact_capacitive",
    "contact_thermal",
    "contact_unknown",
    "rolled",
    "latent",
    "contactless",
    "synthetic",
)

_HANDS = ("left", "right")
_DIGITS = ("thumb", "index", "middle", "ring", "little")
FINGER_LABELS: tuple[str, ...] = tuple(f"{h}_{d}" for h in _HANDS for d in _DIGITS) + (
    "unknown",
)

# Datasets whose finger numbering is not linked to a hand/digit use a local index.
_LOCAL_FINGER = re.compile(r"^finger_\d+$")

# ISO/IEC 19794-2 / ANSI-NIST finger positions 1..10 (right thumb .. left little).
_FINGER_POSITION = {
    1: "right_thumb",
    2: "right_index",
    3: "right_middle",
    4: "right_ring",
    5: "right_little",
    6: "left_thumb",
    7: "left_index",
    8: "left_middle",
    9: "left_ring",
    10: "left_little",
}

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "path": str,
    "dataset": str,
    "subject": str,
    "finger": str,
    "identity": str,
    "impression": int,
    "sensor": str,
    "modality": str,
    "dpi": (int, float),
    "dpi_source": str,
    "width": int,
    "height": int,
    "split": str,
    "license": str,
    "sha256": str,
}

OPTIONAL_FIELDS: dict[str, type | tuple[type, ...]] = {
    "mode": str,  # Pillow mode of the file as stored, e.g. "L", "RGB", "I;16"
    "synthetic": bool,  # rendered rather than captured from a real finger
    "variant": str,  # "" for originals; e.g. "altered:CR:Easy", "postproc:HT1"
    "duplicate_group": (str, type(None)),  # shared by exact / near-duplicate prints
    "exclude": bool,  # True: do not use (duplicate of a kept copy, corrupt, ...)
    "exclude_reason": (str, type(None)),
    "attributes": dict,  # dataset-specific extras (angle, capture, level, ...)
}


class ManifestError(ValueError):
    """A manifest row violates the schema."""


def canonical_finger(hand: str | None = None, digit: str | None = None, *,
                     position: int | None = None) -> str:
    """Return a :data:`FINGER_LABELS` entry from hand + digit or an ISO position 1..10."""

    if position is not None:
        return _FINGER_POSITION.get(int(position), "unknown")
    if hand is None or digit is None:
        return "unknown"
    label = f"{hand.strip().lower()}_{digit.strip().lower()}"
    label = label.replace("pinky", "little").replace("pointer", "index")
    return label if label in FINGER_LABELS else "unknown"


def make_identity(dataset: str, subject: str, finger: str) -> str:
    """Globally unique finger id ``<dataset>:<subject>:<finger>``."""

    return f"{dataset}:{subject}:{finger}"


def validate_record(record: dict[str, Any]) -> None:
    """Raise :class:`ManifestError` if ``record`` violates the schema."""

    for name, kind in REQUIRED_FIELDS.items():
        if name not in record:
            raise ManifestError(f"missing field {name!r}")
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, kind):
            raise ManifestError(f"field {name!r} has type {type(value).__name__}")
    for name, kind in OPTIONAL_FIELDS.items():
        if name in record and not isinstance(record[name], kind):
            raise ManifestError(f"field {name!r} has type {type(record[name]).__name__}")
    unknown = set(record) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise ManifestError(f"unknown field(s) {sorted(unknown)}")
    if record["split"] not in SPLITS:
        raise ManifestError(f"split {record['split']!r} not in {SPLITS}")
    if record["modality"] not in MODALITIES:
        raise ManifestError(f"modality {record['modality']!r} not in {MODALITIES}")
    if record["finger"] not in FINGER_LABELS and not _LOCAL_FINGER.match(record["finger"]):
        raise ManifestError(f"finger {record['finger']!r} not in FINGER_LABELS / finger_<n>")
    expected = make_identity(record["dataset"], record["subject"], record["finger"])
    if record["identity"] != expected:
        raise ManifestError(f"identity {record['identity']!r} != {expected!r}")
    if record["width"] <= 0 or record["height"] <= 0 or record["dpi"] <= 0:
        raise ManifestError("width, height and dpi must be positive")
    if len(record["sha256"]) != 64:
        raise ManifestError("sha256 must be a 64-character hex digest")


def write_manifest(records: Iterable[dict[str, Any]], path: str | Path, *,
                   validate: bool = True) -> int:
    """Write ``records`` as JSONL (atomically); return the number of rows."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            if validate:
                validate_record(record)
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(target)
    return count


def read_manifest(path: str | Path, *, validate: bool = False) -> Iterator[dict[str, Any]]:
    """Yield the rows of a JSONL manifest."""

    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if validate:
                try:
                    validate_record(record)
                except ManifestError as exc:
                    raise ManifestError(f"{path}:{number}: {exc}") from exc
            yield record
