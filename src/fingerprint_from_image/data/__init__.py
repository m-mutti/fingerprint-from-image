"""Training-data layer: manifest schema, dataset indexers, QA and loaders.

Manifests are JSONL files (one row per image) built by
``python -m fingerprint_from_image.data.build``; see ``docs/DATASETS.md``.
Only numpy + Pillow (and the standard library) are required.
"""

from .loader import ManifestDataset, load_gray
from .schema import (
    FINGER_LABELS,
    MODALITIES,
    SPLITS,
    ManifestError,
    canonical_finger,
    make_identity,
    read_manifest,
    validate_record,
    write_manifest,
)
from .splits import assign_subject_splits, check_splits, load_subject_split_file

__all__ = [
    "FINGER_LABELS",
    "MODALITIES",
    "SPLITS",
    "ManifestDataset",
    "ManifestError",
    "assign_subject_splits",
    "canonical_finger",
    "check_splits",
    "load_gray",
    "load_subject_split_file",
    "make_identity",
    "read_manifest",
    "validate_record",
    "write_manifest",
]
