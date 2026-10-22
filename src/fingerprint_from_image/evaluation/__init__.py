"""Biometric evaluation: FVC protocols, ISO/IEC 19795-1 style metrics, benchmark runner.

The metric, protocol and dataset modules depend only on numpy (and Pillow for
image decoding).  Only :class:`~.scorers.ClassicScorer` touches OpenCV, lazily,
through the classic v2 matcher.
"""

from __future__ import annotations

from .datasets import (
    FVC_DPI,
    FVC_SETS,
    FingerprintRecord,
    fvc_set_dir,
    load_fvc_set,
    load_images,
    load_socofing,
    parse_socofing_name,
)
from .metrics import (
    DEFAULT_FMR_TARGETS,
    IdentificationMetrics,
    VerificationMetrics,
    det_points,
    identification_metrics,
    operating_points,
    verification_metrics,
)
from .protocols import (
    Pairs,
    SubjectSplit,
    check_subject_overlap,
    fvc_pairs,
    socofing_subject_split,
    split_subjects,
)
from .scorers import ClassicScorer, Embedder, EmbeddingScorer, Scorer, load_embedder

__all__ = [
    "DEFAULT_FMR_TARGETS",
    "FVC_DPI",
    "FVC_SETS",
    "ClassicScorer",
    "Embedder",
    "EmbeddingScorer",
    "FingerprintRecord",
    "IdentificationMetrics",
    "Pairs",
    "Scorer",
    "SubjectSplit",
    "VerificationMetrics",
    "check_subject_overlap",
    "det_points",
    "fvc_pairs",
    "fvc_set_dir",
    "identification_metrics",
    "load_embedder",
    "load_fvc_set",
    "load_images",
    "load_socofing",
    "operating_points",
    "parse_socofing_name",
    "socofing_subject_split",
    "split_subjects",
    "verification_metrics",
]
