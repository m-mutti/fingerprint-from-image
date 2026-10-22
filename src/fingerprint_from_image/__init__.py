"""Fingerprint extraction, enrollment, and matching from images.

The package exposes a small public API: :class:`FingerprintMatcher` provides
enrollment, identification, and verification backed by a versioned SQLite
template store, while :func:`create_template` and :func:`match_templates` work
without a database. Hand photographs are segmented by
:func:`extract_fingerprints`.
"""

from __future__ import annotations

from .config import APPLICATION_ID, FEATURE_VERSION, SCHEMA_VERSION, MatcherConfig
from .errors import (
    FingerprintDatabaseError,
    FingerprintMatchingError,
    FingerprintQualityError,
)
from .extraction import (
    EXTRACTION_VERSION,
    ExtractionConfig,
    FingerprintCrop,
    FingerprintExtractionError,
    extract_fingerprint_images,
    extract_fingerprints,
    save_fingerprints,
)
from .features import create_template
from .matcher import FingerprintMatcher
from .pairwise import match_templates
from .store import FingerprintStore
from .templates import (
    EnrollmentReport,
    FingerAssignment,
    FingerprintTemplate,
    IdentityMatch,
    MatchResult,
)

__version__ = "2.0.0.dev0"

__all__ = [
    "APPLICATION_ID",
    "EXTRACTION_VERSION",
    "FEATURE_VERSION",
    "SCHEMA_VERSION",
    "EnrollmentReport",
    "ExtractionConfig",
    "FingerAssignment",
    "FingerprintCrop",
    "FingerprintDatabaseError",
    "FingerprintExtractionError",
    "FingerprintMatcher",
    "FingerprintMatchingError",
    "FingerprintQualityError",
    "FingerprintStore",
    "FingerprintTemplate",
    "IdentityMatch",
    "MatchResult",
    "MatcherConfig",
    "__version__",
    "create_template",
    "extract_fingerprint_images",
    "extract_fingerprints",
    "match_templates",
    "save_fingerprints",
]
