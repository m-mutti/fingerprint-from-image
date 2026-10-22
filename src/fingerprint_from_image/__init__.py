"""Fingerprint extraction, enrollment, and matching from images.

Two pipelines share this package:

* **v3 (learned)** — :mod:`fingerprint_from_image.recognition`, built on
  ``numpy``, ``Pillow`` and ``onnxruntime`` with no OpenCV dependency.
* **classic (v2)** — HSV hand segmentation, RootSIFT templates, and affine
  RANSAC matching (:class:`FingerprintMatcher`). Its names are loaded lazily so
  importing the learned pipeline never imports OpenCV.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "2.0.0.dev0"

_CLASSIC_EXPORTS: dict[str, str] = {
    "APPLICATION_ID": "config",
    "FEATURE_VERSION": "config",
    "SCHEMA_VERSION": "config",
    "MatcherConfig": "config",
    "FingerprintDatabaseError": "errors",
    "FingerprintMatchingError": "errors",
    "FingerprintQualityError": "errors",
    "EXTRACTION_VERSION": "extraction",
    "ExtractionConfig": "extraction",
    "FingerprintCrop": "extraction",
    "FingerprintExtractionError": "extraction",
    "extract_fingerprint_images": "extraction",
    "extract_fingerprints": "extraction",
    "save_fingerprints": "extraction",
    "create_template": "features",
    "FingerprintMatcher": "matcher",
    "match_templates": "pairwise",
    "FingerprintStore": "store",
    "EnrollmentReport": "templates",
    "FingerAssignment": "templates",
    "FingerprintTemplate": "templates",
    "IdentityMatch": "templates",
    "MatchResult": "templates",
}

__all__ = sorted([*_CLASSIC_EXPORTS, "__version__"])


def __getattr__(name: str) -> Any:
    module_name = _CLASSIC_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_CLASSIC_EXPORTS])
