"""Backward-compatible entry point for the original ``fingerprint_extraction`` module.

The implementation now lives in ``fingerprint_from_image.extraction``. This
shim keeps ``python fingerprint_extraction.py ...`` and existing imports working
from a source checkout, with or without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import fingerprint_from_image  # noqa: F401
except ImportError:  # pragma: no cover - exercised only from an uninstalled checkout
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fingerprint_from_image.cli.extract import main  # noqa: E402
from fingerprint_from_image.extraction import (  # noqa: E402,F401
    EXTRACTION_VERSION,
    ExtractionConfig,
    FingerprintCrop,
    FingerprintExtractionError,
    extract_fingerprint_images,
    extract_fingerprints,
    plot_fingerprints,
    save_fingerprints,
)

if __name__ == "__main__":
    raise SystemExit(main())
