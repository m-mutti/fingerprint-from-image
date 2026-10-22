"""Backward-compatible entry point for the original ``fingerprint_matching`` module.

The implementation now lives in the ``fingerprint_from_image`` package. This
shim keeps ``python fingerprint_matching.py ...`` and existing imports working
from a source checkout, with or without installing the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import fingerprint_from_image  # noqa: F401
except ImportError:  # pragma: no cover - exercised only from an uninstalled checkout
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fingerprint_from_image import *  # noqa: E402,F403
from fingerprint_from_image import __all__  # noqa: E402,F401
from fingerprint_from_image.cli.match import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
