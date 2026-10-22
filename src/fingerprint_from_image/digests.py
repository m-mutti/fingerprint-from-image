"""Content digests and report-safe source labels used by the store."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import FEATURE_VERSION
from .errors import FingerprintMatchingError
from .extraction import EXTRACTION_VERSION, ExtractionConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _extraction_config_signature(config: ExtractionConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _feature_digest(points: bytes, descriptors: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(FEATURE_VERSION.encode("ascii"))
    digest.update(points)
    digest.update(descriptors)
    return digest.hexdigest()


def _source_digest(source: str | Path | np.ndarray, suffix: str = "") -> str:
    digest = hashlib.sha256()
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise FingerprintMatchingError(f"Could not hash source image {path}: {exc}") from exc
    else:
        array = np.ascontiguousarray(np.asarray(source))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    digest.update(suffix.encode("utf-8"))
    return digest.hexdigest()


def _source_description(source: str | Path | np.ndarray) -> str:
    """Return a report-safe source label without serializing array pixels."""

    if isinstance(source, (str, Path)):
        return str(source)
    array = np.asarray(source)
    return f"<ndarray shape={array.shape} dtype={array.dtype}>"


def _hand_crop_digest(
    source_digest: str, crop_index: int, config: ExtractionConfig
) -> str:
    digest = hashlib.sha256()
    digest.update(source_digest.encode("ascii"))
    digest.update(EXTRACTION_VERSION.encode("ascii"))
    digest.update(_extraction_config_signature(config).encode("utf-8"))
    digest.update(f":crop:{crop_index}".encode("ascii"))
    return digest.hexdigest()
