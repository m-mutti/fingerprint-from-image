"""Registry and checksummed download cache for pretrained model files.

Runtime code asks for a model by name; the file is resolved from an explicit
path or the cache directory (``$FINGERPRINT_FROM_IMAGE_MODELS`` or the user
cache), and is downloaded on first use only when the registry entry has a URL.
Every file is verified against its SHA-256 before use.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ENV_VAR",
    "MODEL_REGISTRY",
    "ModelSpec",
    "ModelUnavailableError",
    "cache_dir",
    "model_path",
    "register",
]

ENV_VAR = "FINGERPRINT_FROM_IMAGE_MODELS"


class ModelUnavailableError(RuntimeError):
    """Raised when a model file is missing, unverifiable, or fails its checksum."""


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable model artifact."""

    name: str
    sha256: str
    url: str | None = None
    license: str = "unspecified"
    description: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    MODEL_REGISTRY[spec.name] = spec
    return spec


def cache_dir() -> Path:
    override = os.environ.get(ENV_VAR)
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "fingerprint-from-image"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_path(name: str, *, explicit: str | Path | None = None, download: bool = True) -> Path:
    """Return a verified local path for registered model ``name``."""

    spec = MODEL_REGISTRY.get(name)
    if spec is None:
        raise ModelUnavailableError(f"Unknown model {name!r}")
    candidates = [Path(explicit)] if explicit is not None else []
    candidates.append(cache_dir() / spec.name)
    for candidate in candidates:
        if candidate.is_file():
            if _sha256(candidate) != spec.sha256:
                raise ModelUnavailableError(f"Checksum mismatch for {candidate}")
            return candidate
    if not (download and spec.url):
        raise ModelUnavailableError(
            f"Model {name!r} is not available locally; place it in {cache_dir()} "
            f"or set {ENV_VAR}"
        )
    destination = cache_dir() / spec.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(spec.url, timeout=60) as response:  # noqa: S310
                shutil.copyfileobj(response, temporary)
        except OSError as exc:
            temporary.close()
            temporary_path.unlink(missing_ok=True)
            raise ModelUnavailableError(f"Could not download {spec.url}: {exc}") from exc
    if _sha256(temporary_path) != spec.sha256:
        temporary_path.unlink(missing_ok=True)
        raise ModelUnavailableError(f"Downloaded {name!r} failed its checksum")
    temporary_path.replace(destination)
    return destination
