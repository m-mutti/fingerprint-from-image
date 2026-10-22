"""Registered fingerprint-embedder model files (checksummed; hosting URL to be added)."""

from __future__ import annotations

from ..models import ModelSpec, register

__all__ = ["DEFAULT_MODEL", "EMBEDDER_MODELS"]

_LICENSE = "CC-BY-NC-SA-4.0 (weights trained on SOCOFing; non-commercial use only)"

EMBEDDER_MODELS: dict[str, ModelSpec] = {}


def _register(name: str, sha256: str, description: str) -> ModelSpec:
    spec = register(ModelSpec(name=name, sha256=sha256, url=None, license=_LICENSE,
                              description=description))
    EMBEDDER_MODELS[name] = spec
    return spec


DEFAULT_MODEL = "fingerprint_embedder_v1_fp32.onnx"
