"""Learned fixed-length fingerprint embeddings (numpy + Pillow + onnxruntime, no OpenCV).

:class:`FingerprintEmbedder` maps fingerprint images to L2-normalised vectors;
identification is a matrix product of embeddings. ``onnxruntime`` is imported
lazily, so :mod:`.preprocess` (the module; ``preprocess.preprocess`` is the
function) is usable without it.
"""

from __future__ import annotations

from . import preprocess
from .embedder import FingerprintEmbedder
from .preprocess import INPUT_SIZE, TARGET_PERIOD, estimate_ridge_period
from .registry import DEFAULT_MODEL, EMBEDDER_MODELS

__all__ = [
    "DEFAULT_MODEL",
    "EMBEDDER_MODELS",
    "INPUT_SIZE",
    "TARGET_PERIOD",
    "FingerprintEmbedder",
    "estimate_ridge_period",
    "preprocess",
]
