"""Learned (v3) fingerprint recognition: embedding store and recognizer.

This subpackage depends only on ``numpy`` and ``Pillow``; the embedding and
hand-detection models it consumes run on ``onnxruntime``.
"""

from __future__ import annotations

from .recognizer import (
    Embedder,
    FingerDetector,
    FingerprintRecognizer,
    IdentityCandidate,
    RecognizerConfig,
    best_assignment,
)
from .store import EMBEDDING_SCHEMA_VERSION, EmbeddingStore, GalleryMatrix

__all__ = [
    "EMBEDDING_SCHEMA_VERSION",
    "Embedder",
    "EmbeddingStore",
    "FingerDetector",
    "FingerprintRecognizer",
    "GalleryMatrix",
    "IdentityCandidate",
    "RecognizerConfig",
    "best_assignment",
]
