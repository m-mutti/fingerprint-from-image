"""Enrollment, identification, and verification over learned embeddings.

The recognizer is model-agnostic: it accepts any embedder with ``model_version``,
``dim`` and ``embed(images) -> (N, dim)`` unit-norm rows, and (for hand
photographs) any detector returning :class:`~fingerprint_from_image.types.FingerCrop`
lists. Scores are cosine similarities mapped to ``[0, 1]`` as ``(1 + cos) / 2``.

Identity decisions follow the same evidence rules as the classic matcher: query
fingertips are assigned one-to-one to enrolled finger labels, below-threshold
assignments contribute nothing, the identity score is averaged over every query
slot, a quorum of matched fingers is required, and near-ties are rejected as
ambiguous.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from ..errors import FingerprintMatchingError
from ..io import load_image, to_gray
from ..types import FingerCrop
from .store import EmbeddingStore

__all__ = [
    "Embedder",
    "FingerDetector",
    "FingerprintRecognizer",
    "IdentityCandidate",
    "RecognizerConfig",
]

ImageInput = str | Path | np.ndarray
Kind = Literal["cropped", "hand"]


class Embedder(Protocol):
    model_version: str
    dim: int

    def embed(self, images: Sequence[np.ndarray]) -> np.ndarray: ...


class FingerDetector(Protocol):
    def __call__(self, rgb: np.ndarray) -> list[FingerCrop]: ...


@dataclass(frozen=True)
class RecognizerConfig:
    """Decision thresholds on the ``[0, 1]`` similarity scale.

    Defaults are placeholders until calibrated on development data for a
    specific model (see ``benchmarks/``); never tune them on the test set.
    """

    finger_threshold: float = 0.75
    identity_threshold: float = 0.75
    min_matched_finger_fraction: float = 0.5
    ambiguity_margin: float = 0.02
    min_crop_quality: float = 0.0

    def __post_init__(self) -> None:
        for name in ("finger_threshold", "identity_threshold", "min_matched_finger_fraction"):
            if not 0.0 < float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be in [0, 1]")
        if not 0.0 <= self.min_crop_quality <= 1.0:
            raise ValueError("min_crop_quality must be in [0, 1]")


@dataclass(frozen=True)
class IdentityCandidate:
    """Aggregated decision for one enrolled identity."""

    identity: str
    score: float
    accepted: bool
    matched_fingers: int
    query_fingers: int
    assignments: tuple[tuple[int, str, float], ...] = ()
    ambiguous: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "score": round(self.score, 6),
            "accepted": self.accepted,
            "ambiguous": self.ambiguous,
            "matched_fingers": self.matched_fingers,
            "query_fingers": self.query_fingers,
            "assignments": [
                {"query_index": q, "finger": finger, "score": round(score, 6)}
                for q, finger, score in self.assignments
            ],
        }


def _digest(image: ImageInput, suffix: str = "") -> str:
    digest = hashlib.sha256()
    if isinstance(image, (str, Path)):
        digest.update(Path(image).read_bytes())
    else:
        array = np.ascontiguousarray(image)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    digest.update(suffix.encode("utf-8"))
    return digest.hexdigest()


def best_assignment(scores: np.ndarray) -> list[tuple[int, int]]:
    """Maximise total score with each row and column used at most once.

    Exact bitmask dynamic programming for up to 16 columns (ten fingers plus
    labels such as ``unknown``); greedy beyond that.
    """

    rows, columns = scores.shape
    if rows == 0 or columns == 0:
        return []
    if columns > 16:
        order = np.dstack(np.unravel_index(np.argsort(-scores, axis=None), scores.shape))[0]
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        pairs = []
        for row, column in order:
            if scores[row, column] > 0 and row not in used_rows and column not in used_columns:
                used_rows.add(int(row))
                used_columns.add(int(column))
                pairs.append((int(row), int(column)))
        return sorted(pairs)

    @cache
    def solve(row: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if row == rows:
            return 0.0, ()
        best_score, best_pairs = solve(row + 1, used)
        for column in range(columns):
            if used & (1 << column):
                continue
            tail_score, tail_pairs = solve(row + 1, used | (1 << column))
            candidate = float(scores[row, column]) + tail_score
            if candidate > best_score + 1e-12:
                best_score, best_pairs = candidate, ((row, column), *tail_pairs)
        return best_score, best_pairs

    return list(solve(0, 0)[1])


class FingerprintRecognizer:
    """Learned-embedding recognizer backed by an :class:`EmbeddingStore`."""

    def __init__(
        self,
        database: str | Path | EmbeddingStore,
        embedder: Embedder,
        *,
        detector: FingerDetector | None = None,
        config: RecognizerConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.detector = detector
        self.config = config or RecognizerConfig()
        self._owns_store = not isinstance(database, EmbeddingStore)
        self.store = (
            database
            if isinstance(database, EmbeddingStore)
            else EmbeddingStore(database, model_version=embedder.model_version, dim=embedder.dim)
        )

    def __enter__(self) -> FingerprintRecognizer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def crops(self, image: ImageInput, *, kind: Kind = "cropped", finger: str = "unknown"
              ) -> list[FingerCrop]:
        """Turn an input image into fingertip crops ready for embedding."""

        source = str(image) if isinstance(image, (str, Path)) else None
        if kind == "cropped":
            gray = load_image(image) if isinstance(image, (str, Path)) else to_gray(image)
            return [FingerCrop(gray, label=finger, source=source)]
        if kind != "hand":
            raise ValueError("kind must be 'cropped' or 'hand'")
        if self.detector is None:
            raise FingerprintMatchingError("Hand images require a finger detector")
        rgb = load_image(image, mode="RGB") if isinstance(image, (str, Path)) else image
        found = [c for c in self.detector(rgb) if c.quality >= self.config.min_crop_quality]
        if not found:
            raise FingerprintMatchingError("No usable fingertips were detected")
        return found

    def _embed(self, crops: Sequence[FingerCrop]) -> np.ndarray:
        vectors = np.asarray(self.embedder.embed([crop.image for crop in crops]), np.float32)
        if vectors.shape != (len(crops), self.embedder.dim):
            raise FingerprintMatchingError(f"Embedder returned shape {vectors.shape}")
        return vectors

    def enroll(self, identity: str, image: ImageInput, *, kind: Kind = "cropped",
               finger: str = "unknown") -> int:
        """Enroll every crop from one capture atomically; return how many were new."""

        crops = self.crops(image, kind=kind, finger=finger)
        vectors = self._embed(crops)
        base = _digest(image)
        added = 0
        for index, (crop, vector) in enumerate(zip(crops, vectors, strict=True)):
            suffix = f":{kind}:{index}:{crop.label}" if kind == "hand" else ""
            stored = self.store.add(
                identity,
                crop.label,
                vector,
                quality=crop.quality,
                source_digest=base if not suffix else hashlib.sha256(
                    (base + suffix).encode()
                ).hexdigest(),
                source_path=crop.source,
            )
            added += stored is not None
        return added

    def _candidates(self, query: np.ndarray, only: str | None = None) -> list[IdentityCandidate]:
        gallery = self.store.gallery()
        if len(gallery) == 0:
            raise FingerprintMatchingError("The fingerprint database is empty")
        similarity = (1.0 + np.clip(query @ gallery.vectors.T, -1.0, 1.0)) / 2.0
        identities = np.asarray(gallery.identities, dtype=object)
        fingers = np.asarray(gallery.fingers, dtype=object)
        names = [only] if only is not None else sorted(set(gallery.identities), key=str.casefold)
        slots = query.shape[0]
        required = max(1, math.ceil(slots * self.config.min_matched_finger_fraction))
        results = []
        for name in names:
            mask = identities == name
            labels = sorted(set(fingers[mask]), key=str.casefold)
            per_label = np.stack(
                [similarity[:, mask & (fingers == label)].max(axis=1) for label in labels], axis=1
            )
            pairs = best_assignment(per_label)
            assignments = tuple(
                (q, labels[c], float(per_label[q, c])) for q, c in pairs
            )
            strong = [a for a in assignments if a[2] >= self.config.finger_threshold]
            score = float(sum(a[2] for a in strong) / slots)
            results.append(
                IdentityCandidate(
                    identity=name,
                    score=score,
                    accepted=score >= self.config.identity_threshold and len(strong) >= required,
                    matched_fingers=len(strong),
                    query_fingers=slots,
                    assignments=assignments,
                )
            )
        return results

    def identify(self, image: ImageInput, *, kind: Kind = "cropped", top_k: int = 5
                 ) -> list[IdentityCandidate]:
        """Rank enrolled identities for a query; reject near-tied leaders."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        ranked = self._candidates(self._embed(self.crops(image, kind=kind)))
        ranked.sort(key=lambda item: (-item.score, item.identity.casefold()))
        margin = self.config.ambiguity_margin
        if len(ranked) > 1 and ranked[0].accepted and ranked[0].score - ranked[1].score < margin:
            best = ranked[0].score
            ranked = [
                replace(item, accepted=False, ambiguous=True)
                if best - item.score < margin
                else item
                for item in ranked
            ]
        return ranked[:top_k]

    def verify(self, identity: str, image: ImageInput, *, kind: Kind = "cropped"
               ) -> IdentityCandidate:
        """Decide whether a query belongs to the claimed identity."""

        if not self.store.has_identity(identity):
            raise FingerprintMatchingError(f"Identity is not enrolled: {identity}")
        return self._candidates(self._embed(self.crops(image, kind=kind)), only=identity)[0]
