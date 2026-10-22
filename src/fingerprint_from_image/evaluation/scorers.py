"""Scorer plugins for the benchmark runner.

A scorer turns images into templates (``None`` marks a failure to enrol) and
scores unordered template pairs.  Scores are similarities; failures are
returned as ``-inf`` so the metrics treat them as the worst possible score.

* :class:`ClassicScorer` wraps the v2 RootSIFT ``create_template`` /
  ``match_templates`` pair.  It is the only part of the evaluation package
  that (indirectly, and lazily) needs OpenCV.
* :class:`EmbeddingScorer` wraps any object exposing
  ``embed(list[np.ndarray]) -> (N, D)`` unit-norm array and scores by cosine.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "ClassicScorer",
    "Embedder",
    "EmbeddingScorer",
    "Scorer",
    "load_embedder",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class Scorer(Protocol):
    """Plugin interface used by :mod:`fingerprint_from_image.evaluation.runner`."""

    name: str

    def enroll(self, images: Sequence[np.ndarray]) -> list[Any | None]:
        """Return one template per image, ``None`` for a failure to enrol."""
        ...

    def compare_pairs(self, templates: Sequence[Any | None], pairs: np.ndarray) -> np.ndarray:
        """Score ``(M, 2)`` index pairs; pairs touching a ``None`` template get ``-inf``."""
        ...

    def describe(self) -> dict[str, Any]:
        """JSON-ready description (versions, configuration) stored with results."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """Anything producing fixed-length embeddings from grayscale images."""

    def embed(self, images: list[np.ndarray]) -> np.ndarray: ...


class ClassicScorer:
    """v2 RootSIFT + affine-RANSAC matcher (requires the OpenCV extra)."""

    name = "classic"

    def __init__(self, config: Any | None = None) -> None:
        from ..config import MatcherConfig

        self.config = config or MatcherConfig()
        self.failures: list[str] = []

    def enroll(self, images: Sequence[np.ndarray]) -> list[Any | None]:
        from ..errors import FingerprintMatchingError
        from ..features import create_template

        templates: list[Any | None] = []
        for index, image in enumerate(images):
            try:
                templates.append(create_template(image, self.config))
            except (FingerprintMatchingError, ValueError) as exc:
                self.failures.append(f"enroll[{index}]: {exc}")
                templates.append(None)
        return templates

    def compare_pairs(self, templates: Sequence[Any | None], pairs: np.ndarray) -> np.ndarray:
        from ..errors import FingerprintMatchingError
        from ..pairwise import match_templates

        scores = np.full(len(pairs), -np.inf)
        for row, (i, j) in enumerate(np.asarray(pairs, dtype=np.int64)):
            reference, probe = templates[i], templates[j]
            if reference is None or probe is None:
                continue
            try:
                scores[row] = match_templates(reference, probe, self.config).score
            except (FingerprintMatchingError, ValueError) as exc:
                self.failures.append(f"compare[{i},{j}]: {exc}")
        return scores

    def describe(self) -> dict[str, Any]:
        from dataclasses import asdict

        import cv2

        from ..config import FEATURE_VERSION

        return {
            "name": self.name,
            "feature_version": FEATURE_VERSION,
            "opencv": cv2.__version__,
            "config": asdict(self.config),
        }


class EmbeddingScorer:
    """Cosine scorer over L2-normalised embeddings from an :class:`Embedder`.

    Rows are re-normalised defensively; the largest deviation of the raw norms
    from 1 is reported by :meth:`describe` so contract violations stay visible.
    Images whose embedding fails or is non-finite count as failures to enrol.
    """

    def __init__(self, embedder: Embedder, *, name: str = "embedding", batch_size: int = 32):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.embedder = embedder
        self.name = name
        self.batch_size = batch_size
        self.max_norm_deviation = 0.0
        self.dim: int | None = None
        self.failures: list[str] = []

    def _embed(self, images: list[np.ndarray], offset: int) -> list[np.ndarray | None]:
        try:
            batch = np.asarray(self.embedder.embed(images), dtype=np.float64)
            if batch.ndim != 2 or batch.shape[0] != len(images):
                raise ValueError(f"embed() returned shape {batch.shape} for {len(images)} images")
            rows: list[np.ndarray | None] = list(batch)
        except Exception as exc:  # noqa: BLE001 - plugin code may raise anything
            if len(images) == 1:
                self.failures.append(f"enroll[{offset}]: {exc}")
                return [None]
            logger.warning("batch at %d failed (%s); retrying image by image", offset, exc)
            rows = []
            for index, image in enumerate(images):
                rows.extend(self._embed([image], offset + index))
            return rows
        return rows

    def enroll(self, images: Sequence[np.ndarray]) -> list[Any | None]:
        templates: list[Any | None] = []
        for start in range(0, len(images), self.batch_size):
            chunk = list(images[start : start + self.batch_size])
            for index, row in enumerate(self._embed(chunk, start)):
                if row is None or not np.isfinite(row).all():
                    if row is not None:
                        self.failures.append(f"enroll[{start + index}]: non-finite embedding")
                    templates.append(None)
                    continue
                norm = float(np.linalg.norm(row))
                if norm <= 1e-12:
                    self.failures.append(f"enroll[{start + index}]: zero embedding")
                    templates.append(None)
                    continue
                self.max_norm_deviation = max(self.max_norm_deviation, abs(norm - 1.0))
                self.dim = row.size
                templates.append((row / norm).astype(np.float32))
        return templates

    def compare_pairs(self, templates: Sequence[Any | None], pairs: np.ndarray) -> np.ndarray:
        index = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        valid = np.asarray([t is not None for t in templates], dtype=bool)
        dim = self.dim or 1
        matrix = np.zeros((len(templates), dim), dtype=np.float32)
        for row, template in enumerate(templates):
            if template is not None:
                matrix[row] = template
        scores = np.einsum("ij,ij->i", matrix[index[:, 0]], matrix[index[:, 1]]).astype(np.float64)
        scores[~(valid[index[:, 0]] & valid[index[:, 1]])] = -np.inf
        return scores

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": self.name,
            "embedder": f"{type(self.embedder).__module__}.{type(self.embedder).__qualname__}",
            "dim": self.dim,
            "batch_size": self.batch_size,
            "max_norm_deviation": self.max_norm_deviation,
        }
        for attribute in ("model_version", "dim"):
            value = getattr(self.embedder, attribute, None)
            if isinstance(value, (str, int)):
                info[f"embedder_{attribute}"] = value
        return info


def load_embedder(spec: str, kwargs_json: str | None = None) -> Embedder:
    """Instantiate ``"package.module:Factory"`` with optional JSON keyword arguments."""

    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError("embedder must look like 'package.module:Factory'")
    factory = getattr(importlib.import_module(module_name), attribute)
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    if not isinstance(kwargs, dict):
        raise ValueError("embedder kwargs must be a JSON object")
    embedder = factory(**kwargs)
    if not callable(getattr(embedder, "embed", None)):
        raise TypeError(f"{spec} did not produce an object with an embed() method")
    return embedder
