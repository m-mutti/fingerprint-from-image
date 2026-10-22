"""Image-free template and match-result value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import FEATURE_VERSION


@dataclass(frozen=True)
class FingerprintTemplate:
    """A compact, image-free fingerprint representation.

    ``keypoints`` has shape ``(N, 2)`` and ``descriptors`` has shape
    ``(N, 128)``.  Arrays contain little-endian float32 values so they can be
    persisted without Python-specific serialization.
    """

    keypoints: np.ndarray = field(repr=False, compare=False)
    descriptors: np.ndarray = field(repr=False, compare=False)
    image_shape: tuple[int, int]
    quality: float
    roi_fraction: float
    identity: str | None = None
    finger: str | None = None
    template_id: int | None = None
    source: str | None = None
    feature_version: str = FEATURE_VERSION

    def __post_init__(self) -> None:
        points = np.ascontiguousarray(self.keypoints, dtype="<f4")
        descriptors = np.ascontiguousarray(self.descriptors, dtype="<f4")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("keypoints must have shape (N, 2)")
        if descriptors.ndim != 2 or descriptors.shape[0] != points.shape[0]:
            raise ValueError("descriptors must have shape (N, D)")
        if descriptors.shape[1] != 128:
            raise ValueError("RootSIFT descriptors must have 128 columns")
        if points.shape[0] == 0:
            raise ValueError("A fingerprint template cannot be empty")
        if not np.isfinite(points).all() or not np.isfinite(descriptors).all():
            raise ValueError("Template feature arrays must contain only finite values")
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("image_shape must contain positive height and width")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        if not 0.0 <= self.roi_fraction <= 1.0:
            raise ValueError("roi_fraction must be in [0, 1]")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError(
                f"Unsupported feature version {self.feature_version!r}; "
                f"expected {FEATURE_VERSION!r}"
            )
        object.__setattr__(self, "keypoints", points)
        object.__setattr__(self, "descriptors", descriptors)

    @property
    def keypoint_count(self) -> int:
        return int(self.keypoints.shape[0])

    def to_dict(self, *, include_features: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "template_id": self.template_id,
            "identity": self.identity,
            "finger": self.finger,
            "source": self.source,
            "image_shape": list(self.image_shape),
            "quality": round(float(self.quality), 6),
            "roi_fraction": round(float(self.roi_fraction), 6),
            "keypoint_count": self.keypoint_count,
            "feature_version": self.feature_version,
        }
        if include_features:
            result["keypoints"] = np.asarray(self.keypoints).tolist()
            result["descriptors"] = np.asarray(self.descriptors).tolist()
        return result


@dataclass(frozen=True)
class MatchResult:
    """Result of matching two fingerprint templates."""

    score: float
    accepted: bool
    good_matches: int
    inliers: int
    inlier_ratio: float
    descriptor_similarity: float
    spatial_coverage: float
    query_quality: float
    reference_quality: float
    transform: tuple[float, float, float, float, float, float] | None = None
    reason: str | None = None
    reference_template_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(float(self.score), 6),
            "accepted": bool(self.accepted),
            "good_matches": int(self.good_matches),
            "inliers": int(self.inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 6),
            "descriptor_similarity": round(float(self.descriptor_similarity), 6),
            "spatial_coverage": round(float(self.spatial_coverage), 6),
            "query_quality": round(float(self.query_quality), 6),
            "reference_quality": round(float(self.reference_quality), 6),
            "transform": list(self.transform) if self.transform is not None else None,
            "reason": self.reason,
            "reference_template_id": self.reference_template_id,
        }


@dataclass(frozen=True)
class FingerAssignment:
    """One query crop assigned to one enrolled finger label."""

    query_index: int
    finger: str
    result: MatchResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_index": int(self.query_index),
            "finger": self.finger,
            "match": self.result.to_dict(),
        }


@dataclass(frozen=True)
class IdentityMatch:
    """Aggregated, one-to-one match of query crops against one identity."""

    identity: str
    score: float
    accepted: bool
    matched_fingers: int
    query_fingers: int
    assignments: tuple[FingerAssignment, ...] = ()
    ambiguous: bool = False
    usable_query_fingers: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "score": round(float(self.score), 6),
            "accepted": bool(self.accepted),
            "ambiguous": bool(self.ambiguous),
            "matched_fingers": int(self.matched_fingers),
            "query_fingers": int(self.query_fingers),
            "usable_query_fingers": int(
                self.query_fingers
                if self.usable_query_fingers is None
                else self.usable_query_fingers
            ),
            "assignments": [assignment.to_dict() for assignment in self.assignments],
        }


@dataclass
class EnrollmentReport:
    """Counts and non-fatal errors produced during incremental enrollment."""

    added: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def merge(self, other: EnrollmentReport) -> None:
        self.added += other.added
        self.skipped += other.skipped
        self.failed += other.failed
        self.errors.extend(other.errors)

    def add_error(self, source: str | Path, error: Exception | str) -> None:
        self.failed += 1
        self.errors.append({"source": str(source), "error": str(error)})

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": self.added,
            "skipped": self.skipped,
            "failed": self.failed,
            "errors": self.errors,
        }
