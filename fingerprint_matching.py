"""Fingerprint enrollment, storage, and matching framework.

The matcher stores RootSIFT feature templates rather than source images.  It can
consume either already-cropped fingerprint images or photographs of hands via
``fingerprint_extraction.extract_fingerprints``.

The public API is intentionally small: :class:`FingerprintMatcher` provides
enrollment, identification, and verification, while ``create_template`` and
``match_templates`` are useful when no database is needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

from fingerprint_extraction import (
    EXTRACTION_VERSION,
    ExtractionConfig,
    FingerprintCrop,
    FingerprintExtractionError,
    extract_fingerprints,
)

SCHEMA_VERSION = 3
FEATURE_VERSION = "rootsift-v1"
APPLICATION_ID = 0x46505254  # ASCII: FPRT
FEATURE_CONFIG_FIELDS = (
    "max_features",
    "sift_contrast_threshold",
    "sift_edge_threshold",
    "max_image_dimension",
)
IMAGE_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class FingerprintMatchingError(RuntimeError):
    """Base class for errors raised by the matching framework."""


class FingerprintQualityError(FingerprintMatchingError):
    """Raised when an image cannot produce a useful feature template."""


class FingerprintDatabaseError(FingerprintMatchingError):
    """Raised for an incompatible or corrupt template database."""


@dataclass(frozen=True)
class MatcherConfig:
    """Parameters used for RootSIFT extraction and geometric matching.

    Scores are normalized to the inclusive range ``[0, 1]``.  The defaults are
    deliberately conservative starting points, not a calibrated biometric
    operating point.  A real deployment should tune the two acceptance
    thresholds on representative genuine/impostor pairs.
    """

    max_features: int = 1600
    sift_contrast_threshold: float = 0.02
    sift_edge_threshold: float = 12.0
    max_image_dimension: int = 1400
    min_keypoints: int = 8
    min_quality: float = 0.08
    min_roi_fraction: float = 0.15
    min_orientation_coherence: float = 0.30
    ratio_threshold: float = 0.78
    min_good_matches: int = 8
    min_inliers: int = 6
    target_inliers: int = 18
    # Fraction of the image diagonal after coordinate normalization.
    ransac_reproj_threshold: float = 0.012
    ransac_max_iterations: int = 3000
    ransac_confidence: float = 0.995
    min_affine_scale: float = 0.5
    max_affine_scale: float = 2.0
    min_spatial_coverage: float = 0.03
    finger_match_threshold: float = 0.38
    identity_accept_threshold: float = 0.42
    min_hand_fingers: int = 2
    min_matched_finger_fraction: float = 0.5
    ambiguity_margin: float = 0.04

    def __post_init__(self) -> None:
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.max_image_dimension < 64:
            raise ValueError("max_image_dimension must be at least 64")
        if self.min_keypoints < 2:
            raise ValueError("min_keypoints must be at least 2")
        if self.min_good_matches < 2:
            raise ValueError("min_good_matches must be at least 2")
        if self.min_inliers < 2:
            raise ValueError("min_inliers must be at least 2")
        if self.target_inliers < self.min_inliers:
            raise ValueError("target_inliers cannot be smaller than min_inliers")
        if self.min_hand_fingers < 2:
            raise ValueError("min_hand_fingers must be at least 2")
        for name in (
            "min_quality",
            "min_roi_fraction",
            "min_orientation_coherence",
            "ratio_threshold",
            "ransac_confidence",
            "min_spatial_coverage",
            "finger_match_threshold",
            "identity_accept_threshold",
            "min_matched_finger_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.ransac_reproj_threshold <= 0:
            raise ValueError("ransac_reproj_threshold must be positive")
        if self.ransac_max_iterations < 1:
            raise ValueError("ransac_max_iterations must be positive")
        if not 0.0 < self.min_affine_scale <= self.max_affine_scale:
            raise ValueError(
                "affine scale bounds must satisfy 0 < min_affine_scale "
                "<= max_affine_scale"
            )
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be in [0, 1]")


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


def _as_gray_uint8(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError as exc:
            raise FingerprintQualityError(f"Could not read image: {path}") from exc
        data = cv.imdecode(encoded, cv.IMREAD_UNCHANGED) if encoded.size else None
        if data is None:
            raise FingerprintQualityError(f"Could not decode image: {path}")
    else:
        data = np.asarray(image)

    if data.size == 0:
        raise FingerprintQualityError("Fingerprint image is empty")
    if data.dtype != np.uint8:
        finite = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
        data = cv.normalize(finite, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if data.ndim == 2:
        gray = data
    elif data.ndim == 3 and data.shape[2] == 3:
        gray = cv.cvtColor(data, cv.COLOR_BGR2GRAY)
    elif data.ndim == 3 and data.shape[2] == 4:
        gray = cv.cvtColor(data, cv.COLOR_BGRA2GRAY)
    else:
        raise FingerprintQualityError(
            f"Expected a grayscale or color image, got shape {data.shape}"
        )
    return np.ascontiguousarray(gray)


def _resize_if_needed(gray: np.ndarray, maximum: int) -> np.ndarray:
    height, width = gray.shape
    largest = max(height, width)
    if largest <= maximum:
        return gray
    scale = maximum / float(largest)
    return cv.resize(
        gray,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv.INTER_AREA,
    )


def _fingerprint_roi(gray: np.ndarray) -> np.ndarray:
    """Estimate a filled ridge region while excluding crop/background edges."""

    height, width = gray.shape
    if min(height, width) < 12:
        return np.full_like(gray, 255)

    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    _threshold, dark = cv.threshold(
        blurred, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU
    )
    # Link ridge fragments into a region, then fill the largest exterior hulls.
    radius = max(1, round(min(height, width) * 0.025))
    radius = min(radius, 15)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    connected = cv.morphologyEx(dark, cv.MORPH_CLOSE, kernel)
    connected = cv.dilate(connected, kernel, iterations=1)
    contours, _ = cv.findContours(connected, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(gray)
    minimum_area = height * width * 0.01
    contours = [item for item in contours if cv.contourArea(item) >= minimum_area]
    if contours:
        contours.sort(key=cv.contourArea, reverse=True)
        kept: list[np.ndarray] = []
        area = 0.0
        for contour in contours[:4]:
            kept.append(cv.convexHull(contour))
            area += cv.contourArea(contour)
            if area >= height * width * 0.70:
                break
        cv.drawContours(roi, kept, -1, 255, cv.FILLED)

    fraction = float(np.count_nonzero(roi) / roi.size)
    if fraction > 0.98:
        # On tightly cropped sensor images the fingerprint can legitimately
        # fill the frame. Retaining a narrow margin avoids border keypoints.
        # A tiny or empty detected region must stay tiny so the caller's
        # minimum-ROI gate can reject a distant/partial fingerprint.
        roi.fill(255)
        margin = max(1, round(min(height, width) * 0.02))
        roi[:margin, :] = 0
        roi[-margin:, :] = 0
        roi[:, :margin] = 0
        roi[:, -margin:] = 0
    elif fraction > 0.0:
        erosion = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        roi = cv.erode(roi, erosion, iterations=1)
    return roi


def _image_quality(gray: np.ndarray, roi: np.ndarray) -> float:
    pixels = gray[roi != 0]
    if pixels.size < 16:
        return 0.0
    low, high = np.percentile(pixels, (5, 95))
    contrast = float(np.clip((high - low) / 160.0, 0.0, 1.0))
    laplacian_variance = float(cv.Laplacian(gray, cv.CV_32F).var())
    sharpness = laplacian_variance / (laplacian_variance + 350.0)
    otsu_value, _ = cv.threshold(
        pixels.reshape(-1, 1), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU
    )
    dark_fraction = float(np.mean(pixels < otsu_value))
    ridge_balance = float(np.clip(1.0 - abs(dark_fraction - 0.5) / 0.5, 0.0, 1.0))
    return float(np.clip(0.42 * contrast + 0.38 * sharpness + 0.20 * ridge_balance, 0, 1))


def _orientation_coherence(gray: np.ndarray, roi: np.ndarray) -> float:
    """Measure whether gradients form a locally consistent fingerprint ridge flow."""

    gradient_x = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    sigma = float(np.clip(min(gray.shape) / 64.0, 3.0, 12.0))
    xx = cv.GaussianBlur(gradient_x * gradient_x, (0, 0), sigma)
    yy = cv.GaussianBlur(gradient_y * gradient_y, (0, 0), sigma)
    xy = cv.GaussianBlur(gradient_x * gradient_y, (0, 0), sigma)
    coherence = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy) / (xx + yy + 1e-6)
    selected = roi != 0
    values = coherence[selected]
    if values.size < 16:
        return 0.0
    energy = (xx + yy)[selected]
    energy_sum = float(energy.sum())
    if energy_sum <= 1e-6:
        return 0.0
    # Energy weighting focuses on actual ridge transitions rather than the
    # pepper-like low-energy regions visible in phone-camera crops.
    return float(np.sum(values * (energy / energy_sum)))


def create_template(
    image: str | Path | np.ndarray,
    config: MatcherConfig | None = None,
    *,
    quality_hint: float | None = None,
    source: str | None = None,
) -> FingerprintTemplate:
    """Create a RootSIFT template from one cropped fingerprint image."""

    config = config or MatcherConfig()
    gray = _resize_if_needed(_as_gray_uint8(image), config.max_image_dimension)
    roi = _fingerprint_roi(gray)
    roi_fraction = float(np.count_nonzero(roi) / roi.size)
    if roi_fraction < config.min_roi_fraction:
        raise FingerprintQualityError(
            f"Fingerprint ridge region is too small ({roi_fraction:.3f} < "
            f"{config.min_roi_fraction:.3f})"
        )
    base_quality = _image_quality(gray, roi)
    orientation_coherence = _orientation_coherence(gray, roi)
    if orientation_coherence < config.min_orientation_coherence:
        raise FingerprintQualityError(
            "Image does not contain sufficiently coherent fingerprint-like "
            f"ridge flow ({orientation_coherence:.3f} < "
            f"{config.min_orientation_coherence:.3f})"
        )

    if not hasattr(cv, "SIFT_create"):
        raise FingerprintMatchingError(
            "This OpenCV build does not provide SIFT; install a current "
            "opencv-python or opencv-contrib-python package"
        )
    sift = cv.SIFT_create(
        nfeatures=config.max_features,
        contrastThreshold=config.sift_contrast_threshold,
        edgeThreshold=config.sift_edge_threshold,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, roi)
    count = len(keypoints) if keypoints is not None else 0
    if descriptors is None or count < config.min_keypoints:
        raise FingerprintQualityError(
            f"Only {count} usable keypoints were found; at least "
            f"{config.min_keypoints} are required"
        )

    descriptors = np.asarray(descriptors, dtype=np.float32)
    # RootSIFT: L1-normalize the non-negative SIFT histogram and apply the
    # Hellinger square root.  Euclidean distance then approximates Hellinger.
    descriptors /= np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-12)
    descriptors = np.sqrt(descriptors, out=descriptors)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)

    feature_quality = min(1.0, count / max(32.0, config.min_keypoints * 3.0))
    quality = 0.78 * base_quality + 0.22 * feature_quality
    if quality_hint is not None:
        if not 0.0 <= quality_hint <= 1.0:
            raise ValueError("quality_hint must be in [0, 1]")
        quality = math.sqrt(max(0.0, quality * quality_hint))
    quality = float(np.clip(quality, 0.0, 1.0))
    if quality < config.min_quality:
        raise FingerprintQualityError(
            f"Fingerprint quality {quality:.3f} is below the configured minimum "
            f"of {config.min_quality:.3f}"
        )

    return FingerprintTemplate(
        keypoints=np.ascontiguousarray(points, dtype="<f4"),
        descriptors=np.ascontiguousarray(descriptors, dtype="<f4"),
        image_shape=(int(gray.shape[0]), int(gray.shape[1])),
        quality=quality,
        roi_fraction=roi_fraction,
        source=source,
    )


def _ratio_matches(
    query: np.ndarray, train: np.ndarray, ratio: float
) -> dict[int, tuple[int, float]]:
    if query.shape[0] < 1 or train.shape[0] < 2:
        return {}
    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    pairs = matcher.knnMatch(query, train, k=2)
    accepted: dict[int, tuple[int, float]] = {}
    for pair in pairs:
        if len(pair) != 2:
            continue
        best, second = pair
        if second.distance > 0 and best.distance < ratio * second.distance:
            accepted[int(best.queryIdx)] = (int(best.trainIdx), float(best.distance))
    return accepted


def _spatial_coverage(points: np.ndarray, shape: tuple[int, int]) -> float:
    if points.shape[0] < 3:
        return 0.0
    hull = cv.convexHull(np.asarray(points, dtype=np.float32))
    hull_area = float(cv.contourArea(hull))
    image_area = max(1.0, float(shape[0] * shape[1]))
    # Fingerprint correspondences seldom span an entire bounding rectangle;
    # reaching 18% is already strong spatial support.
    return float(np.clip(hull_area / (image_area * 0.18), 0.0, 1.0))


def _empty_match(
    query: FingerprintTemplate,
    reference: FingerprintTemplate,
    reason: str,
    *,
    good_matches: int = 0,
) -> MatchResult:
    return MatchResult(
        score=0.0,
        accepted=False,
        good_matches=good_matches,
        inliers=0,
        inlier_ratio=0.0,
        descriptor_similarity=0.0,
        spatial_coverage=0.0,
        query_quality=query.quality,
        reference_quality=reference.quality,
        reason=reason,
        reference_template_id=reference.template_id,
    )


def match_templates(
    query: FingerprintTemplate,
    reference: FingerprintTemplate,
    config: MatcherConfig | None = None,
) -> MatchResult:
    """Match two templates with mutual ratio tests and affine RANSAC."""

    config = config or MatcherConfig()
    if query.feature_version != reference.feature_version:
        raise FingerprintMatchingError("Cannot compare different feature versions")
    forward = _ratio_matches(query.descriptors, reference.descriptors, config.ratio_threshold)
    reverse = _ratio_matches(reference.descriptors, query.descriptors, config.ratio_threshold)
    mutual: list[tuple[int, int, float]] = []
    for query_index, (reference_index, distance) in forward.items():
        backward = reverse.get(reference_index)
        if backward is not None and backward[0] == query_index:
            mutual.append((query_index, reference_index, distance))

    if len(mutual) < config.min_good_matches:
        return _empty_match(
            query,
            reference,
            f"insufficient mutual matches ({len(mutual)} < {config.min_good_matches})",
            good_matches=len(mutual),
        )

    source_pixels = np.asarray(
        [query.keypoints[item[0]] for item in mutual], dtype=np.float32
    )
    target_pixels = np.asarray(
        [reference.keypoints[item[1]] for item in mutual], dtype=np.float32
    )
    query_diagonal = math.hypot(*query.image_shape)
    reference_diagonal = math.hypot(*reference.image_shape)
    source_points = source_pixels / query_diagonal
    target_points = target_pixels / reference_diagonal
    matrix, inlier_mask = cv.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv.RANSAC,
        ransacReprojThreshold=config.ransac_reproj_threshold,
        maxIters=config.ransac_max_iterations,
        confidence=config.ransac_confidence,
        refineIters=10,
    )
    if matrix is None or inlier_mask is None:
        return _empty_match(
            query,
            reference,
            "RANSAC could not estimate a consistent transform",
            good_matches=len(mutual),
        )

    mask = np.asarray(inlier_mask).reshape(-1).astype(bool)
    inliers = int(np.count_nonzero(mask))
    if inliers < config.min_inliers:
        return _empty_match(
            query,
            reference,
            f"insufficient geometric inliers ({inliers} < {config.min_inliers})",
            good_matches=len(mutual),
        )

    inlier_ratio = inliers / float(len(mutual))
    distances = np.asarray([item[2] for item in mutual], dtype=np.float32)[mask]
    descriptor_similarity = float(
        np.clip(1.0 - float(np.mean(distances)) / math.sqrt(2.0), 0.0, 1.0)
    )
    query_coverage = _spatial_coverage(source_pixels[mask], query.image_shape)
    reference_coverage = _spatial_coverage(target_pixels[mask], reference.image_shape)
    spatial_coverage = math.sqrt(query_coverage * reference_coverage)
    support = min(1.0, inliers / float(config.target_inliers))
    quality_weight = 0.78 + 0.22 * math.sqrt(query.quality * reference.quality)
    affine_scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
    transform = tuple(float(value) for value in np.asarray(matrix).reshape(-1))
    if not config.min_affine_scale <= affine_scale <= config.max_affine_scale:
        return MatchResult(
            score=0.0,
            accepted=False,
            good_matches=len(mutual),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            descriptor_similarity=descriptor_similarity,
            spatial_coverage=spatial_coverage,
            query_quality=query.quality,
            reference_quality=reference.quality,
            transform=transform,  # type: ignore[arg-type]
            reason=(
                f"affine scale {affine_scale:.3f} is outside the allowed range "
                f"[{config.min_affine_scale:.3f}, {config.max_affine_scale:.3f}]"
            ),
            reference_template_id=reference.template_id,
        )
    if spatial_coverage < config.min_spatial_coverage:
        return MatchResult(
            score=0.0,
            accepted=False,
            good_matches=len(mutual),
            inliers=inliers,
            inlier_ratio=inlier_ratio,
            descriptor_similarity=descriptor_similarity,
            spatial_coverage=spatial_coverage,
            query_quality=query.quality,
            reference_quality=reference.quality,
            transform=transform,  # type: ignore[arg-type]
            reason=(
                f"spatial coverage {spatial_coverage:.4f} is below the minimum "
                f"{config.min_spatial_coverage:.4f}"
            ),
            reference_template_id=reference.template_id,
        )
    # Descriptor agreement alone is weak evidence on repetitive ridge patterns.
    # Requiring geometric support multiplicatively suppresses small accidental
    # RANSAC clusters while leaving well-supported genuine pairs near one.
    score = quality_weight * math.sqrt(support) * (
        0.55 * inlier_ratio
        + 0.25 * spatial_coverage
        + 0.20 * descriptor_similarity
    )
    score = float(np.clip(score, 0.0, 1.0))
    return MatchResult(
        score=score,
        accepted=score >= config.finger_match_threshold,
        good_matches=len(mutual),
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        descriptor_similarity=descriptor_similarity,
        spatial_coverage=spatial_coverage,
        query_quality=query.quality,
        reference_quality=reference.quality,
        transform=transform,  # type: ignore[arg-type]
        reference_template_id=reference.template_id,
    )


class FingerprintStore:
    """Versioned SQLite persistence for image-free fingerprint templates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                descriptor = os.open(
                    self.path,
                    os.O_CREAT | os.O_EXCL | os.O_RDWR,
                    0o600,
                )
            except FileExistsError:
                pass
            else:
                os.close(descriptor)
        try:
            self.connection = sqlite3.connect(str(self.path), timeout=30.0)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA busy_timeout = 30000")
            self.connection.execute("PRAGMA secure_delete = ON")
            self.connection.execute("PRAGMA journal_size_limit = 0")
            self._initialize()
        except sqlite3.Error as exc:
            if hasattr(self, "connection"):
                self.connection.close()
            raise FingerprintDatabaseError(
                f"Could not open fingerprint database {self.path}: {exc}"
            ) from exc
        except Exception:
            if hasattr(self, "connection"):
                self.connection.close()
            raise

    def __enter__(self) -> FingerprintStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _initialize(self) -> None:
        application_id = int(self.connection.execute("PRAGMA application_id").fetchone()[0])
        existing_objects = {
            (str(row[0]), str(row[1]))
            for row in self.connection.execute(
                "SELECT type, name FROM sqlite_master"
            )
            if not str(row[1]).startswith("sqlite_")
        }
        if application_id not in (0, APPLICATION_ID):
            raise FingerprintDatabaseError(
                f"{self.path} belongs to a different SQLite application"
            )
        existing_tables = {
            name for object_type, name in existing_objects if object_type == "table"
        }
        if existing_objects:
            required = {"fingerprint_metadata", "identities", "templates"}
            if application_id != APPLICATION_ID or not required.issubset(existing_tables):
                raise FingerprintDatabaseError(
                    f"Refusing to modify non-fingerprint SQLite database: {self.path}"
                )
            metadata = {
                str(row["key"]): str(row["value"])
                for row in self.connection.execute(
                    "SELECT key, value FROM fingerprint_metadata"
                )
            }
            try:
                schema = int(metadata.get("schema_version", "-1"))
            except ValueError as exc:
                raise FingerprintDatabaseError("Invalid database schema metadata") from exc
            if schema != SCHEMA_VERSION:
                raise FingerprintDatabaseError(
                    f"Database schema version {schema} is not supported; "
                    f"expected {SCHEMA_VERSION}"
                )
            if metadata.get("feature_version") != FEATURE_VERSION:
                raise FingerprintDatabaseError(
                    "Database contains an incompatible feature representation"
                )
            if metadata.get("extraction_version") != EXTRACTION_VERSION:
                raise FingerprintDatabaseError(
                    "Database contains templates from an incompatible hand extractor"
                )
            user_version = int(
                self.connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if user_version != SCHEMA_VERSION:
                raise FingerprintDatabaseError(
                    "Database schema metadata and SQLite user_version disagree"
                )
            return

        if application_id != 0:
            raise FingerprintDatabaseError(
                f"{self.path} is not an empty SQLite database"
            )
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                """
                CREATE TABLE fingerprint_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE identities (
                    name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity TEXT NOT NULL REFERENCES identities(name) ON DELETE CASCADE,
                    finger TEXT NOT NULL,
                    source_path TEXT,
                    source_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    height INTEGER NOT NULL CHECK(height > 0),
                    width INTEGER NOT NULL CHECK(width > 0),
                    quality REAL NOT NULL CHECK(quality BETWEEN 0 AND 1),
                    roi_fraction REAL NOT NULL CHECK(roi_fraction BETWEEN 0 AND 1),
                    keypoint_count INTEGER NOT NULL CHECK(keypoint_count > 0),
                    descriptor_size INTEGER NOT NULL CHECK(descriptor_size = 128),
                    keypoints BLOB NOT NULL,
                    descriptors BLOB NOT NULL,
                    feature_digest TEXT NOT NULL,
                    UNIQUE(identity, finger, source_digest)
                )
                """
            )
            for name, columns, unique in (
                ("idx_templates_identity", "identity", False),
                ("idx_templates_identity_finger", "identity, finger", False),
                ("uq_templates_source_digest", "source_digest", True),
                ("uq_templates_feature_digest", "feature_digest", True),
            ):
                self.connection.execute(
                    f"CREATE {'UNIQUE ' if unique else ''}INDEX "
                    f"{name} ON templates({columns})"
                )
            self.connection.executemany(
                "INSERT INTO fingerprint_metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("feature_version", FEATURE_VERSION),
                    ("extraction_version", EXTRACTION_VERSION),
                    ("created_at", _utc_now()),
                ),
            )
            self.connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    @staticmethod
    def _validate_label(label: str, kind: str) -> str:
        cleaned = label.strip()
        if not cleaned:
            raise ValueError(f"{kind} cannot be empty")
        if len(cleaned) > 255:
            raise ValueError(f"{kind} cannot exceed 255 characters")
        return cleaned

    def add_template(
        self,
        identity: str,
        finger: str,
        template: FingerprintTemplate,
        *,
        source_digest: str,
        source_path: str | None = None,
        _within_transaction: bool = False,
    ) -> int | None:
        """Add a template, returning its id or ``None`` for a duplicate."""

        bound_keys = int(
            self.connection.execute(
                """
                SELECT COUNT(*)
                FROM fingerprint_metadata
                WHERE key IN ('feature_config', 'extraction_config')
                """
            ).fetchone()[0]
        )
        if bound_keys != 2:
            raise FingerprintDatabaseError(
                "Database configuration is not bound; enroll through "
                "FingerprintMatcher or bind both configurations first"
            )
        identity = self._validate_label(identity, "identity")
        finger = self._validate_label(finger, "finger")
        if template.feature_version != FEATURE_VERSION:
            raise FingerprintDatabaseError("Unsupported template feature version")
        points = np.ascontiguousarray(template.keypoints, dtype="<f4")
        descriptors = np.ascontiguousarray(template.descriptors, dtype="<f4")
        points_blob = points.tobytes(order="C")
        descriptors_blob = descriptors.tobytes(order="C")
        feature_digest = _feature_digest(points_blob, descriptors_blob)
        try:
            transaction = nullcontext() if _within_transaction else self.connection
            with transaction:
                existing = self.connection.execute(
                    """
                    SELECT identity, finger
                    FROM templates
                    WHERE source_digest = ? OR feature_digest = ?
                    LIMIT 1
                    """,
                    (source_digest, feature_digest),
                ).fetchone()
                if existing is not None:
                    enrolled_as = (str(existing["identity"]), str(existing["finger"]))
                    if enrolled_as == (identity, finger):
                        return None
                    raise FingerprintDatabaseError(
                        "The same source or feature template is already enrolled as "
                        f"{enrolled_as[0]}/{enrolled_as[1]}; refusing conflicting "
                        f"enrollment as {identity}/{finger}"
                    )
                self.connection.execute(
                    "INSERT OR IGNORE INTO identities(name, created_at) VALUES (?, ?)",
                    (identity, _utc_now()),
                )
                try:
                    cursor = self.connection.execute(
                        """
                        INSERT INTO templates(
                            identity, finger, source_path, source_digest, created_at,
                            feature_version, height, width, quality, roi_fraction,
                            keypoint_count, descriptor_size, keypoints, descriptors,
                            feature_digest
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identity,
                            finger,
                            source_path,
                            source_digest,
                            _utc_now(),
                            FEATURE_VERSION,
                            int(template.image_shape[0]),
                            int(template.image_shape[1]),
                            float(template.quality),
                            float(template.roi_fraction),
                            int(points.shape[0]),
                            int(descriptors.shape[1]),
                            sqlite3.Binary(points_blob),
                            sqlite3.Binary(descriptors_blob),
                            feature_digest,
                        ),
                    )
                except sqlite3.IntegrityError:
                    raced = self.connection.execute(
                        """
                        SELECT identity, finger
                        FROM templates
                        WHERE source_digest = ? OR feature_digest = ?
                        LIMIT 1
                        """,
                        (source_digest, feature_digest),
                    ).fetchone()
                    if raced is not None and (
                        str(raced["identity"]),
                        str(raced["finger"]),
                    ) == (identity, finger):
                        return None
                    if raced is not None:
                        raise FingerprintDatabaseError(
                            "The same source or feature template was concurrently "
                            f"enrolled as {raced['identity']}/{raced['finger']}; "
                            f"refusing conflicting enrollment as {identity}/{finger}"
                        ) from None
                    raise
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise FingerprintDatabaseError(f"Could not store template: {exc}") from exc

    def _decode_template(self, row: sqlite3.Row) -> FingerprintTemplate:
        count = int(row["keypoint_count"])
        descriptor_size = int(row["descriptor_size"])
        if count < 1 or count > 1_000_000 or descriptor_size != 128:
            raise FingerprintDatabaseError(
                f"Template {row['id']} has invalid array dimensions"
            )
        points_blob = bytes(row["keypoints"])
        descriptors_blob = bytes(row["descriptors"])
        if len(points_blob) != count * 2 * 4:
            raise FingerprintDatabaseError(
                f"Template {row['id']} has a corrupt keypoint array"
            )
        if len(descriptors_blob) != count * descriptor_size * 4:
            raise FingerprintDatabaseError(
                f"Template {row['id']} has a corrupt descriptor array"
            )
        if _feature_digest(points_blob, descriptors_blob) != row["feature_digest"]:
            raise FingerprintDatabaseError(
                f"Template {row['id']} failed its feature checksum"
            )
        points = np.frombuffer(points_blob, dtype="<f4").reshape(count, 2).copy()
        descriptors = (
            np.frombuffer(descriptors_blob, dtype="<f4")
            .reshape(count, descriptor_size)
            .copy()
        )
        return FingerprintTemplate(
            keypoints=points,
            descriptors=descriptors,
            image_shape=(int(row["height"]), int(row["width"])),
            quality=float(row["quality"]),
            roi_fraction=float(row["roi_fraction"]),
            identity=str(row["identity"]),
            finger=str(row["finger"]),
            template_id=int(row["id"]),
            source=str(row["source_path"]) if row["source_path"] else None,
            feature_version=str(row["feature_version"]),
        )

    def templates(self, identity: str | None = None) -> Iterator[FingerprintTemplate]:
        if identity is None:
            cursor = self.connection.execute("SELECT * FROM templates ORDER BY id")
        else:
            cursor = self.connection.execute(
                "SELECT * FROM templates WHERE identity = ? ORDER BY id", (identity,)
            )
        for row in cursor:
            yield self._decode_template(row)

    def identities(self) -> list[str]:
        return [
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM identities WHERE EXISTS "
                "(SELECT 1 FROM templates WHERE templates.identity = identities.name) "
                "ORDER BY name"
            )
        ]

    def has_identity(self, identity: str) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM templates WHERE identity = ? LIMIT 1", (identity,)
            ).fetchone()
            is not None
        )

    def clear(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM templates")
            self.connection.execute("DELETE FROM identities")
        # Rebuild the file so freed template pages are not retained in its
        # freelist. Filesystem snapshots and storage media may still retain
        # historical blocks, so this is not a forensic-erasure guarantee.
        self.connection.execute("VACUUM")

    def stats(self) -> dict[str, Any]:
        template_count = int(
            self.connection.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
        )
        identity_count = int(
            self.connection.execute(
                "SELECT COUNT(DISTINCT identity) FROM templates"
            ).fetchone()[0]
        )
        per_identity = {
            str(row[0]): int(row[1])
            for row in self.connection.execute(
                "SELECT identity, COUNT(*) FROM templates "
                "GROUP BY identity ORDER BY identity"
            )
        }
        per_finger = {
            f"{row[0]}/{row[1]}": int(row[2])
            for row in self.connection.execute(
                "SELECT identity, finger, COUNT(*) FROM templates "
                "GROUP BY identity, finger ORDER BY identity, finger"
            )
        }
        return {
            "database": str(self.path.resolve()),
            "schema_version": SCHEMA_VERSION,
            "feature_version": FEATURE_VERSION,
            "identities": identity_count,
            "templates": template_count,
            "templates_per_identity": per_identity,
            "templates_per_finger": per_finger,
        }

    def bind_feature_config(self, config: MatcherConfig) -> None:
        """Record or validate settings that affect persisted feature vectors."""

        config_dict = asdict(config)
        signature = json.dumps(
            {name: config_dict[name] for name in FEATURE_CONFIG_FIELDS},
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO fingerprint_metadata(key, value) VALUES (?, ?)",
                ("feature_config", signature),
            )
        row = self.connection.execute(
            "SELECT value FROM fingerprint_metadata WHERE key = 'feature_config'"
        ).fetchone()
        if row is None:
            raise FingerprintDatabaseError("Database feature configuration is missing")
        if str(row[0]) != signature:
            raise FingerprintDatabaseError(
                "Matcher feature settings differ from this database; use the "
                "original settings or rebuild the database"
            )

    def bind_extraction_config(self, config: ExtractionConfig) -> None:
        """Record or validate the hand-crop settings used by this database."""

        signature = _extraction_config_signature(config)
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO fingerprint_metadata(key, value) VALUES (?, ?)",
                ("extraction_config", signature),
            )
        row = self.connection.execute(
            "SELECT value FROM fingerprint_metadata WHERE key = 'extraction_config'"
        ).fetchone()
        if row is None:
            raise FingerprintDatabaseError("Database extraction configuration is missing")
        if str(row[0]) != signature:
            raise FingerprintDatabaseError(
                "Hand extraction settings differ from this database; use the "
                "original settings or rebuild the database"
            )


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


def _image_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: str(path).casefold(),
    )


class FingerprintMatcher:
    """End-to-end enrollment and matching service backed by SQLite."""

    def __init__(
        self,
        database: str | Path | FingerprintStore = "fingerprints.db",
        config: MatcherConfig | None = None,
        extraction_config: ExtractionConfig | None = None,
    ) -> None:
        self.config = config or MatcherConfig()
        self.extraction_config = extraction_config or ExtractionConfig()
        self._owns_store = not isinstance(database, FingerprintStore)
        self.store = (
            database if isinstance(database, FingerprintStore) else FingerprintStore(database)
        )
        try:
            self.store.bind_feature_config(self.config)
            self.store.bind_extraction_config(self.extraction_config)
        except Exception:
            if self._owns_store:
                self.store.close()
            raise

    def __enter__(self) -> FingerprintMatcher:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_store:
            self.store.close()

    def clear(self) -> None:
        self.store.clear()

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def template(
        self,
        image: str | Path | np.ndarray,
        *,
        quality_hint: float | None = None,
    ) -> FingerprintTemplate:
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        return create_template(
            image,
            self.config,
            quality_hint=quality_hint,
            source=source,
        )

    def match(
        self,
        query: FingerprintTemplate | str | Path | np.ndarray,
        reference: FingerprintTemplate | str | Path | np.ndarray,
    ) -> MatchResult:
        """Low-level one-fingerprint match convenience method."""

        query_template = query if isinstance(query, FingerprintTemplate) else self.template(query)
        reference_template = (
            reference
            if isinstance(reference, FingerprintTemplate)
            else self.template(reference)
        )
        return match_templates(query_template, reference_template, self.config)

    def enroll_cropped(
        self,
        identity: str,
        image: str | Path | np.ndarray,
        *,
        finger: str = "unknown",
        source_digest: str | None = None,
    ) -> bool:
        """Incrementally enroll one cropped print; return false if duplicated."""

        template = self.template(image)
        digest = source_digest or _source_digest(image)
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        template_id = self.store.add_template(
            identity,
            finger,
            template,
            source_digest=digest,
            source_path=source,
        )
        return template_id is not None

    def _enroll_hand_crops(
        self,
        identity: str,
        image: str | Path | np.ndarray,
        crops: Sequence[FingerprintCrop],
    ) -> EnrollmentReport:
        report = EnrollmentReport()
        source_label = _source_description(image)
        if len(crops) < self.config.min_hand_fingers:
            report.add_error(
                source_label,
                f"hand extraction produced {len(crops)} crop(s); at least "
                f"{self.config.min_hand_fingers} are required",
            )
            return report
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        prepared: list[tuple[str, FingerprintTemplate]] = []
        for index, crop in enumerate(crops, start=1):
            finger = f"finger_{index}"
            try:
                prepared.append(
                    (
                        finger,
                        create_template(
                            crop.image,
                            self.config,
                            quality_hint=crop.quality,
                            source=source,
                        ),
                    )
                )
            except (FingerprintMatchingError, ValueError) as exc:
                report.add_error(f"{source_label}#{finger}", exc)
        if report.failed:
            # One hand capture is a single enrollment unit. Do not create a
            # sparse identity when any expected crop is unusable.
            return report

        base_digest = _source_digest(image)
        added: list[bool] = []
        try:
            with self.store.connection:
                self.store.connection.execute("BEGIN IMMEDIATE")
                for index, (finger, template) in enumerate(prepared, start=1):
                    template_id = self.store.add_template(
                        identity,
                        finger,
                        template,
                        source_digest=_hand_crop_digest(
                            base_digest, index, self.extraction_config
                        ),
                        source_path=source,
                        _within_transaction=True,
                    )
                    added.append(template_id is not None)
        except (FingerprintMatchingError, ValueError, sqlite3.Error) as exc:
            report.add_error(source_label, f"hand enrollment rolled back: {exc}")
            return report
        report.added = sum(added)
        report.skipped = len(added) - report.added
        return report

    def enroll_hand(
        self, identity: str, image: str | Path | np.ndarray
    ) -> EnrollmentReport:
        """Atomically extract and enroll every usable crop in a hand image."""

        try:
            crops = extract_fingerprints(image, self.extraction_config)
        except (FingerprintExtractionError, ValueError) as exc:
            report = EnrollmentReport()
            report.add_error(_source_description(image), exc)
            return report
        if not crops:
            report = EnrollmentReport()
            report.add_error(
                _source_description(image),
                "no fingerprints were extracted from the hand image",
            )
            return report
        return self._enroll_hand_crops(identity, image, crops)

    def enroll(
        self,
        identity: str,
        images: str | Path | np.ndarray | Sequence[str | Path | np.ndarray],
        *,
        finger: str = "unknown",
        kind: str = "cropped",
    ) -> EnrollmentReport:
        """Enroll one or more cropped prints or hand captures.

        ``kind`` may be ``"cropped"``, ``"hand"``, or ``"auto"``.  Auto mode
        treats an image as a hand only when extraction finds at least two crops.
        """

        if kind not in {"auto", "cropped", "hand"}:
            raise ValueError("kind must be 'auto', 'cropped', or 'hand'")
        if isinstance(images, (str, Path, np.ndarray)):
            sources: Sequence[str | Path | np.ndarray] = [images]
        else:
            sources = images
        report = EnrollmentReport()
        for image in sources:
            actual_kind = kind
            crops: list[FingerprintCrop] | None = None
            if kind == "auto":
                try:
                    crops = extract_fingerprints(image, self.extraction_config)
                except (FingerprintExtractionError, ValueError):
                    crops = []
                actual_kind = "hand" if len(crops) >= 2 else "cropped"
            if actual_kind == "hand":
                # Avoid doing extraction twice in the common auto case by using
                # the dedicated path only when no crops were already obtained.
                if crops is None:
                    report.merge(self.enroll_hand(identity, image))
                    continue
                if not crops:
                    report.add_error(
                        _source_description(image), "no fingerprints were extracted"
                    )
                    continue
                report.merge(self._enroll_hand_crops(identity, image, crops))
            else:
                try:
                    added = self.enroll_cropped(identity, image, finger=finger)
                    report.added += added
                    report.skipped += not added
                except (FingerprintMatchingError, ValueError) as exc:
                    report.add_error(_source_description(image), exc)
        return report

    def enroll_directory(self, dataset: str | Path) -> EnrollmentReport:
        """Enroll ``identity/finger/sample`` crops and ``identity/capture`` hands."""

        root = Path(dataset)
        if not root.is_dir():
            raise FingerprintMatchingError(f"Dataset directory does not exist: {root}")
        identities = sorted(
            (item for item in root.iterdir() if item.is_dir()),
            key=lambda path: path.name.casefold(),
        )
        if not identities:
            raise FingerprintMatchingError(
                f"Dataset must contain at least one identity directory: {root}"
            )
        report = EnrollmentReport()
        sources_seen = 0
        for identity_directory in identities:
            identity = identity_directory.name
            # Images directly below an identity are complete hand captures.
            captures = _image_files(identity_directory)
            sources_seen += len(captures)
            for capture in captures:
                report.merge(self.enroll_hand(identity, capture))
            # Images below a named child directory are samples of that finger.
            finger_directories = sorted(
                (item for item in identity_directory.iterdir() if item.is_dir()),
                key=lambda path: path.name.casefold(),
            )
            for finger_directory in finger_directories:
                samples = _image_files(finger_directory, recursive=True)
                sources_seen += len(samples)
                for sample in samples:
                    try:
                        added = self.enroll_cropped(
                            identity, sample, finger=finger_directory.name
                        )
                        report.added += added
                        report.skipped += not added
                    except (FingerprintMatchingError, ValueError) as exc:
                        report.add_error(sample, exc)
        if sources_seen == 0:
            raise FingerprintMatchingError(
                f"Dataset contains no supported fingerprint or hand images: {root}"
            )
        return report

    def _prepare_query(
        self,
        image: str | Path | np.ndarray,
        *,
        kind: str = "cropped",
    ) -> tuple[list[FingerprintTemplate], int]:
        """Return usable templates and the original number of query slots."""

        if kind not in {"auto", "cropped", "hand"}:
            raise ValueError("kind must be 'auto', 'cropped', or 'hand'")
        crops: list[FingerprintCrop] = []
        if kind != "cropped":
            try:
                crops = extract_fingerprints(image, self.extraction_config)
            except (FingerprintExtractionError, ValueError) as exc:
                if kind == "hand":
                    raise FingerprintMatchingError(str(exc)) from exc
            if kind == "hand" and len(crops) < self.config.min_hand_fingers:
                raise FingerprintQualityError(
                    f"Hand query produced {len(crops)} crop(s); at least "
                    f"{self.config.min_hand_fingers} are required"
                )
        if kind == "cropped" or (kind == "auto" and len(crops) < 2):
            return [self.template(image)], 1

        extracted_count = len(crops)
        templates: list[FingerprintTemplate] = []
        errors: list[str] = []
        source = str(Path(image).resolve()) if isinstance(image, (str, Path)) else None
        for index, crop in enumerate(crops, start=1):
            try:
                templates.append(
                    create_template(
                        crop.image,
                        self.config,
                        quality_hint=crop.quality,
                        source=source,
                    )
                )
            except FingerprintQualityError as exc:
                errors.append(f"crop {index}: {exc}")
        if not templates:
            detail = "; ".join(errors) if errors else "no usable crops"
            raise FingerprintQualityError(f"Hand query produced no templates ({detail})")
        if len(templates) < self.config.min_hand_fingers:
            detail = "; ".join(errors) if errors else "insufficient usable crops"
            raise FingerprintQualityError(
                f"Hand query produced only {len(templates)}/{extracted_count} usable "
                f"templates; at least {self.config.min_hand_fingers} are required "
                f"({detail})"
            )
        return templates, extracted_count

    def query_templates(
        self,
        image: str | Path | np.ndarray,
        *,
        kind: str = "cropped",
    ) -> list[FingerprintTemplate]:
        """Convert a query crop or hand image into usable templates."""

        templates, _expected_count = self._prepare_query(image, kind=kind)
        return templates

    def _match_identity(
        self,
        identity: str,
        query: Sequence[FingerprintTemplate],
        references: Sequence[FingerprintTemplate],
        *,
        expected_query_fingers: int | None = None,
    ) -> IdentityMatch:
        query_slots = (
            len(query)
            if expected_query_fingers is None
            else max(len(query), expected_query_fingers)
        )
        grouped: dict[str, list[FingerprintTemplate]] = {}
        for template in references:
            grouped.setdefault(template.finger or "unknown", []).append(template)
        fingers = sorted(grouped, key=str.casefold)
        if not query or not fingers:
            return IdentityMatch(
                identity,
                0.0,
                False,
                0,
                query_slots,
                usable_query_fingers=len(query),
            )

        best: list[list[MatchResult]] = []
        for query_template in query:
            row: list[MatchResult] = []
            for finger in fingers:
                results = [
                    match_templates(query_template, reference, self.config)
                    for reference in grouped[finger]
                ]
                row.append(max(results, key=lambda result: result.score))
            best.append(row)

        assignment = _maximum_assignment(best)
        assignments = tuple(
            FingerAssignment(query_index, fingers[finger_index], best[query_index][finger_index])
            for query_index, finger_index in assignment
        )
        accepted_assignments = [
            item
            for item in assignments
            if item.result.score >= self.config.finger_match_threshold
        ]
        # Weak, below-threshold similarities are not cumulative evidence. The
        # denominator remains the query count so identities with sparse
        # enrollment do not outrank identities that explain the full hand.
        total = sum(item.result.score for item in accepted_assignments)
        score = float(np.clip(total / max(1, query_slots), 0.0, 1.0))
        matched = len(accepted_assignments)
        required_matches = max(
            1, math.ceil(query_slots * self.config.min_matched_finger_fraction)
        )
        accepted = (
            score >= self.config.identity_accept_threshold
            and matched >= required_matches
        )
        return IdentityMatch(
            identity=identity,
            score=score,
            accepted=accepted,
            matched_fingers=matched,
            query_fingers=query_slots,
            assignments=assignments,
            usable_query_fingers=len(query),
        )

    def identify(
        self,
        image: str | Path | np.ndarray | Sequence[FingerprintTemplate],
        *,
        top_k: int = 5,
        kind: str = "cropped",
    ) -> list[IdentityMatch]:
        """Return best identities, sorted by an order-independent hand score."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, Path, np.ndarray)):
            query = list(image)
            if not all(isinstance(item, FingerprintTemplate) for item in query):
                raise TypeError("A query sequence must contain FingerprintTemplate values")
            expected_query_fingers = len(query)
        else:
            query, expected_query_fingers = self._prepare_query(
                image, kind=kind  # type: ignore[arg-type]
            )
        identities = self.store.identities()
        if not identities:
            raise FingerprintMatchingError("The fingerprint database is empty")
        matches = [
            self._match_identity(
                identity,
                query,
                list(self.store.templates(identity)),
                expected_query_fingers=expected_query_fingers,
            )
            for identity in identities
        ]
        matches.sort(key=lambda item: (-item.score, item.identity.casefold()))
        if (
            len(matches) > 1
            and matches[0].accepted
            and matches[0].score - matches[1].score < self.config.ambiguity_margin
        ):
            best_score = matches[0].score
            matches = [
                replace(item, accepted=False, ambiguous=True)
                if best_score - item.score < self.config.ambiguity_margin
                else item
                for item in matches
            ]
        return matches[:top_k]

    def verify(
        self,
        identity: str,
        image: str | Path | np.ndarray | Sequence[FingerprintTemplate],
        *,
        kind: str = "cropped",
    ) -> IdentityMatch:
        """Verify a claimed identity against all of its enrolled samples."""

        if not self.store.has_identity(identity):
            raise FingerprintMatchingError(f"Identity is not enrolled: {identity}")
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, Path, np.ndarray)):
            query = list(image)
            if not all(isinstance(item, FingerprintTemplate) for item in query):
                raise TypeError("A query sequence must contain FingerprintTemplate values")
            expected_query_fingers = len(query)
        else:
            query, expected_query_fingers = self._prepare_query(
                image, kind=kind  # type: ignore[arg-type]
            )
        return self._match_identity(
            identity,
            query,
            list(self.store.templates(identity)),
            expected_query_fingers=expected_query_fingers,
        )


def _maximum_assignment(matrix: Sequence[Sequence[MatchResult]]) -> list[tuple[int, int]]:
    """Maximize score with each query and reference label used at most once."""

    rows = len(matrix)
    columns = len(matrix[0]) if rows else 0
    if rows == 0 or columns == 0:
        return []
    if any(len(row) != columns for row in matrix):
        raise ValueError("Assignment matrix must be rectangular")

    # Finger sets are naturally small.  A bitmask DP gives an exact assignment
    # without adding scipy as a runtime dependency.
    if columns > 20:
        candidates = sorted(
            (
                (matrix[row][column].score, row, column)
                for row in range(rows)
                for column in range(columns)
            ),
            reverse=True,
        )
        used_rows: set[int] = set()
        used_columns: set[int] = set()
        result: list[tuple[int, int]] = []
        for score, row, column in candidates:
            if score <= 0 or row in used_rows or column in used_columns:
                continue
            used_rows.add(row)
            used_columns.add(column)
            result.append((row, column))
        return sorted(result)

    @cache
    def solve(row: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if row == rows:
            return 0.0, ()
        best_score, best_pairs = solve(row + 1, used)  # leave a weak crop unmatched
        for column in range(columns):
            bit = 1 << column
            if used & bit:
                continue
            tail_score, tail_pairs = solve(row + 1, used | bit)
            candidate_score = matrix[row][column].score + tail_score
            if candidate_score > best_score + 1e-12:
                best_score = candidate_score
                best_pairs = ((row, column),) + tail_pairs
        return best_score, best_pairs

    return list(solve(0, 0)[1])


def _add_database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--db", default="fingerprints.db", help="SQLite template database"
    )


def _add_image_kind_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--hand", action="store_true", help="input is a complete hand image")
    group.add_argument(
        "--cropped", action="store_true", help="input is one cropped fingerprint"
    )


def _kind_from_args(args: argparse.Namespace) -> str:
    if getattr(args, "hand", False):
        return "hand"
    if getattr(args, "cropped", False):
        return "cropped"
    # Matching an already-extracted print is the safe default. Hand extraction
    # is intentionally explicit because a connected hand silhouette can look
    # like one contour and cannot be reliably inferred from crop count alone.
    return "cropped"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll and match fingerprints using RootSIFT and affine RANSAC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="incrementally enroll an identity-organized dataset"
    )
    build.add_argument("dataset", help="dataset root (identity/finger/sample or identity/capture)")
    build.add_argument(
        "--replace", action="store_true", help="clear enrolled identities before building"
    )
    _add_database_argument(build)

    enroll = subparsers.add_parser("enroll", help="incrementally add images for an identity")
    enroll.add_argument("identity")
    enroll.add_argument("images", nargs="+")
    enroll.add_argument("--finger", default="unknown", help="label for cropped samples")
    _add_image_kind_arguments(enroll)
    _add_database_argument(enroll)

    for command in ("identify", "match"):
        identify = subparsers.add_parser(
            command,
            help="search an image against the database"
            + (" (alias of identify)" if command == "match" else ""),
        )
        identify.add_argument("image")
        identify.add_argument("--top-k", type=int, default=5)
        identify.add_argument("--threshold", type=float)
        _add_image_kind_arguments(identify)
        _add_database_argument(identify)

    verify = subparsers.add_parser("verify", help="test an image against a claimed identity")
    verify.add_argument("identity")
    verify.add_argument("image")
    verify.add_argument("--threshold", type=float)
    _add_image_kind_arguments(verify)
    _add_database_argument(verify)

    stats = subparsers.add_parser("stats", help="show database counts and versions")
    _add_database_argument(stats)
    return parser


def _json_dump(value: Mapping[str, Any], *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), file=stream)


def _restore_staged_database(source: Path, destination: Path) -> None:
    """Transactionally copy a staged SQLite image into an existing DB file.

    SQLite keeps a write transaction open on the destination during backup and
    rolls it back if copying does not finish. Writing the existing file also
    preserves its filesystem identity and its POSIX mode or Windows DACL.
    """

    source_connection = sqlite3.connect(str(source), timeout=30.0)
    destination_connection = sqlite3.connect(str(destination), timeout=30.0)
    try:
        destination_connection.execute("PRAGMA busy_timeout = 30000")
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = MatcherConfig()
    if getattr(args, "threshold", None) is not None:
        threshold = float(args.threshold)
        if not 0.0 < threshold <= 1.0:
            _json_dump({"ok": False, "error": "threshold must be in (0, 1]"}, stream=sys.stderr)
            return 2
        config = replace(
            config,
            finger_match_threshold=threshold,
            identity_accept_threshold=threshold,
        )

    try:
        if args.command in {"identify", "match", "verify", "stats"}:
            database_path = Path(args.db)
            if not database_path.is_file():
                raise FingerprintDatabaseError(
                    f"Fingerprint database does not exist: {database_path}"
                )
        if args.command == "build" and args.replace and Path(args.db).is_file():
            database_path = Path(args.db)
            # Refuse to overwrite an unrelated or unsupported file. Keeping the
            # existing connection closed during the backup also avoids Windows
            # sharing violations.
            with FingerprintStore(database_path):
                pass
            staging_path = database_path.with_name(
                f".{database_path.name}.{uuid.uuid4().hex}.staging"
            )
            try:
                with FingerprintMatcher(staging_path, config=config) as matcher:
                    report = matcher.enroll_directory(args.dataset)
                    payload = {
                        "ok": report.failed == 0,
                        "command": "build",
                        "dataset": str(Path(args.dataset).resolve()),
                        "enrollment": report.to_dict(),
                        "stats": matcher.stats(),
                        "replacement_committed": False,
                    }
                if report.failed:
                    _json_dump(payload)
                    return 1
                _restore_staged_database(staging_path, database_path)
                with FingerprintStore(database_path):
                    pass
                payload["stats"]["database"] = str(database_path.resolve())
                payload["replacement_committed"] = True
                _json_dump(payload)
                return 0
            finally:
                # Only exact paths derived from the generated staging name are
                # eligible for cleanup; no directory or wildcard is involved.
                for temporary in (
                    staging_path,
                    Path(f"{staging_path}-journal"),
                    Path(f"{staging_path}-wal"),
                    Path(f"{staging_path}-shm"),
                ):
                    temporary.unlink(missing_ok=True)
        with FingerprintMatcher(args.db, config=config) as matcher:
            if args.command == "build":
                report = matcher.enroll_directory(args.dataset)
                payload = {
                    "ok": report.failed == 0,
                    "command": "build",
                    "dataset": str(Path(args.dataset).resolve()),
                    "enrollment": report.to_dict(),
                    "stats": matcher.stats(),
                }
                _json_dump(payload)
                return 0 if report.failed == 0 else 1
            if args.command == "enroll":
                report = matcher.enroll(
                    args.identity,
                    [Path(image) for image in args.images],
                    finger=args.finger,
                    kind=_kind_from_args(args),
                )
                _json_dump(
                    {
                        "ok": report.failed == 0,
                        "command": "enroll",
                        "identity": args.identity,
                        "enrollment": report.to_dict(),
                    }
                )
                return 0 if report.failed == 0 else 1
            if args.command in {"identify", "match"}:
                results = matcher.identify(
                    args.image, top_k=args.top_k, kind=_kind_from_args(args)
                )
                _json_dump(
                    {
                        "ok": True,
                        "command": args.command,
                        "query": str(Path(args.image).resolve()),
                        "matches": [item.to_dict() for item in results],
                    }
                )
                return 0
            if args.command == "verify":
                result = matcher.verify(
                    args.identity, args.image, kind=_kind_from_args(args)
                )
                _json_dump(
                    {
                        "ok": True,
                        "command": "verify",
                        "query": str(Path(args.image).resolve()),
                        "claimed_identity": args.identity,
                        "match": result.to_dict(),
                    }
                )
                return 0 if result.accepted else 1
            if args.command == "stats":
                _json_dump({"ok": True, "command": "stats", "stats": matcher.stats()})
                return 0
    except (FingerprintMatchingError, FingerprintExtractionError, ValueError, OSError) as exc:
        _json_dump({"ok": False, "error": str(exc)}, stream=sys.stderr)
        return 2
    return 2


__all__ = [
    "EnrollmentReport",
    "FingerprintDatabaseError",
    "FingerprintMatcher",
    "FingerprintMatchingError",
    "FingerprintQualityError",
    "FingerprintStore",
    "FingerprintTemplate",
    "IdentityMatch",
    "MatchResult",
    "MatcherConfig",
    "create_template",
    "match_templates",
]


if __name__ == "__main__":
    raise SystemExit(main())
