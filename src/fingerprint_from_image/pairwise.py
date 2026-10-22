"""Pairwise template matching with mutual ratio tests and affine RANSAC."""

from __future__ import annotations

import math

import cv2 as cv
import numpy as np

from .config import MatcherConfig
from .errors import FingerprintMatchingError
from .templates import FingerprintTemplate, MatchResult


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
