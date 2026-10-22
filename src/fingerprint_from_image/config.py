"""Matcher configuration and persisted-format constants."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 3
FEATURE_VERSION = "rootsift-v1"
APPLICATION_ID = 0x46505254  # ASCII: FPRT
FEATURE_CONFIG_FIELDS = (
    "max_features",
    "sift_contrast_threshold",
    "sift_edge_threshold",
    "max_image_dimension",
)


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
