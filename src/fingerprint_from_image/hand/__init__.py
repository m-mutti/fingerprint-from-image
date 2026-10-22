"""Learned hand -> fingertip localisation (numpy + Pillow + onnxruntime, no OpenCV).

Pipeline: BlazePalm palm detection on a letterboxed 192 px tensor, a rotated
hand ROI, MediaPipe's 21-point hand landmark model, then oriented
distal-phalanx boxes sampled from the original photograph and normalised for
ridge visibility. See ``README.md`` in this directory.

``onnxruntime`` is imported lazily, so the numpy geometry in this package can be
used (and tested) without it.
"""

from __future__ import annotations

from .crops import CropConfig, FingerBox, finger_boxes
from .detector import DetectorConfig, HandFingerDetector, HandNotFoundError, HandResult
from .enhance import clahe, flatten_illumination, normalize_fingertip, ridge_quality
from .geometry import RotatedRect
from .runtime import LANDMARK_MODEL, PALM_MODEL
from .topology import FINGERS, side_from_score
from .visualize import crop_mosaic, draw_hands

__all__ = [
    "FINGERS",
    "LANDMARK_MODEL",
    "PALM_MODEL",
    "CropConfig",
    "DetectorConfig",
    "FingerBox",
    "HandFingerDetector",
    "HandNotFoundError",
    "HandResult",
    "RotatedRect",
    "clahe",
    "crop_mosaic",
    "draw_hands",
    "finger_boxes",
    "flatten_illumination",
    "normalize_fingertip",
    "ridge_quality",
    "side_from_score",
]
