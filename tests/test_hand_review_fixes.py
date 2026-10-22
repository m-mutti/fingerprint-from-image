"""Regression tests for the hand-detector code review (grayscale input, error
contract with the recognizer, ROI weights, degenerate crops, config checks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fingerprint_from_image.errors import FingerprintMatchingError
from fingerprint_from_image.hand.crops import CropConfig, finger_boxes
from fingerprint_from_image.hand.detector import (
    DetectorConfig,
    HandFingerDetector,
    HandNotFoundError,
    _color_image,
)
from fingerprint_from_image.hand.geometry import rect_from_landmarks
from fingerprint_from_image.models import ModelUnavailableError
from fingerprint_from_image.recognition import FingerprintRecognizer


def test_grayscale_input_is_rejected_with_actionable_message() -> None:
    with pytest.raises(ValueError, match="colour image"):
        _color_image(np.zeros((32, 32), dtype=np.uint8))
    assert _color_image(np.zeros((4, 4, 3), dtype=np.uint8)).mode == "RGB"


class _NoHand:
    def __call__(self, _rgb: np.ndarray) -> list:
        raise HandNotFoundError("No hand detected in the image")


class _Embedder:
    model_version = "unit"
    dim = 4

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        return np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32), (len(images), 1))


def test_recognizer_maps_missing_hand_to_matching_error(tmp_path: Path) -> None:
    with FingerprintRecognizer(tmp_path / "g.db", _Embedder(), detector=_NoHand()) as rec:
        with pytest.raises(FingerprintMatchingError, match="No hand"):
            rec.crops(np.zeros((8, 8, 3), dtype=np.uint8), kind="hand")


def test_roi_direction_uses_mediapipe_knuckle_weights() -> None:
    points = np.zeros((21, 3))
    points[0, :2] = (100.0, 200.0)                                # wrist
    points[5, :2], points[9, :2], points[13, :2] = (60, 100), (100, 90), (180, 120)
    points[:, :2] += np.linspace(0, 1, 21)[:, None]              # avoid exact degeneracy
    expected_middle = ((points[5, :2] + points[13, :2]) / 2 + points[9, :2]) / 2
    rect = rect_from_landmarks(points)
    vector = expected_middle - points[0, :2]
    # MediaPipe: rotation = pi/2 - atan2(-(y_mid - y_wrist), x_mid - x_wrist), normalised.
    expected = np.pi / 2 - np.arctan2(-vector[1], vector[0])
    expected = (expected + np.pi) % (2 * np.pi) - np.pi
    assert rect.rotation == pytest.approx(expected, abs=1e-6)


def test_degenerate_landmarks_do_not_produce_zero_width_boxes() -> None:
    boxes = finger_boxes(np.zeros((21, 3)), CropConfig())
    assert all(box.finger_width >= 1.0 for box in boxes)


def test_detector_config_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="palm_score_threshold"):
        DetectorConfig(palm_score_threshold=1.5)


def test_missing_explicit_model_dir_is_reported(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    with pytest.raises(ModelUnavailableError, match=str(tmp_path.name)):
        HandFingerDetector(tmp_path / "does-not-exist")
