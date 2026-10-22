"""End-to-end test of the ONNX hand models on real photographs.

Skipped unless onnxruntime is installed and the converted models are in the
model cache (``$FINGERPRINT_FROM_IMAGE_MODELS``); see tools/hand_models/README.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fingerprint_from_image.hand import (
    DetectorConfig,
    HandFingerDetector,
    HandNotFoundError,
    crop_mosaic,
    draw_hands,
)
from fingerprint_from_image.hand.topology import TIP_INDICES
from fingerprint_from_image.io import load_image
from fingerprint_from_image.models import ModelUnavailableError

pytest.importorskip("onnxruntime")
DATA = Path(__file__).parent / "data"
FINGERS = ("thumb", "index", "middle", "ring", "little")


@pytest.fixture(scope="module")
def detector() -> HandFingerDetector:
    try:
        return HandFingerDetector()
    except ModelUnavailableError as exc:
        pytest.skip(f"hand models not installed: {exc}")


def _inside(point: np.ndarray, polygon) -> bool:
    poly = np.asarray(polygon)
    signs = []
    for i in range(4):
        edge, rel = poly[(i + 1) % 4] - poly[i], point - poly[i]
        signs.append(edge[0] * rel[1] - edge[1] * rel[0])
    return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)


def _check_hand(rgb: np.ndarray, detector: HandFingerDetector, side: str) -> list:
    hands = detector.detect_hands(rgb)
    assert len(hands) == 1
    hand = hands[0]
    assert hand.side == side and hand.presence > 0.9
    crops = detector.finger_crops(rgb, hand)
    assert [c.label for c in crops] == [f"{side}_{f}" for f in FINGERS]
    height, width = rgb.shape[:2]
    for crop, tip in zip(crops, TIP_INDICES, strict=True):
        assert _inside(hand.landmarks[tip, :2], crop.polygon)
        poly = np.asarray(crop.polygon)
        assert np.all(poly > -0.1 * max(height, width))
        assert crop.image.ndim == 2 and crop.image.dtype == np.uint8
        assert min(crop.image.shape) >= 30
        assert 0.0 <= crop.quality <= 1.0
    tips_y = {c.label.split("_")[1]: np.mean(np.asarray(c.polygon)[:2, 1]) for c in crops}
    assert tips_y["middle"] < tips_y["little"] and tips_y["middle"] < tips_y["thumb"]
    return hands


def test_open_left_palm(detector: HandFingerDetector) -> None:
    rgb = load_image(DATA / "hand_palm_open.jpg", mode="RGB")
    (hand,) = _check_hand(rgb, detector, "left")
    assert hand.palm_facing is True


def test_back_of_left_hand(detector: HandFingerDetector) -> None:
    rgb = load_image(DATA / "hand_dorsal_left.jpg", mode="RGB")
    (hand,) = _check_hand(rgb, detector, "left")
    assert hand.palm_facing is False
    crops = detector(rgb)
    assert len(crops) == 5 and max(c.quality for c in crops) < 0.2  # nails, low resolution
    strict = HandFingerDetector(config=DetectorConfig(require_palm_side=True))
    with pytest.raises(HandNotFoundError):
        strict(rgb)


def test_right_palm_and_mirrored_flag(detector: HandFingerDetector) -> None:
    rgb = load_image(DATA / "hands_two_palms.jpg", mode="RGB")
    hands = detector.detect_hands(rgb)  # the blurred left-hand palm is not detected
    assert [h.side for h in hands] == ["right"] and hands[0].palm_facing is True
    mirrored = HandFingerDetector(config=DetectorConfig(mirrored=True))
    assert mirrored.detect_hands(rgb)[0].side == "left"


def test_two_hands_from_flipped_composite(detector: HandFingerDetector) -> None:
    rgb = load_image(DATA / "hand_palm_open.jpg", mode="RGB")
    pair = np.concatenate([rgb, rgb[:, ::-1]], axis=1)  # left palm | mirror = right palm
    crops = detector(pair, source="pair")
    assert len(crops) == 10 and all(c.source == "pair" for c in crops)
    labels = {c.label for c in crops}
    assert labels == {f"{s}_{f}" for s in ("left", "right") for f in FINGERS}
    for crop in crops:
        centre_x = np.mean(np.asarray(crop.polygon)[:, 0])
        assert (centre_x < rgb.shape[1]) == crop.label.startswith("left")
    single = HandFingerDetector(config=DetectorConfig(max_hands=1))
    assert len(single(pair)) == 5
    hands = detector.detect_hands(pair)
    assert draw_hands(pair, hands, crops).size == (pair.shape[1], pair.shape[0])
    assert crop_mosaic(crops).height > 256


def test_no_hand_raises(detector: HandFingerDetector) -> None:
    with pytest.raises(HandNotFoundError):
        detector(np.full((240, 320, 3), 30, dtype=np.uint8))
