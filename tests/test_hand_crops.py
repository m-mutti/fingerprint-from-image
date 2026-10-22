from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from fingerprint_from_image.hand.crops import CropConfig, finger_boxes, sample_crop
from fingerprint_from_image.hand.geometry import normalize_radians
from fingerprint_from_image.hand.topology import DIP_INDICES, TIP_INDICES

# Upright right palm facing the camera (thumb on image right), wrist at origin.
CANONICAL = np.array([
    (0, 0),
    (25, -15), (45, -35), (60, -55), (72, -72),
    (30, -90), (33, -125), (35, -148), (36, -168),
    (8, -95), (8, -135), (8, -160), (8, -182),
    (-14, -90), (-17, -125), (-19, -148), (-20, -167),
    (-34, -80), (-40, -105), (-43, -122), (-45, -137),
], dtype=float)


def make_hand(angle: float = 0.0, scale: float = 1.0, offset=(400.0, 400.0)) -> np.ndarray:
    """Canonical landmarks rotated so that "up" maps to ``(sin a, -cos a)``."""

    c, s = math.cos(angle), math.sin(angle)
    return CANONICAL @ np.array([[c, s], [-s, c]]) * scale + np.asarray(offset)


def test_upright_boxes_cover_distal_phalanges() -> None:
    boxes = finger_boxes(make_hand())
    assert [b.finger for b in boxes] == [0, 1, 2, 3, 4]
    middle = boxes[2]
    assert middle.rect.rotation == pytest.approx(0.0)
    tl, tr, br, bl = np.asarray(middle.polygon)
    assert tl[1] == pytest.approx(tr[1]) and tl[0] < tr[0]  # tip edge on top, left->right
    assert bl[1] > tl[1]
    # Length: DIP->tip (22 px) plus 35 % beyond the tip and 5 % below the DIP.
    assert middle.rect.height == pytest.approx(22 * 1.40)
    # Width from MCP spacing: mean(|5-9|, |9-13|) * 0.85, plus 8 % margin per side.
    spacing = np.mean([np.hypot(22, 5), np.hypot(22, 5)])
    assert middle.finger_width == pytest.approx(0.85 * spacing)
    assert middle.rect.width == pytest.approx(middle.finger_width * 1.16)
    hand = make_hand()
    for box in boxes:
        uv = box.rect.to_normalized(hand[[TIP_INDICES[box.finger], DIP_INDICES[box.finger]]])
        assert np.all((uv > 0) & (uv < 1)), box.finger
        assert uv[0, 1] < uv[1, 1]  # tip above DIP in the upright crop


@pytest.mark.parametrize("angle", [0.3, -1.0, 2.5, math.pi])
def test_boxes_are_rotation_and_scale_equivariant(angle: float) -> None:
    reference = finger_boxes(make_hand())
    rotated = finger_boxes(make_hand(angle, scale=2.0, offset=(900.0, 700.0)))
    for ref, rot in zip(reference, rotated, strict=True):
        assert normalize_radians(rot.rect.rotation - ref.rect.rotation - angle) == \
            pytest.approx(0.0, abs=1e-9)
        assert rot.rect.width == pytest.approx(2 * ref.rect.width)
        assert rot.rect.height == pytest.approx(2 * ref.rect.height)


def test_width_clamps_and_foreshortened_finger() -> None:
    hand = make_hand()
    hand[12] = hand[11] + (0.0, -2.0)  # middle finger pointing at the camera
    box = finger_boxes(hand)[2]
    assert box.rect.height >= 0.9 * box.finger_width  # min_length_per_width keeps a usable box
    hand[12] = hand[11]  # fully degenerate: falls back to the PIP->DIP direction
    box = finger_boxes(hand)[2]
    assert box.rect.rotation == pytest.approx(0.0)
    with pytest.raises(ValueError):
        finger_boxes(hand[:20])
    with pytest.raises(ValueError):
        CropConfig(tip_margin=-0.1)


def test_sample_crop_is_upright_native_resolution_and_reports_coverage() -> None:
    angle = math.pi / 2  # fingers point to the image right
    hand = make_hand(angle, scale=3.0, offset=(150.0, 400.0))  # middle tip at x ~ 696
    box = finger_boxes(hand)[2]
    tip, dip = hand[12], hand[11]
    image = np.full((800, 800), 100, dtype=np.float32)
    ty, tx = int(tip[1]), int(tip[0])
    image[ty - 3:ty + 3, tx - 3:tx + 3] = 250  # bright marker at the tip
    gray = Image.fromarray(image)
    crop, coverage, scale = sample_crop(gray, box)
    assert coverage == pytest.approx(1.0) and scale == pytest.approx(1.0, rel=0.05)
    assert crop.shape == (round(box.rect.height), round(box.rect.width))
    rows = np.nonzero(crop.max(axis=1) > 200)[0]
    assert rows.mean() < crop.shape[0] / 2  # tip marker in the upper half
    assert box.rect.height == pytest.approx(np.linalg.norm(tip - dip) * 1.4)
    _, _, scale = sample_crop(gray, box, CropConfig(output_width=200))
    assert scale == pytest.approx(200 / box.finger_width, rel=0.02)
    edge = finger_boxes(make_hand(offset=(20.0, 175.0)))[2]  # tip box crosses the top edge
    _, coverage, _ = sample_crop(Image.fromarray(image[:200, :200]), edge)
    assert 0.0 < coverage < 1.0
