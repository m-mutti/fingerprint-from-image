from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from fingerprint_from_image.hand.geometry import (
    RotatedRect,
    letterbox,
    normalize_radians,
    rect_from_landmarks,
    rect_from_palm,
    rect_to_image_affine,
    warp_rect,
)


def test_letterbox_round_trip_and_padding() -> None:
    image = Image.new("RGB", (300, 150), (255, 255, 255))
    tensor, mapping = letterbox(image, 192)
    assert tensor.shape == (192, 192, 3) and tensor.dtype == np.float32
    assert tensor.max() == pytest.approx(1.0)
    assert tensor[0, :, 0].max() == 0.0 and tensor[96, :, 0].min() == pytest.approx(1.0)
    points = np.array([[0.0, 0.0], [300.0, 150.0], [123.4, 56.7]])
    np.testing.assert_allclose(mapping.to_source(mapping.to_tensor(points)), points, atol=1e-9)
    # Source corners land on the letterbox content edges.
    np.testing.assert_allclose(mapping.to_tensor(points[:2]) * 192, [[0, 48], [192, 144]])


def test_normalize_radians() -> None:
    assert normalize_radians(3 * math.pi) == pytest.approx(-math.pi)
    assert normalize_radians(-0.5) == pytest.approx(-0.5)


@pytest.mark.parametrize("rotation", [0.0, 0.7, -2.1, math.pi])
def test_rotated_rect_round_trip_and_corners(rotation: float) -> None:
    rect = RotatedRect(40.0, 30.0, 20.0, 10.0, rotation)
    pts = np.random.default_rng(0).uniform(-0.5, 1.5, (25, 2))
    np.testing.assert_allclose(rect.to_normalized(rect.to_image(pts)), pts, atol=1e-12)
    corners = rect.corners()
    np.testing.assert_allclose(corners, rect.to_image([[0, 0], [1, 0], [1, 1], [0, 1]]))
    assert np.linalg.norm(corners[1] - corners[0]) == pytest.approx(20.0)
    assert np.linalg.norm(corners[3] - corners[0]) == pytest.approx(10.0)


def test_rect_transformation_shift_follows_rotation() -> None:
    rect = RotatedRect(0.0, 0.0, 10.0, 4.0, math.pi / 2)  # local "down" points to -x
    out = rect.transformed(scale_x=2, scale_y=2, shift_y=-0.5)
    assert (out.cx, out.cy) == pytest.approx((2.0, 0.0))  # moved half a height "up" = +x
    assert out.width == out.height == pytest.approx(20.0)  # square_long, then scaled


@pytest.mark.parametrize("rotation", [0.0, 0.4, -1.3, 2.8])
def test_warp_rect_affine_round_trip(rotation: float) -> None:
    """A bright source pixel must appear where the analytic affine predicts."""

    source = np.zeros((200, 240), dtype=np.uint8)
    source[90, 130] = 255
    rect = RotatedRect(125.0, 95.0, 60.0, 60.0, rotation)
    out = np.asarray(warp_rect(Image.fromarray(source), rect, 60, 60,
                               resample=Image.Resampling.NEAREST))
    hits = np.argwhere(out == 255)
    assert len(hits) >= 1
    matrix = rect_to_image_affine(rect, 60, 60)
    for row, col in hits:
        x, y = matrix @ np.array([col + 0.5, row + 0.5, 1.0])
        assert (int(math.floor(x)), int(math.floor(y))) == (130, 90)
    centre = rect.to_normalized(np.array([130.5, 90.5])) * 60
    assert np.min(np.abs(hits[:, ::-1] + 0.5 - centre).max(axis=1)) <= 0.75


def test_warp_rect_reduces_large_rects_without_shifting() -> None:
    source = np.zeros((800, 800), dtype=np.uint8)
    source[380:420, 580:620] = 255  # blob centred at (600, 400)
    rect = RotatedRect(400.0, 400.0, 800.0, 800.0, 0.0)
    out = np.asarray(warp_rect(Image.fromarray(source), rect, 80, 80), dtype=float)
    ys, xs = np.nonzero(out > 64)
    assert xs.mean() + 0.5 == pytest.approx(60.0, abs=0.6)
    assert ys.mean() + 0.5 == pytest.approx(40.0, abs=0.6)


def test_rect_from_palm_orientation() -> None:
    box = np.array([100.0, 100.0, 200.0, 200.0])
    upright = np.zeros((7, 2))
    upright[0], upright[2] = (150, 200), (150, 120)  # wrist below middle MCP
    rect = rect_from_palm(box, upright)
    assert rect.rotation == pytest.approx(0.0)
    assert (rect.cx, rect.cy, rect.width) == pytest.approx((150.0, 100.0, 260.0))
    sideways = np.zeros((7, 2))
    sideways[0], sideways[2] = (100, 150), (180, 150)  # fingers point right
    rect = rect_from_palm(box, sideways)
    assert rect.rotation == pytest.approx(math.pi / 2)
    assert (rect.cx, rect.cy) == pytest.approx((200.0, 150.0))  # shifted toward the fingers


@pytest.mark.parametrize("angle", [0.0, 0.5, -1.2, 3.0])
def test_rect_from_landmarks_recovers_rotation(angle: float) -> None:
    rng = np.random.default_rng(1)
    local = np.zeros((21, 2))
    local[0] = (0, 0)
    local[[5, 9, 13]] = [(-20, -100), (0, -100), (20, -100)]
    others = [i for i in range(21) if i not in (0, 5, 9, 13)]
    local[others] = rng.uniform([-40, -180], [40, -10], (len(others), 2))
    c, s = math.cos(angle), math.sin(angle)
    points = local @ np.array([[c, s], [-s, c]]) + (300, 300)
    rect = rect_from_landmarks(points)
    assert normalize_radians(rect.rotation - angle) == pytest.approx(0.0, abs=1e-9)
    assert rect.width == rect.height
    inside = rect.to_normalized(points)
    assert np.all((inside > 0) & (inside < 1))
