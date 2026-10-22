"""Letterboxing, rotated regions of interest and affine warps (numpy + Pillow).

Coordinate convention: *continuous* pixel coordinates, ``x`` to the right and
``y`` down, with the centre of pixel ``(i, j)`` at ``(j + 0.5, i + 0.5)``. This
matches Pillow's ``Image.transform`` and MediaPipe's normalised coordinates
multiplied by the image size.

A :class:`RotatedRect` follows MediaPipe's ``NormalizedRect`` semantics in
pixels: its local x axis is ``(cos r, sin r)`` and its local "down" axis is
``(-sin r, cos r)``, so ``rotation = 0`` is an upright, axis-aligned square.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = [
    "Letterbox",
    "RotatedRect",
    "letterbox",
    "normalize_radians",
    "rect_from_landmarks",
    "rect_from_palm",
    "rect_to_image_affine",
    "warp_rect",
]


@dataclass(frozen=True)
class Letterbox:
    """Aspect-preserving resize of a ``width x height`` image into a square tensor."""

    size: int
    scale_x: float  # tensor pixels per source pixel
    scale_y: float
    offset_x: float
    offset_y: float

    def to_source(self, normalized_xy: np.ndarray) -> np.ndarray:
        """Map normalised tensor coordinates ``(..., 2)`` to source pixels."""

        points = np.asarray(normalized_xy, dtype=np.float64) * self.size
        offset = np.asarray([self.offset_x, self.offset_y])
        return (points - offset) / np.asarray([self.scale_x, self.scale_y])

    def to_tensor(self, source_xy: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_source` (normalised tensor coordinates)."""

        points = np.asarray(source_xy, dtype=np.float64) * np.asarray([self.scale_x, self.scale_y])
        return (points + np.asarray([self.offset_x, self.offset_y])) / self.size


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, Letterbox]:
    """Resize ``image`` to fit ``size x size`` (bilinear, antialiased), zero-padded.

    Returns the ``(size, size, 3)`` float32 tensor in ``[0, 1]`` and the mapping.
    """

    width, height = image.size
    scale = size / max(width, height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    resized = image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    off_x, off_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    tensor = np.asarray(canvas, dtype=np.float32) / 255.0
    # Store the realised per-axis scales (after rounding) so the inverse is exact.
    return tensor, Letterbox(size, new_w / width, new_h / height, off_x, off_y)


def normalize_radians(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


@dataclass(frozen=True)
class RotatedRect:
    """Rotated rectangle in source pixels (MediaPipe ``NormalizedRect`` semantics)."""

    cx: float
    cy: float
    width: float
    height: float
    rotation: float = 0.0

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the unit local x axis and the unit local "down" axis."""

        c, s = math.cos(self.rotation), math.sin(self.rotation)
        return np.asarray([c, s]), np.asarray([-s, c])

    def corners(self) -> np.ndarray:
        """``(4, 2)`` corners clockwise from the local top-left."""

        ex, ey = self.axes()
        center = np.asarray([self.cx, self.cy])
        hx, hy = ex * self.width / 2, ey * self.height / 2
        return np.stack([center - hx - hy, center + hx - hy, center + hx + hy, center - hx + hy])

    def to_image(self, normalized_xy: np.ndarray) -> np.ndarray:
        """Map ``(..., 2)`` rect-normalised coordinates (``[0, 1]``) to source pixels."""

        pts = np.asarray(normalized_xy, dtype=np.float64) - 0.5
        ex, ey = self.axes()
        return (np.asarray([self.cx, self.cy]) + pts[..., :1] * self.width * ex
                + pts[..., 1:2] * self.height * ey)

    def to_normalized(self, image_xy: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`to_image`."""

        rel = np.asarray(image_xy, dtype=np.float64) - np.asarray([self.cx, self.cy])
        ex, ey = self.axes()
        return np.stack([rel @ ex / self.width, rel @ ey / self.height], axis=-1) + 0.5

    def transformed(
        self, *, scale_x: float = 1.0, scale_y: float = 1.0, shift_x: float = 0.0,
        shift_y: float = 0.0, square_long: bool = True,
    ) -> RotatedRect:
        """MediaPipe ``RectTransformationCalculator`` in pixel units."""

        ex, ey = self.axes()
        center = (np.asarray([self.cx, self.cy]) + ex * self.width * shift_x
                  + ey * self.height * shift_y)
        width, height = self.width, self.height
        if square_long:
            width = height = max(width, height)
        return RotatedRect(float(center[0]), float(center[1]), width * scale_x,
                           height * scale_y, self.rotation)


def _rotation(start: np.ndarray, end: np.ndarray, target: float = math.pi / 2) -> float:
    return normalize_radians(target - math.atan2(-(end[1] - start[1]), end[0] - start[0]))


def rect_from_palm(
    box: np.ndarray, keypoints: np.ndarray, *, scale: float = 2.6, shift_y: float = -0.5,
) -> RotatedRect:
    """Hand ROI from a palm box and its 7 keypoints, both in source pixels.

    Rotation aligns wrist (keypoint 0) -> middle-finger MCP (keypoint 2) with
    "up"; the box is shifted towards the fingers and enlarged to cover the whole
    hand, as in MediaPipe's hand detector graph.
    """

    xmin, ymin, xmax, ymax = (float(v) for v in box)
    rect = RotatedRect((xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin,
                       _rotation(keypoints[0], keypoints[2]))
    return rect.transformed(scale_x=scale, scale_y=scale, shift_y=shift_y)


_ROI_LANDMARKS = (0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)


def rect_from_landmarks(
    landmarks: np.ndarray, *, scale: float = 2.0, shift_y: float = -0.1,
) -> RotatedRect:
    """Hand ROI recomputed from 21 landmarks (``HandLandmarksToRectCalculator``)."""

    pts = np.asarray(landmarks, dtype=np.float64)[:, :2]
    middle = 0.25 * (pts[5] + pts[13]) / 2 + 0.75 * pts[9]
    rotation = _rotation(pts[0], middle)
    subset = pts[list(_ROI_LANDMARKS)]
    c, s = math.cos(rotation), math.sin(rotation)
    # Express points in the rotated frame, take the axis-aligned box there.
    local = subset @ np.asarray([[c, -s], [s, c]])
    lo, hi = local.min(axis=0), local.max(axis=0)
    center_local = (lo + hi) / 2
    center = center_local @ np.asarray([[c, s], [-s, c]])
    rect = RotatedRect(float(center[0]), float(center[1]), float(hi[0] - lo[0]),
                       float(hi[1] - lo[1]), rotation)
    return rect.transformed(scale_x=scale, scale_y=scale, shift_y=shift_y)


def rect_to_image_affine(rect: RotatedRect, out_w: int, out_h: int) -> np.ndarray:
    """``(2, 3)`` matrix mapping output pixel coords of a warped ``rect`` to the source."""

    ex, ey = rect.axes()
    sx, sy = rect.width / out_w, rect.height / out_h
    origin = np.asarray([rect.cx, rect.cy]) - ex * rect.width / 2 - ey * rect.height / 2
    return np.asarray([[ex[0] * sx, ey[0] * sy, origin[0]],
                       [ex[1] * sx, ey[1] * sy, origin[1]]])


def warp_rect(
    image: Image.Image,
    rect: RotatedRect,
    out_w: int,
    out_h: int,
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
    fill: float | tuple[int, ...] = 0,
) -> Image.Image:
    """Sample ``rect`` from ``image`` into an ``out_w x out_h`` upright image.

    When the rect is much larger than the output, the source is first reduced
    by an integer box filter so that bilinear/bicubic sampling does not alias.
    """

    factor = int(min(rect.width / out_w, rect.height / out_h) // 2)
    source = image
    if factor >= 2:
        source = image.reduce(factor)
        rect = RotatedRect(rect.cx / factor, rect.cy / factor, rect.width / factor,
                           rect.height / factor, rect.rotation)
    matrix = rect_to_image_affine(rect, out_w, out_h)
    return source.transform((out_w, out_h), Image.Transform.AFFINE,
                            data=tuple(matrix.reshape(-1)), resample=resample, fillcolor=fill)
