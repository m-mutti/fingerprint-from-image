"""Oriented distal-phalanx boxes from hand landmarks and upright crops.

For each finger the box axis runs from the DIP joint (IP for the thumb)
through the tip landmark. Its length is the DIP->tip segment plus margins (the
tip landmark sits slightly inside the fingertip outline), and its width comes
from the spacing of neighbouring MCP knuckles (a stable proxy for finger
width, unaffected by finger spread) clamped against the segment length.
Crops are sampled from the *original* image at native resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .geometry import RotatedRect, rect_to_image_affine, warp_rect
from .topology import DIP_INDICES, MCP_INDICES, PIP_INDICES, TIP_INDICES

__all__ = ["CropConfig", "FingerBox", "finger_boxes", "sample_crop"]


@dataclass(frozen=True)
class CropConfig:
    """Geometry of the distal-phalanx crop (all ratios are dimensionless).

    Attributes:
        tip_margin: extension beyond the tip landmark, in DIP->tip lengths.
        base_margin: extension below the DIP joint, in DIP->tip lengths.
        side_margin: extra width on each side, as a fraction of finger width.
        width_per_spacing: distal-phalanx width / neighbouring-MCP spacing.
        thumb_width_per_spacing: same for the thumb (relative to index-middle).
        min_width_per_length, max_width_per_length: clamp of width / segment
            length, guarding against odd spacings (e.g. crossed fingers).
        min_length_per_width: keeps a usable box when the finger points at the
            camera and the 2-D segment is foreshortened.
        finger_width_mm: assumed physical distal width (for ridge-period priors).
        ridge_period_mm: assumed adult ridge period (~0.4-0.5 mm).
        output_width: resample crops so the finger width spans this many pixels
            (``None`` keeps the source resolution).
        gray_weights: RGB weights for the grayscale conversion.
    """

    tip_margin: float = 0.35
    base_margin: float = 0.05
    side_margin: float = 0.08
    width_per_spacing: float = 0.85
    thumb_width_per_spacing: float = 1.0
    min_width_per_length: float = 0.55
    max_width_per_length: float = 1.6
    min_length_per_width: float = 0.9
    finger_width_mm: float = 16.0
    ridge_period_mm: float = 0.45
    output_width: int | None = None
    gray_weights: tuple[float, float, float] = (0.299, 0.587, 0.114)

    def __post_init__(self) -> None:
        if min(self.tip_margin, self.base_margin, self.side_margin) < 0:
            raise ValueError("margins must be non-negative")
        if self.output_width is not None and self.output_width < 8:
            raise ValueError("output_width must be at least 8 pixels")


@dataclass(frozen=True)
class FingerBox:
    """Oriented distal-phalanx box in source pixels, tip end on top."""

    finger: int  # 0 thumb .. 4 little
    rect: RotatedRect  # rotation makes local "up" point from DIP to tip
    finger_width: float  # estimated distal width without margins, pixels

    @property
    def polygon(self) -> tuple[tuple[float, float], ...]:
        """Corners clockwise from the upright crop's top-left, in source pixels."""

        return tuple((float(x), float(y)) for x, y in self.rect.corners())

    def ridge_period_px(self, config: CropConfig) -> float:
        """Expected ridge period in source pixels under the size priors."""

        return self.finger_width * config.ridge_period_mm / config.finger_width_mm


def _spacing(points: np.ndarray, finger: int) -> float:
    mcp = points[list(MCP_INDICES)]
    # Knuckle gaps: index-middle, middle-ring, ring-little.
    gaps = np.linalg.norm(np.diff(mcp[1:], axis=0), axis=1)
    if finger == 0:
        return float(gaps[0])
    neighbours = [gaps[k] for k in (finger - 2, finger - 1) if 0 <= k < len(gaps)]
    return float(np.mean(neighbours))


def finger_boxes(landmarks: np.ndarray, config: CropConfig | None = None) -> list[FingerBox]:
    """Compute the five distal-phalanx boxes from ``(21, 2|3)`` source-pixel landmarks."""

    config = config or CropConfig()
    points = np.asarray(landmarks, dtype=np.float64)[:, :2]
    if points.shape != (21, 2):
        raise ValueError(f"expected 21 landmarks, got {points.shape}")
    boxes: list[FingerBox] = []
    for finger in range(5):
        tip, dip = points[TIP_INDICES[finger]], points[DIP_INDICES[finger]]
        axis = tip - dip
        length = float(np.linalg.norm(axis))
        if length < 1e-6:  # degenerate: fall back to the PIP->DIP direction
            axis = dip - points[PIP_INDICES[finger]]
            length = 0.0
        norm = float(np.linalg.norm(axis))
        direction = axis / norm if norm > 1e-9 else np.asarray([0.0, -1.0])
        ratio = config.thumb_width_per_spacing if finger == 0 else config.width_per_spacing
        width = ratio * _spacing(points, finger)
        if length > 0:
            width = float(np.clip(width, config.min_width_per_length * length,
                                  config.max_width_per_length * length))
        width = max(width, 1.0)  # degenerate landmarks must not yield a zero-width box
        length = max(length, config.min_length_per_width * width)
        top = tip + direction * config.tip_margin * length
        bottom = dip - direction * config.base_margin * length
        box_len = float(np.linalg.norm(top - bottom))
        center = (top + bottom) / 2
        # Local "down" axis (-sin r, cos r) must equal -direction.
        rotation = math.atan2(direction[0], -direction[1])
        rect = RotatedRect(float(center[0]), float(center[1]),
                           width * (1 + 2 * config.side_margin), box_len, rotation)
        boxes.append(FingerBox(finger, rect, width))
    return boxes


def sample_crop(image: Image.Image, box: FingerBox, config: CropConfig | None = None
                ) -> tuple[np.ndarray, float, float]:
    """Sample an upright grayscale crop from an ``"RGB"``, ``"L"`` or ``"F"`` image.

    Only the crop is resampled (bicubic) and converted to gray with
    ``config.gray_weights``, so the cost does not grow with the photo size.

    Returns ``(crop float64, valid_fraction, scale)`` where ``valid_fraction``
    is the share of the crop inside the source image and ``scale`` is output
    pixels per source pixel.
    """

    config = config or CropConfig()
    rect = box.rect
    scale = 1.0 if config.output_width is None else config.output_width / box.finger_width
    out_w = max(8, round(rect.width * scale))
    out_h = max(8, round(rect.height * scale))
    crop = warp_rect(image, rect, out_w, out_h, resample=Image.Resampling.BICUBIC)
    values = np.asarray(crop, dtype=np.float64)
    if values.ndim == 3:
        values = values[..., :3] @ np.asarray(config.gray_weights, dtype=np.float64)
    # Coverage: map every output pixel centre back to the source analytically.
    matrix = rect_to_image_affine(rect, out_w, out_h)
    ys, xs = np.mgrid[:out_h, :out_w] + 0.5
    src_x = matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]
    src_y = matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]
    valid = (src_x >= 0) & (src_x < image.width) & (src_y >= 0) & (src_y < image.height)
    fraction = float(valid.mean())
    if fraction < 1.0:
        values = np.where(valid, values, np.median(values[valid]) if valid.any() else 0.0)
    return np.clip(values, 0, 255), fraction, out_w / rect.width
