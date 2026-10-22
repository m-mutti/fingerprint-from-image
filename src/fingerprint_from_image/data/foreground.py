"""Ridge-foreground cropping and masked high-pass descriptors for duplicate search.

Sensor artefacts (platen frames, fixed squares, large white canvases) make
whole-image descriptors collide across different fingers.  These helpers
restrict the comparison to the ridge area: blocks with strong high-pass
energy, eroded once so thin straight edges (frames) disappear.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

__all__ = ["crop_foreground", "foreground_blocks", "masked_descriptor"]

_WORK_SIDE = 256
_BLOCK = 8


def _shift_and(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return out


def _shift_or(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= padded[1 + dy:1 + dy + mask.shape[0], 1 + dx:1 + dx + mask.shape[1]]
    return out


def foreground_blocks(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Boolean block mask (8x8 blocks of a <=256 px copy) and the copy's scale."""

    height, width = gray.shape
    scale = min(1.0, _WORK_SIDE / max(height, width))
    image = Image.fromarray(gray)
    if scale < 1.0:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                             Image.Resampling.BILINEAR)
    small = np.asarray(image, dtype=np.float32)
    low = np.asarray(image.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    high = small - low
    rows, cols = high.shape[0] // _BLOCK, high.shape[1] // _BLOCK
    if rows < 3 or cols < 3:
        return np.ones((max(rows, 1), max(cols, 1)), dtype=bool), scale
    blocks = high[:rows * _BLOCK, :cols * _BLOCK].reshape(rows, _BLOCK, cols, _BLOCK)
    energy = blocks.std(axis=(1, 3))
    threshold = max(2.0, 0.35 * float(np.percentile(energy, 95)))
    mask = _shift_or(_shift_and(energy > threshold))  # opening removes 1-block-thin edges
    if mask.mean() < 0.03:
        mask = energy > threshold
    return mask, scale


def crop_foreground(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop ``gray`` to the ridge bounding box; return (crop, float mask of the crop)."""

    mask, scale = foreground_blocks(gray)
    if not mask.any():
        return gray, np.ones(gray.shape, dtype=np.float32)
    ys, xs = np.nonzero(mask)
    step = _BLOCK / scale
    top, bottom = int(ys.min() * step), int(min(gray.shape[0], (ys.max() + 1) * step))
    left, right = int(xs.min() * step), int(min(gray.shape[1], (xs.max() + 1) * step))
    crop = gray[top:bottom, left:right]
    block_mask = Image.fromarray((mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255)
                                 .astype(np.uint8))
    full = np.asarray(block_mask.resize((crop.shape[1], crop.shape[0]), Image.Resampling.NEAREST),
                      dtype=np.float32) / 255.0
    return crop, full


def masked_descriptor(gray: np.ndarray, side: int = 64, blur: float = 2.0, *,
                      crop: bool = True) -> np.ndarray:
    """Unit-norm high-pass descriptor with background zeroed.

    ``crop=True`` uses the ridge bounding box (robust to canvas size / padding);
    ``crop=False`` keeps the full frame (robust for pixel-aligned copies, where
    two slightly different bounding boxes would misalign the ridges).
    """

    if crop:
        crop_img, mask = crop_foreground(gray)
    else:
        blocks, _ = foreground_blocks(gray)
        crop_img = gray
        mask = np.asarray(Image.fromarray((blocks * 255).astype(np.uint8)).resize(
            (gray.shape[1], gray.shape[0]), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    image = Image.fromarray(crop_img).resize((side, side), Image.Resampling.BILINEAR)
    small = np.asarray(image, dtype=np.float32)
    high = small - np.asarray(image.filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32)
    weight = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize(
        (side, side), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    high = high * weight
    high -= high.mean()
    norm = float(np.linalg.norm(high))
    return (high.ravel() / norm if norm > 1e-6 else high.ravel()).astype(np.float32)
