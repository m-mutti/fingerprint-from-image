"""Deterministic numpy/Pillow preprocessing shared by training and inference.

The embedder sees every fingerprint at the same *ridge scale*, whatever the
sensor resolution: the dominant ridge period is estimated from the image
spectrum and the image is resampled so that one ridge+valley spans
``TARGET_PERIOD`` pixels. A ``INPUT_SIZE`` square window is then cut around
the centroid of the ridge-bearing foreground and intensity-normalised with a
soft foreground mask (masked z-score, clipped to ``[-CLIP, CLIP]``, background
suppressed to zero).

Training code imports these exact functions to build its image cache and to
evaluate, so preprocessing at inference is bit-for-bit what was used in
evaluation. No OpenCV, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

__all__ = [
    "BLOCK",
    "CLIP",
    "INPUT_SIZE",
    "PREPROCESS_VERSION",
    "TARGET_PERIOD",
    "Prepared",
    "block_mask",
    "crop_around",
    "estimate_ridge_period",
    "foreground_center",
    "normalize_intensity",
    "prepare",
    "preprocess",
    "rescale_to_period",
    "rotate_input",
    "soft_mask",
    "trim_uniform_border",
]

PREPROCESS_VERSION = "pp-v3"  # bump on any change that alters model inputs
TARGET_PERIOD = 6.0  # pixels per ridge period after rescaling
INPUT_SIZE = 256  # network input side
BLOCK = 16  # foreground block size (≈ 2.7 ridge periods)
CLIP = 3.0  # z-score clip
MIN_PERIOD, MAX_PERIOD = 2.5, 25.0  # plausible ridge periods in source pixels
MIN_SCALE, MAX_SCALE = 0.2, 4.0
_ABS_STD_FLOOR = 4.0  # block std (0..255 units) below which a block is background
_REL_STD = 0.3  # block is foreground if std > _REL_STD * 90th percentile
_MAX_FFT_SIDE = 1024
_MIN_RIDGES = 16.0  # size prior: a print spans at least this many ridges on its long side


@dataclass(frozen=True)
class Prepared:
    """A rescaled grayscale image plus the measurements used to make it."""

    image: np.ndarray  # (H, W) float32 in [0, 255], ridge period == TARGET_PERIOD
    period: float  # estimated ridge period in the source image (pixels)
    scale: float  # applied resampling factor
    center: tuple[float, float]  # (row, col) of the foreground centroid in ``image``
    background: float  # background grey level used for padding


def _as_float_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3].astype(np.float32)
        array = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 8:
        raise ValueError(f"Expected a 2-D grayscale image of at least 8x8, got {array.shape}")
    return array.astype(np.float32)


def _welch_blocks(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Mean-removed, high-variance square blocks (50 % overlap) and their side."""

    height, width = image.shape
    side = max(16, min(32 if min(height, width) < 160 else 64, height, width))
    stride = side // 2
    blocks = np.stack([
        image[y : y + side, x : x + side]
        for y in range(0, height - side + 1, stride)
        for x in range(0, width - side + 1, stride)
    ])
    std = blocks.std(axis=(1, 2))
    keep = std > max(_ABS_STD_FLOOR, 0.5 * float(np.quantile(std, 0.9)))
    if keep.any():
        blocks = blocks[keep]
    return blocks - blocks.mean(axis=(1, 2), keepdims=True), side


def estimate_ridge_period(gray: np.ndarray) -> float:
    """Return the dominant ridge period (pixels) by Welch spectral averaging.

    High-variance blocks are Hann-windowed, zero-padded 2x, and their power
    spectra normalised to unit energy before averaging, so every ridge block
    votes equally and large blotches (obliterations, smudges) cannot dominate.
    Energy is accumulated in 64 log-spaced radial bands between
    ``1/max_period`` and ``1/MIN_PERIOD``; the peak band is refined by an
    energy-weighted mean in log-frequency. Prior: a print spans at least
    ``_MIN_RIDGES`` (16) ridges along its longer side, so
    ``max_period = min(MAX_PERIOD, side / 16)``.
    """

    image = _as_float_gray(gray)
    step = int(np.ceil(max(image.shape) / _MAX_FFT_SIDE))
    if step > 1:  # very large inputs: subsample (period is rescaled below)
        image = image[::step, ::step]
    min_period = max(MIN_PERIOD / step, 2.05)
    max_period = max(min(MAX_PERIOD / step, max(image.shape) / _MIN_RIDGES), 1.5 * min_period)
    blocks, side = _welch_blocks(image)
    window = np.outer(np.hanning(side), np.hanning(side)).astype(np.float32)
    n = 2 * side
    power = np.abs(np.fft.rfft2(blocks * window, s=(n, n))) ** 2
    power /= power.sum(axis=(1, 2), keepdims=True) + 1e-12
    power = power.mean(axis=0)
    radius = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.rfftfreq(n)[None, :] ** 2)
    low, high = 1.0 / max_period, 1.0 / min_period
    valid = (radius >= low) & (radius < high)
    edges = np.geomspace(low, high, 65)
    index = np.clip(np.digitize(radius[valid], edges) - 1, 0, 63)
    energy = np.bincount(index, weights=power[valid], minlength=64)[:64].astype(np.float64)
    profile: np.ndarray = np.convolve(np.pad(energy, 1, mode="edge"), [0.25, 0.5, 0.25],
                                      mode="valid")
    if not np.isfinite(profile).all() or profile.max() <= 0:
        return TARGET_PERIOD * step
    peak = int(np.argmax(profile))
    lo, hi = max(peak - 2, 0), min(peak + 3, 64)
    centres = np.log(np.sqrt(edges[:-1] * edges[1:]))[lo:hi]
    weights = profile[lo:hi]
    frequency = float(np.exp(np.sum(centres * weights) / np.sum(weights)))
    return float(np.clip(step / frequency, MIN_PERIOD, MAX_PERIOD))


def trim_uniform_border(gray: np.ndarray, tolerance: float = 2.0) -> np.ndarray:
    """Strip constant sensor-frame rows/columns (bars) at the image border.

    Removes outer rows/columns whose standard deviation is below ``tolerance``
    grey levels, at most a quarter of the image per side.
    """

    image = np.asarray(gray)
    height, width = image.shape
    top, bottom, left, right = 0, height, 0, width
    while top < height // 4 and float(image[top].std()) < tolerance:
        top += 1
    while bottom > height - height // 4 and float(image[bottom - 1].std()) < tolerance:
        bottom -= 1
    while left < width // 4 and float(image[top:bottom, left].std()) < tolerance:
        left += 1
    while right > width - width // 4 and float(image[top:bottom, right - 1].std()) < tolerance:
        right -= 1
    return image[top:bottom, left:right]


def rescale_to_period(gray: np.ndarray, period: float) -> tuple[np.ndarray, float]:
    """Resample so the ridge period becomes ``TARGET_PERIOD``; returns (image, scale)."""

    image = _as_float_gray(gray)
    scale = float(np.clip(TARGET_PERIOD / period, MIN_SCALE, MAX_SCALE))
    height, width = image.shape
    size = (max(8, round(width * scale)), max(8, round(height * scale)))
    resized = Image.fromarray(image).resize(size, Image.Resampling.BILINEAR)
    return np.clip(np.asarray(resized, dtype=np.float32), 0.0, 255.0), scale


def block_mask(image: np.ndarray) -> np.ndarray:
    """Boolean foreground mask on the ``BLOCK`` grid (majority-filtered).

    ``image`` sides must be multiples of ``BLOCK``.
    """

    height, width = image.shape
    blocks = image.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK).astype(np.float64)
    std = blocks.std(axis=(1, 3))
    threshold = max(_ABS_STD_FLOOR, _REL_STD * float(np.quantile(std, 0.9)))
    raw = (std > threshold).astype(np.int32)
    padded = np.pad(raw, 1, mode="edge")
    votes = np.sum([
        padded[dy : dy + raw.shape[0], dx : dx + raw.shape[1]] for dy in range(3) for dx in range(3)
    ], axis=0)
    return np.asarray(votes >= 5)


def _pad_to_blocks(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    pad_h, pad_w = (-height) % BLOCK, (-width) % BLOCK
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge") if pad_h or pad_w else image


def foreground_center(image: np.ndarray) -> tuple[tuple[float, float], float]:
    """Return ((row, col) foreground centroid, background grey level)."""

    padded = _pad_to_blocks(image)
    mask = block_mask(padded)
    height, width = image.shape
    if not mask.any():
        return ((height - 1) / 2.0, (width - 1) / 2.0), float(np.quantile(image, 0.9))
    rows, cols = np.nonzero(mask)
    center = (float(rows.mean() * BLOCK + (BLOCK - 1) / 2.0),
              float(cols.mean() * BLOCK + (BLOCK - 1) / 2.0))
    center = (min(center[0], height - 1.0), min(center[1], width - 1.0))
    full = np.kron(mask, np.ones((BLOCK, BLOCK), dtype=bool))[:height, :width]
    background = image[~full]
    level = float(np.median(background)) if background.size else float(np.quantile(image, 0.99))
    return center, level


def crop_around(
    image: np.ndarray, center: tuple[float, float], size: int, fill: float
) -> np.ndarray:
    """Cut a ``size`` x ``size`` window centred on ``center``, padding with ``fill``."""

    top = int(round(center[0] - size / 2.0))
    left = int(round(center[1] - size / 2.0))
    out = np.full((size, size), fill, dtype=np.float32)
    height, width = image.shape
    src_t, src_l = max(top, 0), max(left, 0)
    src_b, src_r = min(top + size, height), min(left + size, width)
    if src_b > src_t and src_r > src_l:
        patch = image[src_t:src_b, src_l:src_r]
        out[src_t - top : src_b - top, src_l - left : src_r - left] = patch
    return out


def soft_mask(image: np.ndarray) -> np.ndarray:
    """Soft (H, W) float32 foreground weight: block mask, nearest-upsampled, box-blurred."""

    mask = block_mask(image).astype(np.float64)
    full = np.kron(mask, np.ones((BLOCK, BLOCK)))
    k = BLOCK + 1
    r = k // 2
    padded = np.pad(full, r, mode="edge")
    summed = padded.cumsum(0).cumsum(1)
    summed = np.pad(summed, ((1, 0), (1, 0)))
    height, width = full.shape
    box = (
        summed[k : k + height, k : k + width]
        - summed[0:height, k : k + width]
        - summed[k : k + height, 0:width]
        + summed[0:height, 0:width]
    )
    return (box / (k * k)).astype(np.float32)


def normalize_intensity(window: np.ndarray) -> np.ndarray:
    """Masked z-score of a square window, clipped and multiplied by the soft mask."""

    image = np.asarray(window, dtype=np.float32)
    weight = soft_mask(image).astype(np.float64)
    total = weight.sum()
    if total < BLOCK * BLOCK:
        weight = np.ones_like(weight)
        total = weight.sum()
    values = image.astype(np.float64)
    mean = (weight * values).sum() / total
    var = (weight * (values - mean) ** 2).sum() / total
    z = np.clip((values - mean) / np.sqrt(var + 1.0), -CLIP, CLIP)
    return (z * weight).astype(np.float32)


def prepare(gray: np.ndarray, period: float | None = None) -> Prepared:
    """Trim frame bars, estimate ridge period (unless given), rescale, locate foreground."""

    trimmed = trim_uniform_border(_as_float_gray(gray))
    if period is None:
        period = estimate_ridge_period(trimmed)
    image, scale = rescale_to_period(trimmed, period)
    center, background = foreground_center(image)
    return Prepared(image, period, scale, center, background)


def preprocess(gray: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Full inference preprocessing: ``(H, W)`` image → ``(size, size)`` float32 model input."""

    prepared = prepare(gray)
    window = crop_around(prepared.image, prepared.center, size, prepared.background)
    return normalize_intensity(window)


def rotate_input(window: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a normalised model input about its centre (bilinear, background = 0).

    Used for optional rotation test-time augmentation; positive angles are
    counter-clockwise.
    """

    image = Image.fromarray(np.asarray(window, dtype=np.float32))
    rotated = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0.0)
    return np.asarray(rotated, dtype=np.float32)
