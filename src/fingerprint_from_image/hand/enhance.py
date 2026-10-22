"""Contactless fingertip normalisation and ridge-visibility quality (numpy only).

* :func:`flatten_illumination` removes smooth shading by dividing by a wide
  low-pass estimate, then standardises local contrast.
* :func:`clahe` is a contrast-limited adaptive histogram equalisation with
  bilinear interpolation between tile look-up tables (as in Zuiderveld 1994).
* :func:`ridge_quality` scores how visible friction ridges are, from the
  share of spectral energy in the expected ridge-frequency band and the local
  orientation coherence of the band-passed signal.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = [
    "block_mean",
    "box_blur",
    "clahe",
    "flatten_illumination",
    "gaussian_blur",
    "normalize_fingertip",
    "ridge_quality",
    "smooth_wide",
]


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter of size ``2*radius+1`` with reflected borders (separable)."""

    out = np.asarray(image, dtype=np.float64)
    if radius < 1:
        return out.copy()
    for axis in (0, 1):
        r = min(radius, out.shape[axis] - 1)
        if r < 1:
            continue
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        padded = np.pad(out, pad, mode="reflect" if out.shape[axis] > r + 1 else "edge")
        csum = np.cumsum(padded, axis=axis)
        upper = np.take(csum, np.arange(2 * r + 1, padded.shape[axis]), axis=axis)
        lower = np.take(csum, np.arange(0, padded.shape[axis] - 2 * r - 1), axis=axis)
        out = (upper - lower) / (2 * r + 1)
    return out


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate Gaussian blur by three box passes (Wells 1986)."""

    if sigma <= 0:
        return np.asarray(image, dtype=np.float64).copy()
    radius = max(1, int(round((np.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))
    out = np.asarray(image, dtype=np.float64)
    for _ in range(3):
        out = box_blur(out, radius)
    return out


def block_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Average non-overlapping ``factor x factor`` blocks (edge-padded)."""

    array = np.asarray(image, dtype=np.float64)
    if factor <= 1:
        return array
    h, w = array.shape
    ph, pw = -h % factor, -w % factor
    if ph or pw:
        array = np.pad(array, ((0, ph), (0, pw)), mode="edge")
    return array.reshape(array.shape[0] // factor, factor, -1, factor).mean(axis=(1, 3))


def smooth_wide(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass for large ``sigma``, computed on a decimated grid.

    The field is band-limited far below the decimated Nyquist rate, so block
    averaging, blurring with ``sigma / k`` and bilinear upsampling matches a
    full-resolution blur closely at a fraction of the cost.
    """

    array = np.asarray(image, dtype=np.float64)
    factor = int(sigma // 4)
    if factor < 2 or min(array.shape) < 4 * factor:
        return gaussian_blur(array, sigma)
    h, w = array.shape
    small = gaussian_blur(block_mean(array, factor), sigma / factor)
    up_w, up_h = small.shape[1] * factor, small.shape[0] * factor
    upsampled = Image.fromarray(small.astype(np.float32)).resize((up_w, up_h),
                                                                 Image.Resampling.BILINEAR)
    return np.asarray(upsampled, dtype=np.float64)[:h, :w]


def flatten_illumination(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Remove shading and normalise local contrast; returns zero-mean float64.

    ``sigma`` (pixels) should be several ridge periods so ridges survive.
    """

    image = np.asarray(gray, dtype=np.float64)
    background = smooth_wide(image, sigma)
    flat = image / np.maximum(background, 1.0) - 1.0
    local_std = np.sqrt(np.maximum(smooth_wide(flat * flat, sigma), 0.0))
    return flat / (local_std + 1e-3)


def clahe(image: np.ndarray, *, tiles: tuple[int, int] = (8, 8), clip_limit: float = 2.0,
          bins: int = 256) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation of a uint8 image."""

    img = np.asarray(image)
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError("clahe expects a 2-D uint8 image")
    h, w = img.shape
    ty, tx = max(1, min(tiles[0], h)), max(1, min(tiles[1], w))
    ys = np.linspace(0, h, ty + 1).round().astype(int)
    xs = np.linspace(0, w, tx + 1).round().astype(int)
    luts = np.empty((ty, tx, bins), dtype=np.float64)
    for i in range(ty):
        for j in range(tx):
            tile = img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            hist = np.bincount(tile.ravel(), minlength=bins).astype(np.float64)
            limit = max(1.0, clip_limit * tile.size / bins)
            excess = np.clip(hist - limit, 0, None).sum()
            hist = np.minimum(hist, limit) + excess / bins
            cdf = np.cumsum(hist)
            luts[i, j] = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-9) * 255.0
    # Bilinear interpolation between the LUTs of the four nearest tile centres.
    cy = (ys[:-1] + ys[1:]) / 2.0
    cx = (xs[:-1] + xs[1:]) / 2.0
    gy = np.interp(np.arange(h) + 0.5, cy, np.arange(ty))
    gx = np.interp(np.arange(w) + 0.5, cx, np.arange(tx))
    y0 = np.floor(gy).astype(int)
    x0 = np.floor(gx).astype(int)
    y1, x1 = np.minimum(y0 + 1, ty - 1), np.minimum(x0 + 1, tx - 1)
    wy, wx = (gy - y0)[:, None], (gx - x0)[None, :]
    v = img.astype(np.intp)
    top = luts[y0[:, None], x0[None, :], v] * (1 - wx) + luts[y0[:, None], x1[None, :], v] * wx
    bottom = luts[y1[:, None], x0[None, :], v] * (1 - wx) + luts[y1[:, None], x1[None, :], v] * wx
    return np.clip(top * (1 - wy) + bottom * wy + 0.5, 0, 255).astype(np.uint8)


def _to_uint8(values: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    low, high = np.percentile(values, [low_pct, high_pct])
    if high <= low:
        return np.full(values.shape, 128, dtype=np.uint8)
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def normalize_fingertip(gray: np.ndarray, ridge_period_px: float, *, clip_limit: float = 2.0,
                        tile_periods: float = 8.0) -> np.ndarray:
    """Illumination flattening followed by CLAHE; returns ``(H, W) uint8``.

    Tiles span roughly ``tile_periods`` ridge periods so each tile holds a few
    ridges but still adapts to shading across the fingertip.
    """

    image = np.asarray(gray, dtype=np.float64)
    period = max(float(ridge_period_px), 2.0)
    flat = flatten_illumination(image, sigma=2.5 * period)
    tile = max(8.0, tile_periods * period)
    tiles = (max(1, round(image.shape[0] / tile)), max(1, round(image.shape[1] / tile)))
    return clahe(_to_uint8(flat), tiles=tiles, clip_limit=clip_limit)


def _radial_frequency(shape: tuple[int, int]) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    return np.sqrt(fx * fx + fy * fy)


def ridge_quality(gray: np.ndarray, ridge_period_px: float, *, band: float = 1.6,
                  central_fraction: float = 0.6, min_period_px: float = 2.5,
                  good_period_px: float = 5.0) -> float:
    """Ridge visibility in ``[0, 1]`` for an upright fingertip crop.

    Three factors are multiplied:

    * **resolution** - ramps from 0 at ``min_period_px`` pixels per ridge
      period to 1 at ``good_period_px``; below ~2.5 px ridges are not resolved;
    * **band energy** - share of non-illumination spectral energy within
      ``[f0 / band, f0 * band]`` (``f0 = 1 / ridge_period_px``), rescaled so that
      white noise scores 0;
    * **coherence** - energy-weighted structure-tensor coherence of the
      band-passed image over blocks of ~2 periods (ridges are locally parallel,
      noise and skin texture are not).

    The score is measured on the central part of the crop (the pad), so the
    background around the finger does not count. Crops with more than
    ``2 * good_period_px`` pixels per period are block-averaged first (the score
    is resolution-saturated there), which keeps the cost independent of the
    photo resolution.
    """

    image = np.asarray(gray, dtype=np.float64)
    period = float(ridge_period_px)
    resolution = float(np.clip((period - min_period_px) / (good_period_px - min_period_px), 0, 1))
    factor = int(period // good_period_px)
    if factor >= 2:
        image, period = block_mean(image, factor), period / factor
    h, w = image.shape
    ch, cw = int(h * central_fraction), int(w * central_fraction)
    if resolution == 0.0 or min(ch, cw) < 4 * period or min(ch, cw) < 16:
        return 0.0
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    patch = image[y0:y0 + ch, x0:x0 + cw]
    flat = flatten_illumination(patch, sigma=2.5 * period)
    window = np.outer(np.hanning(ch), np.hanning(cw))
    spectrum = np.fft.rfft2(flat * window)
    power = np.abs(spectrum) ** 2
    rho = _radial_frequency((ch, cw))
    f0 = 1.0 / period
    lowest = f0 / (2.5 * band)
    valid = (rho >= lowest) & (rho <= 0.5)
    in_band = valid & (rho >= f0 / band) & (rho <= min(f0 * band, 0.5))
    total = power[valid].sum()
    if total <= 0 or not in_band.any():
        return 0.0
    ratio = power[in_band].sum() / total
    chance = in_band.sum() / valid.sum()
    band_score = float(np.clip((ratio - chance) / max(1.0 - chance, 1e-6), 0, 1))
    bandpassed = np.fft.irfft2(np.where(in_band, spectrum, 0), s=(ch, cw))
    gy, gx = np.gradient(bandpassed)
    block = max(1, int(round(period)))
    jxx = gaussian_blur(gx * gx, block)
    jyy = gaussian_blur(gy * gy, block)
    jxy = gaussian_blur(gx * gy, block)
    energy = jxx + jyy
    coherence_map = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(energy, 1e-12)
    coherence = float((coherence_map * energy * window).sum() / max((energy * window).sum(), 1e-12))
    coherence_score = float(np.clip((coherence - 0.35) / 0.45, 0, 1))
    return float(np.clip(resolution * np.sqrt(band_score * coherence_score), 0.0, 1.0))
