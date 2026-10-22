"""Per-image checks: decode, size, bit depth, hash, near-duplicate descriptor, ridge period.

Everything here uses numpy + Pillow only, so it runs in CI and on CPU.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .foreground import masked_descriptor

__all__ = [
    "DESCRIPTOR_SHAPE",
    "ImageProbe",
    "estimate_ridge_period",
    "fine_similarity",
    "near_dup_descriptor",
    "probe_image",
    "to_gray_uint8",
]

DESCRIPTOR_SHAPE = (64, 64)
_MAX_PIXELS = 40_000_000


@dataclass
class ImageProbe:
    """Result of :func:`probe_image`; ``ok`` is False when the file is unusable."""

    path: str
    ok: bool
    sha256: str = ""
    width: int = 0
    height: int = 0
    mode: str = ""
    issues: list[str] = field(default_factory=list)
    ridge_period: float | None = None
    descriptor: np.ndarray | None = None


def to_gray_uint8(image: Image.Image) -> np.ndarray:
    """Convert any Pillow image (L, P, RGB(A), I;16, I, F, 1) to ``(H, W) uint8``."""

    if image.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        array = np.asarray(image, dtype=np.float64)
        low, high = float(array.min()), float(array.max())
        scale = 255.0 / (high - low) if high > low else 0.0
        return np.clip((array - low) * scale, 0, 255).astype(np.uint8)
    if image.mode != "L":
        image = image.convert("L")
    return np.asarray(image, dtype=np.uint8)


def near_dup_descriptor(gray: np.ndarray) -> np.ndarray:
    """Coarse descriptor for candidate search: [whole-frame | ridge-crop], each 64x64.

    Both halves are unit-norm masked high-pass images (background zeroed).
    The whole-frame half matches pixel-aligned copies; the crop half matches
    the same print on a different canvas.  Candidates are re-checked by
    :func:`fine_similarity`, because 64 px cannot resolve ridges of large prints.
    """

    side = DESCRIPTOR_SHAPE[0]
    return np.concatenate([masked_descriptor(gray, side=side, blur=1.5, crop=False),
                           masked_descriptor(gray, side=side, blur=1.5, crop=True)])


def fine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Max of whole-frame (192 px) and ridge-crop (160 px) masked high-pass cosines."""

    whole = masked_descriptor(a, 192, 2.5, crop=False) @ masked_descriptor(b, 192, 2.5,
                                                                           crop=False)
    cropped = masked_descriptor(a, 160, 2.5) @ masked_descriptor(b, 160, 2.5)
    return float(max(whole, cropped))


def estimate_ridge_period(gray: np.ndarray, *, window: int = 128) -> float | None:
    """Median dominant ridge period (pixels) over foreground blocks, via 2-D FFT.

    Returns ``None`` when no block shows a clear ridge peak.  Typical adult
    fingertips have an inter-ridge distance of ~0.45-0.5 mm, i.e. ~9-10 px at
    500 dpi; ``dpi ~= period_px / 0.47 mm * 25.4``.
    """

    height, width = gray.shape
    if min(height, width) < window:
        window = 1 << int(np.log2(max(min(height, width), 32)))
    image = gray.astype(np.float32)
    periods: list[float] = []
    hann = np.outer(np.hanning(window), np.hanning(window)).astype(np.float32)
    yy, xx = np.mgrid[:window, :window]
    radius = np.hypot(yy - window // 2, xx - window // 2)
    step = window // 4
    candidates = 0
    for top in range(0, height - window + 1, step):
        for left in range(0, width - window + 1, step):
            block = image[top:top + window, left:left + window]
            candidates += 1
            if block.std() < 12:
                continue
            spectrum = np.abs(np.fft.fftshift(np.fft.fft2((block - block.mean()) * hann)))
            band = (radius >= window / 30) & (radius <= window / 2.4)
            values = np.where(band, spectrum, 0)
            peak = np.unravel_index(int(values.argmax()), values.shape)
            if values[peak] < 4 * np.median(spectrum[band]):
                continue
            periods.append(window / float(radius[peak]))
    needed = 3 if candidates >= 6 else 1
    return float(np.median(periods)) if len(periods) >= needed else None


def probe_image(path: str | Path, *, descriptor: bool = True,
                ridge_period: bool = False) -> ImageProbe:
    """Hash and fully decode one file; record issues instead of raising."""

    source = Path(path)
    probe = ImageProbe(path=str(source), ok=False)
    try:
        data = source.read_bytes()
    except OSError as exc:
        probe.issues.append(f"unreadable: {exc}")
        return probe
    probe.sha256 = hashlib.sha256(data).hexdigest()
    if not data:
        probe.issues.append("empty file")
        return probe
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as image:
                probe.mode = image.mode
                probe.width, probe.height = image.size
                if probe.width * probe.height > _MAX_PIXELS:
                    probe.issues.append("too many pixels")
                    return probe
                image.load()
                frames = int(getattr(image, "n_frames", 1))
                if frames > 1:
                    probe.issues.append(f"multi-frame ({frames}); first frame used")
                gray = to_gray_uint8(image)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning) as exc:
        probe.issues.append(f"decode error: {type(exc).__name__}: {exc}")
        return probe
    if probe.mode not in ("L", "P", "RGB", "RGBA", "1"):
        probe.issues.append(f"mode {probe.mode} rescaled to 8-bit")
    if probe.mode in ("RGB", "RGBA"):
        rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.int16)
        spread = max(np.abs(rgb[..., 0] - rgb[..., 1]).max(),
                     np.abs(rgb[..., 1] - rgb[..., 2]).max())
        if spread > 8:
            probe.issues.append("colour image")
    if int(gray.max()) - int(gray.min()) < 16:
        probe.issues.append("nearly constant image")
    probe.ok = min(probe.width, probe.height) >= 32 and "nearly constant image" not in probe.issues
    if not probe.ok and "nearly constant image" not in probe.issues:
        probe.issues.append("too small")
    if probe.ok and descriptor:
        probe.descriptor = near_dup_descriptor(gray)
    if probe.ok and ridge_period:
        probe.ridge_period = estimate_ridge_period(gray)
    return probe
