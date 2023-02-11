from __future__ import annotations

import cv2 as cv
import numpy as np


def synthetic_fingerprint(seed: int, size: int = 520) -> np.ndarray:
    """Create deterministic fingerprint-like ridge detail without biometric data."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    x = (xx - size / 2) / size
    y = (yy - size / 2) / size

    core_x, core_y = rng.uniform(-0.08, 0.08, size=2)
    radius = np.sqrt((x - core_x) ** 2 + 1.35 * (y - core_y) ** 2)
    angle = np.arctan2(y - core_y, x - core_x)
    phase = (
        (82.0 + seed % 7) * radius
        + (2.0 + seed % 3) * angle
        + 2.3 * np.sin(3.0 * angle + seed)
        + 1.6 * np.sin(11.0 * x + seed * 0.7)
    )
    ridges = 142.0 + 82.0 * np.cos(phase)

    mask = ((x / 0.46) ** 2 + ((y + 0.015) / 0.49) ** 2) <= 1.0
    image = np.full((size, size), 246.0, dtype=np.float32)
    image[mask] = ridges[mask]

    # Pores and short ridge interruptions make each generated identity locally
    # distinctive, while an affine variant retains those same landmarks.
    for _ in range(95):
        px = int(rng.uniform(0.17, 0.83) * size)
        py = int(rng.uniform(0.12, 0.88) * size)
        radius_px = int(rng.integers(1, 4))
        color = int(rng.choice((35, 225)))
        cv.circle(image, (px, py), radius_px, color, cv.FILLED)
    noise = rng.normal(0.0, 3.0, image.shape).astype(np.float32)
    image[mask] += noise[mask]
    image = cv.GaussianBlur(image, (3, 3), 0.55)
    return np.clip(image, 0, 255).astype(np.uint8)


def affine_variant(
    image: np.ndarray,
    *,
    angle: float = 7.0,
    scale: float = 0.96,
    tx: float = 11.0,
    ty: float = -8.0,
    noise_seed: int = 100,
) -> np.ndarray:
    height, width = image.shape
    transform = cv.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    transform[:, 2] += (tx, ty)
    result = cv.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=246,
    )
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(0.0, 2.0, result.shape)
    return np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
