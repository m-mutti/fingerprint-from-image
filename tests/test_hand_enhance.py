from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from fingerprint_from_image.hand.enhance import (
    box_blur,
    clahe,
    flatten_illumination,
    gaussian_blur,
    normalize_fingertip,
    ridge_quality,
)

DATA = Path(__file__).parent / "data"


def _ridges(period: float = 8.0, shape=(320, 240), amplitude: float = 30.0) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(float)
    radius = np.hypot(xx - shape[1] / 2, yy - shape[0] * 0.55)
    return 128 + amplitude * np.cos(2 * np.pi * radius / period)


def _camera(image: np.ndarray, seed: int = 0) -> np.ndarray:
    noisy = image + np.random.default_rng(seed).normal(0, 2.0, image.shape)
    return np.clip(np.round(noisy), 0, 255)


def test_box_and_gaussian_blur_preserve_mean_and_constants() -> None:
    image = np.random.default_rng(0).uniform(0, 255, (40, 50))
    assert box_blur(image, 3).shape == image.shape
    assert box_blur(image, 3).mean() == pytest.approx(image.mean(), rel=0.02)
    np.testing.assert_allclose(gaussian_blur(np.full((9, 7), 5.0), 4.0), 5.0)
    assert box_blur(image, 0) is not image
    assert np.std(gaussian_blur(image, 2.0)) < np.std(image) / 2


def test_flatten_illumination_removes_shading() -> None:
    ridges = _ridges()
    shading = 0.4 + 0.6 * np.linspace(0, 1, ridges.shape[1])[None, :]
    flat = flatten_illumination(ridges * shading, sigma=20)
    left, right = flat[100:220, 30:90], flat[100:220, 150:210]
    assert left.std() == pytest.approx(right.std(), rel=0.25)  # contrast equalised
    assert abs(left.mean() - right.mean()) < 0.2


def test_clahe_properties() -> None:
    ramp = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
    out = clahe(ramp, tiles=(1, 1))
    assert out.dtype == np.uint8 and out.shape == ramp.shape
    assert np.all(np.diff(out[0].astype(int)) >= 0)
    low_contrast = (100 + _ridges(amplitude=8.0) / 16).astype(np.uint8)
    assert clahe(low_contrast, tiles=(4, 4)).std() > 2 * low_contrast.std()
    assert len(np.unique(clahe(np.full((32, 32), 77, np.uint8)))) == 1
    with pytest.raises(ValueError):
        clahe(np.zeros((4, 4), dtype=np.float32))


def test_normalize_fingertip_output() -> None:
    shading = np.linspace(0.3, 1.0, 240)[None, :]
    out = normalize_fingertip(_camera(_ridges() * shading), 8.0)
    assert out.dtype == np.uint8 and out.shape == (320, 240)
    assert out[:, :60].std() > 40 and out[:, -60:].std() > 40  # ridges on both sides


def test_ridge_quality_on_synthetic_signals() -> None:
    ridges = _camera(_ridges())
    sharp = ridge_quality(ridges, 8.0)
    assert sharp > 0.7
    assert ridge_quality(_camera(gaussian_blur(_ridges(), 3.0)), 8.0) < 0.1
    rng = np.random.default_rng(1)
    assert ridge_quality(_camera(128 + rng.normal(0, 30, (320, 240))), 8.0) < 0.1
    skin = 128 + 300 * gaussian_blur(rng.normal(0, 1, (320, 240)), 6)
    assert ridge_quality(_camera(skin), 8.0) < 0.1
    assert ridge_quality(np.full((320, 240), 120.0), 8.0) == 0.0
    assert ridge_quality(ridges, 2.0) == 0.0  # 2 px periods are below resolvable
    assert ridge_quality(ridges[:20, :20], 8.0) == 0.0  # too small to judge
    assert 0.0 <= ridge_quality(ridges, 11.0) <= 1.0


def test_ridge_quality_on_real_fingertip_macro() -> None:
    """Public-domain macro photo (tests/data/README.md): ~15 px ridge period."""

    pad = Image.open(DATA / "fingertip_macro_pad.jpg").convert("L")
    prior = 410 * 0.45 / 16  # finger ~410 px wide in this photo
    sharp = ridge_quality(np.asarray(pad, dtype=float), prior)
    blurred = ridge_quality(np.asarray(pad.filter(ImageFilter.GaussianBlur(4)), dtype=float),
                            prior)
    small = pad.resize((pad.width // 4, pad.height // 4), Image.Resampling.BOX)
    tiny = ridge_quality(np.asarray(small, dtype=float), prior / 4)
    assert sharp > 0.5
    assert blurred < sharp - 0.2
    assert tiny == 0.0
