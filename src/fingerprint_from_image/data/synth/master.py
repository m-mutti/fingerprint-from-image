"""Master ridge patterns: orientation field (zero-pole model) + iterative Gabor growth.

Follows the SFinGe recipe (Cappelli, Maio, Maltoni): pick a pattern class
and its singular points, build the orientation field with the Sherlock-Monro
zero-pole model plus smooth noise, then grow ridges from random noise by
repeatedly applying orientation-tuned Gabor filters.  numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CLASS_PRIORS", "MasterPrint", "make_master", "smooth_noise"]

# Approximate natural frequencies (Wilson et al. / NIST): loops dominate.
CLASS_PRIORS = {"left_loop": 0.32, "right_loop": 0.32, "whorl": 0.28, "arch": 0.04,
                "tented_arch": 0.04}
_ORIENTATION_BINS = 16


@dataclass(frozen=True)
class MasterPrint:
    """A finger's identity: ridge field in [-1, 1], finger mask in [0, 1], metadata."""

    ridges: np.ndarray
    mask: np.ndarray
    pattern: str
    period: float


def smooth_noise(rng: np.random.Generator, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Zero-mean, unit-std Gaussian random field with correlation length ``sigma`` px."""

    white = rng.standard_normal(shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx**2 + fy**2))
    field = np.fft.irfft2(np.fft.rfft2(white) * kernel, s=shape)
    field -= field.mean()
    return field / (field.std() + 1e-9)


def _singular_points(pattern: str, rng: np.random.Generator, w: int, h: int
                     ) -> tuple[list[complex], list[complex]]:
    cx, cy = w * rng.uniform(0.42, 0.58), h * rng.uniform(0.38, 0.52)
    core = complex(cx, cy)
    if pattern == "left_loop":
        return [core], [complex(cx + w * rng.uniform(0.18, 0.3), cy + h * rng.uniform(0.25, 0.35))]
    if pattern == "right_loop":
        return [core], [complex(cx - w * rng.uniform(0.18, 0.3), cy + h * rng.uniform(0.25, 0.35))]
    if pattern == "tented_arch":
        return [core], [complex(cx + w * rng.uniform(-0.03, 0.03), cy + h * rng.uniform(0.12, 0.2))]
    if pattern == "whorl":
        offset = complex(w * rng.uniform(-0.06, 0.06), h * rng.uniform(0.04, 0.1))
        deltas = [complex(cx - w * rng.uniform(0.25, 0.33), cy + h * rng.uniform(0.25, 0.33)),
                  complex(cx + w * rng.uniform(0.25, 0.33), cy + h * rng.uniform(0.25, 0.33))]
        return [core - offset, core + offset], deltas
    return [], []


def orientation_field(pattern: str, rng: np.random.Generator, shape: tuple[int, int]
                      ) -> np.ndarray:
    """Ridge orientation (radians, mod pi) on the canvas."""

    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    z = xx + 1j * yy
    cores, deltas = _singular_points(pattern, rng, w, h)
    if pattern == "arch":
        cx = w * rng.uniform(0.4, 0.6)
        strength = rng.uniform(1.0, 2.2)
        theta = np.arctan(strength * (cx - xx) / w * np.exp(-((yy - 0.75 * h) / h) ** 2))
    else:
        # Zero-pole model in image coordinates (y down): cores +, deltas -.
        theta = np.zeros(shape)
        for c in cores:
            theta += np.angle(z - c)
        for d in deltas:
            theta -= np.angle(z - d)
        theta *= 0.5
    theta += rng.uniform(-0.12, 0.12) + 0.12 * smooth_noise(rng, shape, 60)
    return np.mod(theta, np.pi)


def _gabor_bank(shape: tuple[int, int], periods: tuple[float, ...]) -> np.ndarray:
    """rFFT transfer functions [period, orientation, fy, fx] of even Gabor filters."""

    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    bank = np.empty((len(periods), _ORIENTATION_BINS, h, w // 2 + 1), dtype=np.float32)
    for p, period in enumerate(periods):
        f0, sigma_f = 1.0 / period, 0.35 / period
        for o in range(_ORIENTATION_BINS):
            theta = np.pi * o / _ORIENTATION_BINS
            # Wave vector is perpendicular to the ridge direction theta.
            ux, uy = -np.sin(theta), np.cos(theta)
            along = fx * ux + fy * uy
            across = -fx * uy + fy * ux
            response = (np.exp(-((along - f0) ** 2 + across**2) / (2 * sigma_f**2))
                        + np.exp(-((along + f0) ** 2 + across**2) / (2 * sigma_f**2)))
            bank[p, o] = response.astype(np.float32)
    return bank


def _finger_mask(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(np.float64)
    ax, ay = w * rng.uniform(0.36, 0.44), h * rng.uniform(0.40, 0.47)
    cx, cy = w / 2 + rng.uniform(-8, 8), h * rng.uniform(0.47, 0.52)
    r = ((xx - cx) / ax) ** 2 + ((np.minimum(yy - cy, 0)) / ay) ** 2 \
        + ((np.maximum(yy - cy, 0)) / (1.25 * ay)) ** 4
    r += 0.06 * smooth_noise(rng, shape, 40)
    return np.clip((1.0 - r) * 8, 0, 1)


def make_master(seed: int, shape: tuple[int, int] = (480, 400), iterations: int = 14
                ) -> MasterPrint:
    """Deterministically generate one master print from ``seed``."""

    rng = np.random.default_rng(seed)
    names = list(CLASS_PRIORS)
    pattern = names[int(rng.choice(len(names), p=list(CLASS_PRIORS.values())))]
    theta = orientation_field(pattern, rng, shape)
    base = rng.uniform(8.2, 10.2)
    periods = (base * 0.88, base * 1.12)
    bank = _gabor_bank(shape, periods)
    # Interpolation weights over orientation bins and the two period bins.
    pos = theta / np.pi * _ORIENTATION_BINS
    lower = np.floor(pos).astype(int) % _ORIENTATION_BINS
    upper = (lower + 1) % _ORIENTATION_BINS
    frac = (pos - np.floor(pos)).astype(np.float32)
    mix = np.clip(0.5 + 0.35 * smooth_noise(rng, shape, 90), 0, 1).astype(np.float32)
    x = rng.standard_normal(shape).astype(np.float32)
    for _ in range(iterations):
        spectrum = np.fft.rfft2(x)
        out = np.zeros(shape, dtype=np.float32)
        for p, weight_p in enumerate((1 - mix, mix)):
            responses = {}
            for o in np.unique(np.concatenate([lower.ravel(), upper.ravel()])):
                responses[int(o)] = np.fft.irfft2(spectrum * bank[p, o], s=shape)
            stack_lower = np.choose(lower, [responses.get(o, 0) for o in range(_ORIENTATION_BINS)])
            stack_upper = np.choose(upper, [responses.get(o, 0) for o in range(_ORIENTATION_BINS)])
            out += weight_p * ((1 - frac) * stack_lower + frac * stack_upper)
        x = np.tanh(2.5 * out / (out.std() + 1e-6)).astype(np.float32)
    return MasterPrint(ridges=x, mask=_finger_mask(rng, shape).astype(np.float32),
                       pattern=pattern, period=float(base))
