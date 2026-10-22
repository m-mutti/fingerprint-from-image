"""Turn a master ridge pattern into one sensor-like impression (numpy + Pillow)."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from .master import MasterPrint, smooth_noise

__all__ = ["bilinear_sample", "render_impression"]


def bilinear_sample(image: np.ndarray, ys: np.ndarray, xs: np.ndarray, fill: float = 0.0
                    ) -> np.ndarray:
    """Sample ``image`` at float coordinates (bilinear, ``fill`` outside)."""

    h, w = image.shape
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    wy, wx = ys - y0, xs - x0
    out = np.zeros(ys.shape, dtype=np.float32)
    for dy, fy in ((0, 1 - wy), (1, wy)):
        for dx, fx in ((0, 1 - wx), (1, wx)):
            yi, xi = y0 + dy, x0 + dx
            inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            values = np.where(inside, image[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], fill)
            out += (fy * fx * values).astype(np.float32)
    return out


def render_impression(master: MasterPrint, seed: int) -> np.ndarray:
    """One impression: pose + elastic distortion, partial contact, pressure, sensor noise.

    Returns a ``(H, W) uint8`` image with dark ridges on a light background.
    """

    rng = np.random.default_rng(seed)
    h, w = master.ridges.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    # Pose: rotation, translation, slight scale (inverse mapping output -> master).
    angle = np.deg2rad(rng.uniform(-18, 18))
    scale = rng.uniform(0.96, 1.04)
    ty, tx = rng.uniform(-28, 28), rng.uniform(-28, 28)
    cy, cx = h / 2, w / 2
    cos, sin = np.cos(angle) / scale, np.sin(angle) / scale
    src_x = cos * (xx - cx - tx) + sin * (yy - cy - ty) + cx
    src_y = -sin * (xx - cx - tx) + cos * (yy - cy - ty) + cy
    # Elastic (skin plasticity) displacement, larger away from the contact centre.
    amplitude = rng.uniform(2.0, 7.0)
    radial = np.clip(np.hypot(xx - cx, yy - cy) / (0.5 * min(h, w)), 0, 1.5)
    src_x += amplitude * radial * smooth_noise(rng, (h, w), 70).astype(np.float32)
    src_y += amplitude * radial * smooth_noise(rng, (h, w), 70).astype(np.float32)
    ridges = bilinear_sample(master.ridges, src_y, src_x)
    mask = bilinear_sample(master.mask, src_y, src_x)

    # Partial contact: an elliptical contact area with soft border.
    ey, ex = cy + rng.uniform(-50, 50), cx + rng.uniform(-40, 40)
    ay, ax = h * rng.uniform(0.33, 0.55), w * rng.uniform(0.32, 0.55)
    contact = 1.0 - (((yy - ey) / ay) ** 2 + ((xx - ex) / ax) ** 2)
    contact += 0.05 * smooth_noise(rng, (h, w), 45)
    contact = np.clip(contact * 6, 0, 1) * mask

    # Pressure / moisture: threshold shift thins (dry) or thickens (wet) ridges.
    wetness = rng.uniform(-0.45, 0.35)
    local = wetness + 0.25 * smooth_noise(rng, (h, w), 25)
    ink = 1.0 / (1.0 + np.exp(-6.0 * (ridges + local)))
    dry_breaks = smooth_noise(rng, (h, w), 2.5) > rng.uniform(1.1, 2.4)
    ink = np.where(dry_breaks, ink * 0.25, ink)
    ink *= np.clip(1.0 + 0.25 * smooth_noise(rng, (h, w), 15), 0.4, 1.2)

    background = rng.uniform(175, 248)
    darkness = rng.uniform(110, 210)
    image = background - darkness * ink * contact
    image += 0.03 * darkness * smooth_noise(rng, (h, w), 40) * (1 - contact)
    image += rng.normal(0, rng.uniform(2, 11), (h, w))
    image = np.clip(image, 0, 255).astype(np.uint8)
    blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.1)))
    gamma = rng.uniform(0.8, 1.25)
    out = 255.0 * (np.asarray(blurred, dtype=np.float32) / 255.0) ** gamma
    return np.clip(out, 0, 255).astype(np.uint8)
