"""Batched GPU acquisition augmentation and the torch port of the runtime normaliser.

``normalize_intensity_torch`` reproduces
:func:`fingerprint_from_image.embedding.preprocess.normalize_intensity`
(verified to ~1e-4 by ``training/tests_parity.py``), so the network is trained on
exactly the input distribution produced at inference.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from fingerprint_from_image.embedding import preprocess as pp
from training.config import AugConfig

__all__ = ["Augmenter", "normalize_intensity_torch"]


def _block_mask(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(N, H, W) float → (N, H/B, W/B) bool, mirroring ``preprocess.block_mask``."""

    n, h, w = x.shape
    b = pp.BLOCK
    blocks = x.to(dtype).reshape(n, h // b, b, w // b, b)
    std = blocks.std(dim=(2, 4), unbiased=False)
    q90 = torch.quantile(std.reshape(n, -1), 0.9, dim=1)
    thr = torch.clamp(pp._REL_STD * q90, min=pp._ABS_STD_FLOOR)
    raw = (std > thr[:, None, None]).float()
    padded = F.pad(raw[:, None], (1, 1, 1, 1), mode="replicate")
    votes = F.avg_pool2d(padded, 3, stride=1)[:, 0] * 9.0
    return votes > 4.5


def normalize_intensity_torch(x: torch.Tensor, *, precise: bool = False) -> torch.Tensor:
    """(N, 1, S, S) float in [0, 255] → masked, clipped z-score, like the numpy runtime.

    ``precise=True`` computes in float64 (bit-faithful to numpy, ~5x slower on
    consumer GPUs); the float32 default differs by < 1e-3 (see tests).
    """

    dtype = torch.float64 if precise else torch.float32
    b = pp.BLOCK
    mask = _block_mask(x[:, 0], dtype).to(dtype)
    full = mask.repeat_interleave(b, dim=1).repeat_interleave(b, dim=2)[:, None]
    k = b + 1
    padded = F.pad(full, (k // 2,) * 4, mode="replicate")
    weight = F.avg_pool2d(F.avg_pool2d(padded, (1, k), stride=1), (k, 1), stride=1)
    total = weight.sum(dim=(1, 2, 3), keepdim=True)
    empty = total < b * b
    weight = torch.where(empty, torch.ones_like(weight), weight)
    total = weight.sum(dim=(1, 2, 3), keepdim=True)
    values = x.to(dtype)
    mean = (weight * values).sum(dim=(1, 2, 3), keepdim=True) / total
    var = (weight * (values - mean) ** 2).sum(dim=(1, 2, 3), keepdim=True) / total
    z = torch.clamp((values - mean) / torch.sqrt(var + 1.0), -pp.CLIP, pp.CLIP)
    return (z * weight).float()


def _uniform(n: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    return torch.empty(n, device=device).uniform_(low, high)


def _chance(n: int, p: float, device: torch.device) -> torch.Tensor:
    return torch.rand(n, device=device) < p


def _per_sample_conv(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Depthwise conv with a different (k, k) kernel per sample; reflect padding."""

    n, _, h, w = x.shape
    k = kernels.shape[-1]
    padded = F.pad(x.reshape(1, n, h, w), (k // 2,) * 4, mode="reflect")
    return F.conv2d(padded, kernels[:, None], groups=n).reshape(n, 1, h, w)


def _gaussian_kernels(sigma: torch.Tensor, k: int = 7) -> torch.Tensor:
    r = torch.arange(k, device=sigma.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(r[None] ** 2) / (2 * sigma[:, None].clamp(min=1e-3) ** 2))
    g = g / g.sum(dim=1, keepdim=True)
    return g[:, :, None] * g[:, None, :]


_PROBABILITIES = ("partial", "cutout", "lowres", "blur", "sharpen", "morph", "noise",
                  "illumination", "polarity", "gamma", "frame")


class Augmenter:
    """Callable producing normalised ``(N, 1, S, S)`` training inputs from uint8 canvases."""

    def __init__(self, cfg: AugConfig, size: int) -> None:
        if cfg.strength == "light":
            cfg = dataclasses.replace(
                cfg, rotation=15.0, scale=0.08, translate=0.05, elastic=0.0, partial=0.0,
                cutout=0.0, lowres=0.1, blur=0.1, sharpen=0.0, morph=0.0, noise=0.2,
                illumination=0.0, polarity=0.0, gamma=0.2, frame=0.0,
            )
        elif cfg.strength == "medium":  # heavy ranges x0.7, probabilities x0.6
            cfg = dataclasses.replace(
                cfg, rotation=cfg.rotation * 0.7, scale=cfg.scale * 0.7,
                translate=cfg.translate * 0.7, elastic=cfg.elastic * 0.7,
                **{name: getattr(cfg, name) * 0.6 for name in _PROBABILITIES},
            )
        self.cfg = cfg
        self.size = size

    def _geometry(
        self, canvas: torch.Tensor, background: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One fused grid_sample for rotation/zoom/translation/elastic; warps ``valid`` too."""

        cfg, s = self.cfg, self.size
        n, _, c, _ = canvas.shape
        dev = canvas.device
        angle = _uniform(n, -cfg.rotation, cfg.rotation, dev) * math.pi / 180.0
        zoom = torch.exp(_uniform(n, -cfg.scale, cfg.scale, dev))
        aniso = torch.exp(_uniform(n, -cfg.anisotropy, cfg.anisotropy, dev))
        ratio = s / c
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(n, 2, 3, device=dev)
        theta[:, 0, 0] = cos * ratio / (zoom * aniso)
        theta[:, 0, 1] = -sin * ratio / zoom
        theta[:, 1, 0] = sin * ratio / (zoom * aniso)
        theta[:, 1, 1] = cos * ratio / zoom
        theta[:, :, 2] = _uniform(2 * n, -cfg.translate, cfg.translate, dev).view(n, 2) * 2 * ratio
        grid = F.affine_grid(theta, [n, 1, s, s], align_corners=False)
        if cfg.elastic > 0:
            amp = _uniform(n, 0.0, cfg.elastic, dev)[:, None, None, None]
            field = torch.randn(n, 2, 4, 4, device=dev) * amp
            field = F.interpolate(field, size=(s, s), mode="bicubic", align_corners=True)
            grid = grid + field.permute(0, 2, 3, 1) * ratio
        bg = background[:, None, None, None]
        stacked = torch.cat([canvas - bg, valid], dim=1)
        out = F.grid_sample(stacked, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)
        return out[:, :1] + bg, out[:, 1:] > 0.5

    def _occlude(self, x: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
        """Partial-print windows and cutouts; keeps ≥ ``min_print`` of the print area."""

        cfg = self.cfg
        n, _, s, _ = x.shape
        dev = x.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, s, device=dev), torch.linspace(-1, 1, s, device=dev),
            indexing="ij",
        )
        keep = torch.ones_like(x, dtype=torch.bool)
        partial = _chance(n, cfg.partial, dev)
        if partial.any():
            phi = _uniform(n, 0, 2 * math.pi, dev)[:, None, None]
            offset = _uniform(n, -0.1, 0.5, dev)[:, None, None]
            plane = (xx * torch.cos(phi) + yy * torch.sin(phi)) < offset
            cy, cx = _uniform(n, -0.3, 0.3, dev), _uniform(n, -0.3, 0.3, dev)
            ay, ax = _uniform(n, 0.45, 0.9, dev), _uniform(n, 0.35, 0.8, dev)
            ellipse = ((yy - cy[:, None, None]) / ay[:, None, None]) ** 2 + (
                (xx - cx[:, None, None]) / ax[:, None, None]
            ) ** 2 < 1
            kind = _chance(n, 0.5, dev)[:, None, None]
            window = torch.where(kind, plane, ellipse)
            keep &= (window | ~partial[:, None, None])[:, None]
        cut = _chance(n, cfg.cutout, dev)
        if cut.any():
            for _ in range(2):
                cy, cx = _uniform(n, -0.8, 0.8, dev), _uniform(n, -0.8, 0.8, dev)
                hy, hx = _uniform(n, 0.05, 0.25, dev), _uniform(n, 0.05, 0.25, dev)
                box = ((yy - cy[:, None, None]).abs() < hy[:, None, None]) & (
                    (xx - cx[:, None, None]).abs() < hx[:, None, None]
                )
                keep &= ~(box & cut[:, None, None])[:, None]
        b = pp.BLOCK
        print_mask = _block_mask(x[:, 0], torch.float32)
        print_mask = print_mask.repeat_interleave(b, 1).repeat_interleave(b, 2)[:, None]
        total = print_mask.sum(dim=(1, 2, 3)).clamp(min=1)
        retained = (print_mask & keep).sum(dim=(1, 2, 3)) / total
        keep |= (retained < cfg.min_print)[:, None, None, None]  # too little left: undo
        return torch.where(keep, x, bg[:, None, None, None].expand_as(x))

    def _frame(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Synthetic sensor-frame edges: a constant region beyond a line near one border."""

        cfg = self.cfg
        n, _, s, _ = x.shape
        dev = x.device
        hit = _chance(n, cfg.frame, dev)
        if not hit.any():
            return x
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, s, device=dev), torch.linspace(-1, 1, s, device=dev),
            indexing="ij",
        )
        side = torch.randint(0, 4, (n,), device=dev).float() * (math.pi / 2)
        phi = (side + _uniform(n, -0.3, 0.3, dev))[:, None, None]
        offset = _uniform(n, 0.55, 0.95, dev)[:, None, None]
        region = (xx * torch.cos(phi) + yy * torch.sin(phi)) > offset
        level = torch.where(_chance(n, 0.5, dev), _uniform(n, 0, 40, dev),
                            _uniform(n, 150, 255, dev))[:, None, None, None]
        apply = (region[:, None] & valid & hit[:, None, None, None])
        return torch.where(apply, level.expand_as(x), x)

    def _texture(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        n, _, s, _ = x.shape
        dev = x.device
        morph = _chance(n, cfg.morph, dev)
        if morph.any():
            wet = -F.max_pool2d(-x, 3, stride=1, padding=1)  # spreads dark: thick, merged ridges
            dry = F.max_pool2d(x, 3, stride=1, padding=1)  # spreads light: thin, broken ridges
            pick = _chance(n, 0.5, dev)[:, None, None, None]
            alpha = _uniform(n, 0.3, 1.0, dev)[:, None, None, None]
            changed = torch.where(pick, wet, dry) * alpha + x * (1 - alpha)
            x = torch.where(morph[:, None, None, None], changed, x)
        low = _chance(n, cfg.lowres, dev)
        if low.any():
            factor = float(_uniform(1, 0.35, 0.8, dev))
            small = F.interpolate(x, scale_factor=factor, mode="area")
            back = F.interpolate(small, size=(s, s), mode="bilinear", align_corners=False)
            x = torch.where(low[:, None, None, None], back, x)
        blur = _chance(n, cfg.blur, dev)
        if blur.any():
            blurred = _per_sample_conv(x, _gaussian_kernels(_uniform(n, 0.4, 1.4, dev)))
            x = torch.where(blur[:, None, None, None], blurred, x)
        sharp = _chance(n, cfg.sharpen, dev)
        if sharp.any():
            soft = _per_sample_conv(x, _gaussian_kernels(_uniform(n, 0.8, 1.5, dev)))
            amount = _uniform(n, 0.3, 1.2, dev)[:, None, None, None]
            x = torch.where(sharp[:, None, None, None], x + amount * (x - soft), x)
        return x

    def _photometric(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Gamma/contrast globally; illumination and sensor noise only on real pixels."""

        cfg = self.cfg
        n, _, s, _ = x.shape
        dev = x.device
        x = x.clamp(0, 255)
        gamma = torch.where(
            _chance(n, cfg.gamma, dev), torch.exp(_uniform(n, -0.5, 0.5, dev)),
            torch.ones(n, device=dev),
        )
        x = 255.0 * (x / 255.0) ** gamma[:, None, None, None]
        contrast = _uniform(n, 0.6, 1.4, dev)[:, None, None, None]
        bright = _uniform(n, -25, 25, dev)[:, None, None, None]
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * contrast + mean + bright
        illum = _chance(n, cfg.illumination, dev)
        if illum.any():
            field = F.interpolate(torch.randn(n, 1, 3, 3, device=dev), size=(s, s),
                                  mode="bicubic", align_corners=True)
            gain = 1 + 0.25 * field * (illum[:, None, None, None] & valid)
            x = 255 - (255 - x) * gain
        noisy = _chance(n, cfg.noise, dev)
        sigma = _uniform(n, 2.0, 18.0, dev) * noisy
        x = x + torch.randn_like(x) * sigma[:, None, None, None] * valid
        invert = _chance(n, cfg.polarity, dev)[:, None, None, None]
        x = torch.where(invert, 255 - x, x)
        return x.clamp(0, 255)

    @torch.no_grad()
    def __call__(
        self, canvas: torch.Tensor, background: torch.Tensor, boxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``canvas`` (N, 1, C, C) uint8 on GPU, ``background`` (N,), ``boxes`` (N, 4)
        real-pixel rectangles (top, left, bottom, right) → normalised (N, 1, S, S)."""

        n, _, c, _ = canvas.shape
        if boxes is None:
            valid = torch.ones_like(canvas, dtype=torch.float32)
        else:
            idx = torch.arange(c, device=canvas.device)
            rows = (idx[None] >= boxes[:, 0:1]) & (idx[None] < boxes[:, 2:3])
            cols = (idx[None] >= boxes[:, 1:2]) & (idx[None] < boxes[:, 3:4])
            valid = (rows[:, :, None] & cols[:, None, :]).float()[:, None]
        x, valid_out = self._geometry(canvas.float(), background.float(), valid)
        x = self._occlude(x, background.float())
        x = self._frame(x, valid_out)
        x = self._texture(x)
        x = self._photometric(x, valid_out)
        return normalize_intensity_torch(x)
