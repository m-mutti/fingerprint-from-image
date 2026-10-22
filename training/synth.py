"""SFinGe-style synthetic fingerprints on the GPU (identities × impressions → PNG).

Master print per identity:
  * orientation field from the Sherlock–Monro zero-pole model (arch, tented arch,
    left/right loop, whorl) plus a smooth random perturbation;
  * smooth ridge-frequency map (period 7.5–11 px, i.e. ~500 dpi);
  * ridges grown by iterated orientation/frequency-adaptive Gabor filtering from
    random noise (16 orientation bins, per-print kernels), which produces
    naturally distributed endings and bifurcations.
Impressions of the same master differ by elastic distortion, rotation,
translation, contact region, pressure (ridge width), pores, creases and noise.

Usage::

    python -m training.synth --out E:/Personal_tries/Mutti_Finger_Prints/data/synthetic_v1 \
        --identities 20000 --impressions 2 --seed 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from training.data import Sample

__all__ = ["generate", "synthetic_samples"]

SIZE = 384
BINS = 16


def synthetic_samples(directory: Path) -> list[Sample]:
    """Index ``{identity}_{impression}.png`` files written by :func:`generate`."""

    samples = []
    for path in sorted(directory.glob("*.png")):
        identity = path.stem.split("_")[0]
        samples.append(Sample(str(path), f"synth_{identity}", "synthetic", f"synth_{identity}"))
    return samples


def _smooth_field(n: int, channels: int, grid: int, size: int, gen: torch.Generator,
                  device: torch.device) -> torch.Tensor:
    coarse = torch.randn(n, channels, grid, grid, generator=gen, device=device)
    return F.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=True)


def _orientation(n: int, gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """(N, H, W) ridge orientation in [0, pi) from random singular points."""

    lin = torch.linspace(-1, 1, SIZE, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    z = torch.complex(xx, yy)[None]

    def u(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, generator=gen, device=device) * (hi - lo) + lo

    kind = torch.multinomial(torch.tensor([0.05, 0.07, 0.3, 0.3, 0.28], device=device), n,
                             replacement=True, generator=gen)
    cx, cy = u(-0.12, 0.12), u(-0.2, 0.05)
    side = torch.where(torch.rand(n, generator=gen, device=device) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, SIZE, SIZE, device=device)

    def arg(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        return torch.angle(z - torch.complex(px, py)[:, None, None])

    core_a = arg(cx, cy)
    delta_tent = arg(cx + u(-0.03, 0.03), cy + u(0.18, 0.3))
    loop_dx = side * u(0.3, 0.5)
    delta_loop = arg(cx + loop_dx, cy + u(0.35, 0.55))
    core_b = arg(cx + u(-0.05, 0.05), cy + u(0.08, 0.18))
    delta_wl = arg(cx - u(0.35, 0.55), cy + u(0.35, 0.55))
    delta_wr = arg(cx + u(0.35, 0.55), cy + u(0.35, 0.55))
    bump = torch.exp(-((yy[None] - cy[:, None, None]) ** 2) * 2) * (yy[None] < 0.6)
    arch = torch.atan(u(0.4, 1.2)[:, None, None] * xx[None] * bump)
    theta = torch.where((kind == 0)[:, None, None], arch, theta)
    theta = torch.where((kind == 1)[:, None, None], 0.5 * (core_a - delta_tent), theta)
    loops = (kind == 2) | (kind == 3)
    theta = torch.where(loops[:, None, None], 0.5 * (core_a - delta_loop), theta)
    whorl = 0.5 * (core_a + core_b - delta_wl - delta_wr)
    theta = torch.where((kind == 4)[:, None, None], whorl, theta)
    theta = theta + 0.12 * _smooth_field(n, 1, 5, SIZE, gen, device)[:, 0]
    return torch.remainder(theta, math.pi)


def _gabor_bank(period: torch.Tensor, ksize: int) -> torch.Tensor:
    """(N*BINS, 1, k, k) even Gabor kernels per print (ridges along each bin angle)."""

    r = torch.arange(ksize, device=period.device, dtype=torch.float32) - ksize // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    angles = torch.arange(BINS, device=period.device) * (math.pi / BINS)
    sin, cos = torch.sin(angles)[:, None, None], torch.cos(angles)[:, None, None]
    across = -xx[None] * sin + yy[None] * cos
    along = xx[None] * cos + yy[None] * sin
    p = period[:, None, None, None]
    sigma_x, sigma_y = 0.45 * p, 0.6 * p
    g = torch.exp(-(across[None] ** 2) / (2 * sigma_x ** 2) - along[None] ** 2 / (2 * sigma_y ** 2))
    kernel = g * torch.cos(2 * math.pi * across[None] / p)
    kernel = kernel - kernel.mean(dim=(2, 3), keepdim=True)
    kernel = kernel / kernel.abs().sum(dim=(2, 3), keepdim=True)
    return kernel.reshape(-1, 1, ksize, ksize)


def _grow(theta: torch.Tensor, period: torch.Tensor, gen: torch.Generator,
          iterations: int = 24) -> torch.Tensor:
    """Iterated adaptive Gabor filtering → (N, H, W) ridge field in [-1, 1]."""

    n = theta.shape[0]
    ksize = int(2 * math.ceil(float(period.max())) + 1)
    r_pad = ksize // 2
    side = SIZE + 2 * r_pad
    # FFT convolution (kernels are even, so correlation == convolution): one rfft2 of the
    # padded field per iteration instead of n*BINS grouped spatial convolutions.
    bank = _gabor_bank(period, ksize).reshape(n, BINS, ksize, ksize)
    placed = torch.zeros(n, BINS, side, side, device=theta.device)
    placed[:, :, :ksize, :ksize] = bank
    placed = torch.roll(placed, shifts=(-r_pad, -r_pad), dims=(2, 3))
    bank_f = torch.fft.rfft2(placed)
    pos = theta / math.pi * BINS
    i0 = torch.floor(pos).long() % BINS
    i1 = (i0 + 1) % BINS
    w = (pos - torch.floor(pos))[:, None]
    x = torch.randn(n, 1, SIZE, SIZE, generator=gen, device=theta.device) * 0.5
    for _ in range(iterations):
        padded = F.pad(x, (r_pad,) * 4, mode="reflect")
        full = torch.fft.irfft2(torch.fft.rfft2(padded) * bank_f, s=(side, side))
        resp = full[:, :, r_pad : r_pad + SIZE, r_pad : r_pad + SIZE]
        r = (1 - w) * resp.gather(1, i0[:, None]) + w * resp.gather(1, i1[:, None])
        r = r / (r.flatten(1).std(dim=1)[:, None, None, None] + 1e-6)
        x = torch.tanh(2.0 * r)
    return x[:, 0]


def _render(field: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """One impression per master: distortion, contact region, pressure, noise → uint8."""

    n = field.shape[0]
    dev = field.device

    def u(lo: float, hi: float) -> torch.Tensor:
        return torch.rand(n, generator=gen, device=dev) * (hi - lo) + lo

    lin = torch.linspace(-1, 1, SIZE, device=dev)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    angle = u(-0.35, 0.35)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 1] = torch.cos(angle), -torch.sin(angle)
    theta[:, 1, 0], theta[:, 1, 1] = torch.sin(angle), torch.cos(angle)
    theta[:, :, 2] = torch.stack([u(-0.08, 0.08), u(-0.08, 0.08)], dim=1)
    grid = F.affine_grid(theta, [n, 1, SIZE, SIZE], align_corners=False)
    radial = (xx ** 2 + yy ** 2)[None, :, :, None]  # distortion grows away from the centre
    grid = grid + 0.04 * _smooth_field(n, 2, 4, SIZE, gen, dev).permute(0, 2, 3, 1) * radial
    warped = F.grid_sample(field[:, None], grid, mode="bilinear", padding_mode="reflection",
                           align_corners=False)[:, 0]
    ax, ay = u(0.55, 0.8)[:, None, None], u(0.7, 0.95)[:, None, None]
    ox, oy = u(-0.1, 0.1)[:, None, None], u(-0.1, 0.15)[:, None, None]
    edge = 1 - ((xx - ox) / ax) ** 2 - ((yy - oy) / ay) ** 2
    edge = edge + 0.15 * _smooth_field(n, 1, 6, SIZE, gen, dev)[:, 0]
    contact = torch.sigmoid(edge * 25)
    pressure = u(-0.35, 0.35)[:, None, None] + 0.25 * _smooth_field(n, 1, 4, SIZE, gen, dev)[:, 0]
    ink = torch.sigmoid((warped + pressure) * u(3, 8)[:, None, None])  # 1 = ridge
    pores = (torch.rand(n, SIZE, SIZE, generator=gen, device=dev) < 0.004).float()
    pores = F.max_pool2d(pores[:, None], 3, stride=1, padding=1)[:, 0] * (ink > 0.7)
    ink = ink * (1 - 0.7 * pores)
    for _ in range(3):  # creases: thin bright lines
        phi = u(0, math.pi)[:, None, None]
        off = u(-0.8, 0.8)[:, None, None]
        line = ((xx * torch.cos(phi) + yy * torch.sin(phi) - off).abs() < 0.006)
        chosen = torch.rand(n, 1, 1, generator=gen, device=dev) < 0.4
        ink = torch.where(line & chosen, 0.1 * ink, ink)
    ink = ink * contact
    dark, light = u(10, 80)[:, None, None], u(170, 250)[:, None, None]
    image = light - (light - dark) * ink
    image = image + torch.randn(n, SIZE, SIZE, generator=gen, device=dev) * u(2, 12)[:, None, None]
    return image.clamp(0, 255).round().to(torch.uint8)


def generate(out: Path, identities: int, impressions: int, seed: int, batch: int = 32,
             device: str | None = None, start_id: int = 0) -> None:
    """Write ``identities × impressions`` PNGs named ``{identity:06d}_{impression}.png``."""

    out.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gen = torch.Generator(device=device_t).manual_seed(seed)
    for start in range(start_id, start_id + identities, batch):
        n = min(batch, start_id + identities - start)
        with torch.no_grad():
            theta = _orientation(n, gen, device_t)
            period = torch.rand(n, generator=gen, device=device_t) * 3.5 + 7.5
            field = _grow(theta, period, gen)
            for imp in range(impressions):
                images = _render(field, gen).cpu().numpy()
                for k in range(n):
                    Image.fromarray(images[k]).save(out / f"{start + k:06d}_{imp + 1}.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identities", type=int, default=20000)
    parser.add_argument("--impressions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda | cpu (default: cuda if present)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 = torch default)")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    generate(args.out, args.identities, args.impressions, args.seed, args.batch, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
