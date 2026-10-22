"""Render frozen frames of an animated SVG with headless Chrome, and build GIFs.

Requires the optional ``playwright`` package plus a Chromium-family browser
(``playwright install chromium``, or an installed Chrome/Edge via
``--channel chrome``). Animations are frozen through the Web Animations API, so
every frame is exact rather than a timing-dependent screenshot::

    python scripts/make_preview_frames.py assets/pipeline-animated.svg \
        --times 0.5 2 4 6 --out temp/frames
    python scripts/make_preview_frames.py assets/pipeline-animated.svg \
        --gif assets/pipeline-preview.gif --fps 12 --width 800
"""

from __future__ import annotations

import argparse
import io
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from pathlib import Path

from PIL import Image

ACCENTS = ("#F59E0B", "#2DD4BF", "#14B8A6", "#818CF8", "#6366F1", "#E2E8F0")
FREEZE_JS ="""(ms) => {
  for (const a of document.getAnimations()) { a.pause(); a.currentTime = ms; }
  return new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
}"""


def svg_size(path: Path) -> tuple[int, int]:
    """Intrinsic ``width``/``height`` of an SVG file in CSS pixels."""

    root = ET.parse(path).getroot()
    return int(float(root.get("width", "800"))), int(float(root.get("height", "400")))


def _flatten(png: bytes, background: str | None) -> Image.Image:
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    if background is None:
        return image
    canvas = Image.new("RGBA", image.size, background)
    canvas.alpha_composite(image)
    return canvas


def render_frames(
    svg: Path,
    times: Iterable[float],
    *,
    scale: float = 1.0,
    background: str | None = None,
    reduced_motion: bool = False,
    channel: str | None = "chrome",
) -> list[Image.Image]:
    """Return one RGBA image per time (seconds) with animations frozen there."""

    from playwright.sync_api import sync_playwright

    width, height = svg_size(svg)
    frames = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=channel) if channel else pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
            reduced_motion="reduce" if reduced_motion else "no-preference",
        )
        page.goto(svg.resolve().as_uri())
        for t in times:
            page.evaluate(FREEZE_JS, t * 1000.0)
            frames.append(_flatten(page.screenshot(omit_background=True), background))
        browser.close()
    return frames


def save_frames(
    svg: Path,
    times: Sequence[float],
    out_dir: Path,
    *,
    scale: float = 2.0,
    background: str | None = None,
    suffix: str = "",
    channel: str | None = "chrome",
) -> list[Path]:
    """Write ``<stem>-t<time>.png`` inspection frames and return their paths."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    images = render_frames(svg, times, scale=scale, background=background, channel=channel)
    for t, image in zip(times, images, strict=True):
        path = out_dir / f"{svg.stem}{suffix}-t{t:04.1f}.png"
        image.save(path)
        paths.append(path)
    return paths


def make_gif(
    svg: Path,
    out: Path,
    *,
    duration: float,
    fps: float = 12.0,
    width: int = 800,
    background: str | None = None,
    channel: str | None = "chrome",
) -> Path:
    """Render a seamless looping GIF with one shared palette (no flicker)."""

    count = round(duration * fps)
    times = [i / fps for i in range(count)]
    natural_width, _ = svg_size(svg)
    frames = render_frames(
        svg, times, scale=width / natural_width, background=background, channel=channel
    )
    rgb = [f.convert("RGB") for f in frames]
    # Build the palette from a mosaic of evenly spaced frames so every scene's
    # colours are represented, then map all frames onto it without dithering.
    # Small accents (e.g. amber minutiae) are seeded as swatches so median-cut
    # keeps them instead of folding them into neighbouring greys.
    sample = rgb[:: max(1, len(rgb) // 12)]
    swatch_h = rgb[0].height // 4
    mosaic = Image.new("RGB", (rgb[0].width, rgb[0].height * len(sample) + swatch_h))
    for index, frame in enumerate(sample):
        mosaic.paste(frame, (0, index * frame.height))
    step = rgb[0].width // len(ACCENTS)
    for index, colour in enumerate(ACCENTS):
        box = (index * step, mosaic.height - swatch_h, (index + 1) * step, mosaic.height)
        mosaic.paste(colour, box)
    palette = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in rgb]
    out.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        out,
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=1,
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("svg", type=Path)
    parser.add_argument("--times", type=float, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=Path("temp/frames"))
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--background", default=None, help="e.g. '#ffffff' or '#0d1117'")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--channel", default="chrome", help="'' for bundled Chromium")
    args = parser.parse_args()
    channel = args.channel or None
    if args.times:
        for path in save_frames(
            args.svg,
            args.times,
            args.out,
            scale=args.scale,
            background=args.background,
            suffix=args.suffix,
            channel=channel,
        ):
            print(f"wrote {path}")
    if args.gif:
        gif = make_gif(
            args.svg,
            args.gif,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            background=args.background,
            channel=channel,
        )
        print(f"wrote {gif} ({gif.stat().st_size / 1024:.0f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
