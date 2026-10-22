"""Regenerate every brand asset in ``assets/`` deterministically.

SVGs are pure standard-library output. Raster outputs use optional tools and
are skipped with a notice when those are missing:

* ``social-preview.png`` needs ``resvg-py``;
* ``pipeline-preview.gif`` needs ``playwright`` and a Chromium-family browser.

::

    python scripts/make_brand_assets.py                 # everything available
    python scripts/make_brand_assets.py --svg-only      # SVGs only (no extra deps)
    python scripts/make_brand_assets.py --output-dir /tmp/brand
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_logo_assets import (  # noqa: E402
    animated_logo_svg,
    logo_small_svg,
    logo_svg,
    wordmark_svg,
)
from make_pipeline_animation import LOOP, pipeline_svg  # noqa: E402
from make_social_preview import render_png, social_svg  # noqa: E402

# Byte budgets enforced by tests/test_brand_assets.py.
SIZE_LIMITS = {
    "pipeline-animated.svg": 150_000,
    "pipeline-animated-light.svg": 150_000,
    "logo-animated.svg": 60_000,
    "social-preview.svg": 150_000,
    "logo.svg": 20_000,
    "logo-small.svg": 10_000,
    "wordmark-light.svg": 20_000,
    "wordmark-dark.svg": 20_000,
}
GIF_LIMIT = 3_000_000
ANIMATED = ("pipeline-animated.svg", "pipeline-animated-light.svg", "logo-animated.svg")

# Backwards-compatible alias for the previous generator API.
animated_svg = animated_logo_svg


def build_svgs() -> dict[str, str]:
    """Return ``{file name: SVG text}`` for every vector asset."""

    return {
        "logo.svg": logo_svg(),
        "logo-small.svg": logo_small_svg(),
        "wordmark-light.svg": wordmark_svg(dark=False),
        "wordmark-dark.svg": wordmark_svg(dark=True),
        "logo-animated.svg": animated_logo_svg(),
        "pipeline-animated.svg": pipeline_svg("dark"),
        "pipeline-animated-light.svg": pipeline_svg("light"),
        "social-preview.svg": social_svg(),
    }


def write_svgs(output_dir: Path) -> list[Path]:
    """Write all SVGs (LF line endings) and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in build_svgs().items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def write_rasters(output_dir: Path, *, gif: bool = True, channel: str | None = "chrome") -> None:
    """Render the social PNG and the pipeline GIF when their tools exist."""

    social = output_dir / "social-preview.png"
    if render_png((output_dir / "social-preview.svg").read_text(encoding="utf-8"), social):
        print(f"wrote {social}")
    else:
        print("skip social-preview.png (pip install resvg-py)")
    if not gif:
        return
    try:
        from make_preview_frames import make_gif

        out = make_gif(
            output_dir / "pipeline-animated.svg",
            output_dir / "pipeline-preview.gif",
            duration=LOOP,
            fps=12.5,
            width=800,
            channel=channel,
        )
    except ImportError:
        print("skip pipeline-preview.gif (pip install playwright)")
        return
    size = out.stat().st_size
    print(f"wrote {out} ({size / 1e6:.2f} MB)")
    if size > GIF_LIMIT:
        raise SystemExit(f"{out} exceeds {GIF_LIMIT} bytes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=Path("assets"), type=Path)
    parser.add_argument("--svg-only", action="store_true", help="skip PNG/GIF rendering")
    parser.add_argument("--no-gif", action="store_true", help="skip the headless-browser GIF")
    parser.add_argument("--channel", default="chrome", help="browser channel; '' = bundled")
    args = parser.parse_args()
    for path in write_svgs(args.output_dir):
        print(f"wrote {path} ({path.stat().st_size / 1000:.1f} kB)")
    if not args.svg_only:
        write_rasters(args.output_dir, gif=not args.no_gif, channel=args.channel or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
