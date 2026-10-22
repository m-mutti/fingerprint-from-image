"""Generate the project logo, wordmark, and animated banner as SVG files.

The fingerprint mark is built from deterministic elliptical ridge arcs so the
artwork can be regenerated and tweaked without a vector editor::

    python scripts/make_brand_assets.py --output-dir assets
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

TEAL = "#14B8A6"
INDIGO = "#6366F1"
INK = "#0F172A"
PAPER = "#F8FAFC"

# (radius, [(start_deg, end_deg), ...]) — angles measured clockwise from +x in
# SVG coordinates. Gaps between segments mimic ridge endings and bifurcations.
RIDGES: tuple[tuple[float, tuple[tuple[float, float], ...]], ...] = (
    (6.0, ((200, 520),)),
    (11.0, ((185, 300), (312, 470))),
    (16.0, ((170, 255), (268, 395), (405, 465))),
    (21.0, ((160, 330), (342, 452))),
    (26.0, ((152, 228), (240, 372), (384, 440))),
    (31.0, ((148, 300), (314, 432))),
    (36.0, ((150, 212), (224, 350), (362, 424))),
    (41.0, ((160, 282), (296, 412))),
)
CENTER = (64.0, 64.0)
ROTATION = -25.0  # turns the ridge-ending gap toward the bottom, like a fingertip
ASPECT = 1.18  # fingertips are taller than they are wide


def _arc_path(radius: float, start: float, end: float) -> str:
    """Polyline approximation of an elliptical arc, rounded for compact output."""

    cx, cy = CENTER
    steps = max(6, int(abs(end - start) / 6))
    points = []
    for index in range(steps + 1):
        angle = math.radians(ROTATION + start + (end - start) * index / steps)
        points.append(
            (cx + radius * math.cos(angle), cy + ASPECT * radius * math.sin(angle))
        )
    head = f"M{points[0][0]:.1f} {points[0][1]:.1f}"
    return head + "".join(f"L{x:.1f} {y:.1f}" for x, y in points[1:])


def _ridge_paths() -> list[tuple[str, float]]:
    """Return ``(path_data, approximate_length)`` for every ridge segment."""

    paths = []
    for radius, segments in RIDGES:
        for start, end in segments:
            length = math.radians(end - start) * radius * (1 + ASPECT) / 2
            paths.append((_arc_path(radius, start, end), length))
    return paths


def _viewfinder(stroke: str, width: float = 5.0) -> str:
    """Camera viewfinder corners: the "from image" half of the mark."""

    corner = 22
    lo, hi = 8, 120
    return (
        f'<g fill="none" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M{lo} {lo + corner}V{lo + 6}Q{lo} {lo} {lo + 6} {lo}H{lo + corner}"/>'
        f'<path d="M{hi - corner} {lo}H{hi - 6}Q{hi} {lo} {hi} {lo + 6}V{lo + corner}"/>'
        f'<path d="M{hi} {hi - corner}V{hi - 6}Q{hi} {hi} {hi - 6} {hi}H{hi - corner}"/>'
        f'<path d="M{lo + corner} {hi}H{lo + 6}Q{lo} {hi} {lo} {hi - 6}V{hi - corner}"/>'
        "</g>"
    )


def _gradient(identifier: str) -> str:
    return (
        f'<linearGradient id="{identifier}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{TEAL}"/><stop offset="1" stop-color="{INDIGO}"/>'
        "</linearGradient>"
    )


def logo_svg() -> str:
    ridges = "".join(f'<path d="{d}"/>' for d, _ in _ridge_paths())
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" '
        'width="128" height="128" role="img" aria-labelledby="t">'
        "<title id=\"t\">fingerprint-from-image logo</title>"
        f"<defs>{_gradient('g')}</defs>"
        f'<g fill="none" stroke="url(#g)" stroke-width="3.6" stroke-linecap="round">'
        f"{ridges}</g>"
        f"{_viewfinder('url(#g)')}"
        "</svg>\n"
    )


def wordmark_svg(*, dark: bool) -> str:
    text = PAPER if dark else INK
    muted = "#94A3B8" if dark else "#475569"
    ridges = "".join(f'<path d="{d}"/>' for d, _ in _ridge_paths())
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 128" '
        'width="560" height="128" role="img" aria-labelledby="t">'
        "<title id=\"t\">fingerprint-from-image</title>"
        f"<defs>{_gradient('g')}</defs>"
        f'<g fill="none" stroke="url(#g)" stroke-width="3.6" stroke-linecap="round">'
        f"{ridges}</g>{_viewfinder('url(#g)')}"
        '<g font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">'
        f'<text x="146" y="66" font-size="40" font-weight="700" fill="{text}" '
        'letter-spacing="-0.5">fingerprint</text>'
        f'<text x="148" y="100" font-size="22" font-weight="500" fill="{muted}" '
        'letter-spacing="3">FROM · IMAGE</text>'
        "</g></svg>\n"
    )


def animated_svg() -> str:
    """Ridges draw in, a scan line sweeps, then the mark "locks" with a pulse.

    Uses CSS keyframes only, so it animates inside GitHub's ``<img>`` sandbox.
    """

    paths = []
    for index, (d, length) in enumerate(_ridge_paths()):
        dash = math.ceil(length) + 2
        delay = 0.08 * index
        paths.append(
            f'<path class="r" d="{d}" style="stroke-dasharray:{dash};'
            f'stroke-dashoffset:{dash};animation-delay:{delay:.2f}s"/>'
        )
    css = (
        "@keyframes draw{to{stroke-dashoffset:0}}"
        "@keyframes scan{0%{transform:translateY(0);opacity:0}"
        "10%{opacity:1}90%{opacity:1}100%{transform:translateY(104px);opacity:0}}"
        "@keyframes lock{0%,100%{opacity:.35}50%{opacity:1}}"
        ".r{animation:draw 1.1s cubic-bezier(.4,0,.2,1) forwards}"
        ".s{animation:scan 2.4s ease-in-out 1.6s infinite}"
        ".v{animation:lock 2.4s ease-in-out 1.6s infinite}"
        "@media (prefers-reduced-motion:reduce){.r{animation:none;stroke-dashoffset:0}"
        ".s{display:none}.v{animation:none;opacity:1}}"
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" '
        'width="256" height="256" role="img" aria-labelledby="t">'
        "<title id=\"t\">Animated fingerprint scan</title>"
        f"<defs>{_gradient('g')}"
        '<linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{TEAL}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{TEAL}" stop-opacity=".55"/></linearGradient>'
        f"</defs><style>{css}</style>"
        f'<g fill="none" stroke="url(#g)" stroke-width="3.6" stroke-linecap="round">'
        f"{''.join(paths)}</g>"
        f'<g class="v">{_viewfinder("url(#g)")}</g>'
        '<g class="s"><rect x="12" y="2" width="104" height="12" fill="url(#sg)"/>'
        f'<rect x="12" y="13" width="104" height="1.6" rx=".8" fill="{TEAL}"/></g>'
        "</svg>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="assets", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "logo.svg": logo_svg(),
        "wordmark-light.svg": wordmark_svg(dark=False),
        "wordmark-dark.svg": wordmark_svg(dark=True),
        "logo-animated.svg": animated_svg(),
    }
    for name, content in outputs.items():
        (args.output_dir / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {args.output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
