"""Logo, favicon-size logo, wordmarks, and the animated mark.

The animated mark tells a miniature version of the product story on a 7.2 s
loop: ridges draw on, a scan beam sweeps and *detects* minutiae as it passes
them, the viewfinder locks inward with a small overshoot, then the mark idles
with a breathing glow before fading out and restarting.
"""

from __future__ import annotations

import math

from make_brand_core import (
    AMBER,
    EASE_BACK,
    EASE_IN_OUT,
    EASE_OUT,
    FONT_SANS,
    INK,
    PAPER,
    REDUCED_MOTION,
    SLATE_400,
    SLATE_600,
    SMALL_RIDGES,
    TEAL,
    TEAL_GLOW,
    Timeline,
    gradient,
    minutiae,
    ridges,
    viewfinder,
    viewfinder_corners,
)

SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'


def _ridge_group(stroke: str = "url(#g)", width: float = 3.6) -> str:
    paths = "".join(f'<path d="{r.path()}"/>' for r in ridges())
    return (
        f'<g fill="none" stroke="{stroke}" stroke-width="{width:g}" stroke-linecap="round">'
        f"{paths}</g>"
    )


def logo_svg() -> str:
    """Static 128 px mark: ridges inside viewfinder corners."""

    return (
        f'<svg {SVG_NS} viewBox="0 0 128 128" width="128" height="128" role="img" '
        'aria-labelledby="t"><title id="t">fingerprint-from-image logo</title>'
        f"<defs>{gradient('g')}</defs>{_ridge_group()}{viewfinder('url(#g)')}</svg>\n"
    )


def logo_small_svg() -> str:
    """Favicon-size mark (16–32 px): four thick ridges, heavier corners."""

    paths = "".join(f'<path d="{r.path()}"/>' for r in ridges(SMALL_RIDGES))
    return (
        f'<svg {SVG_NS} viewBox="0 0 128 128" width="32" height="32" role="img" '
        'aria-labelledby="t"><title id="t">fingerprint-from-image icon</title>'
        f"<defs>{gradient('g')}</defs>"
        f'<g fill="none" stroke="url(#g)" stroke-width="9.5" stroke-linecap="round">{paths}</g>'
        f"{viewfinder('url(#g)', 11, lo=7, hi=121)}</svg>\n"
    )


def wordmark_svg(*, dark: bool) -> str:
    """Horizontal lockup: mark + "fingerprint / FROM · IMAGE"."""

    text = PAPER if dark else INK
    muted = SLATE_400 if dark else SLATE_600
    return (
        f'<svg {SVG_NS} viewBox="0 0 560 128" width="560" height="128" role="img" '
        'aria-labelledby="t"><title id="t">fingerprint-from-image</title>'
        f"<defs>{gradient('g')}</defs>{_ridge_group()}{viewfinder('url(#g)')}"
        f'<g font-family="{FONT_SANS}">'
        f'<text x="146" y="66" font-size="40" font-weight="700" fill="{text}" '
        'letter-spacing="-0.5">fingerprint</text>'
        f'<text x="148" y="100" font-size="22" font-weight="500" fill="{muted}" '
        'letter-spacing="3">FROM · IMAGE</text>'
        "</g></svg>\n"
    )


# --- animated mark -----------------------------------------------------------

LOOP = 7.2
SCAN_START, SCAN_END = 1.55, 3.0
SCAN_Y0, SCAN_Y1 = 12.0, 116.0
LOCK = 3.0
FADE_OUT = 6.6


def _animated_ridges(tl: Timeline) -> str:
    out = []
    for index, ridge in enumerate(ridges()):
        start = 0.05 + 0.065 * index
        frames = [
            (0.0, "stroke-dashoffset:100;opacity:1"),
            (start, "stroke-dashoffset:100;opacity:1", EASE_OUT),
            (start + 0.85, "stroke-dashoffset:0;opacity:1"),
            (FADE_OUT, "stroke-dashoffset:0;opacity:1", EASE_IN_OUT),
            (FADE_OUT + 0.4, "stroke-dashoffset:0;opacity:0"),
        ]
        out.append(
            f'<path d="{ridge.path()}" pathLength="100" stroke-dasharray="100 101" '
            f'style="{tl.style(frames)}"/>'
        )
    return (
        '<g fill="none" stroke="url(#g)" stroke-width="3.6" stroke-linecap="round">'
        + "".join(out)
        + "</g>"
    )


def _glow(tl: Timeline) -> str:
    """Blurred copy of the ridges whose opacity breathes during the idle phase."""

    frames = [
        (0.0, "opacity:0"),
        (LOCK + 0.2, "opacity:0", EASE_OUT),
        (LOCK + 0.5, "opacity:.7", EASE_IN_OUT),
        (4.6, "opacity:.18", EASE_IN_OUT),
        (5.6, "opacity:.6", EASE_IN_OUT),
        (FADE_OUT, "opacity:.2", EASE_IN_OUT),
        (FADE_OUT + 0.4, "opacity:0"),
    ]
    paths = "".join(f'<path d="{r.path()}"/>' for r in ridges())
    return (
        f'<g opacity="0" filter="url(#blur)" fill="none" stroke="{TEAL_GLOW}" stroke-width="4" '
        f'stroke-linecap="round" style="{tl.style(frames)}">{paths}</g>'
    )


def _minutiae(tl: Timeline) -> str:
    out = []
    span = SCAN_END - SCAN_START
    for m in sorted(minutiae(10), key=lambda p: p.y):
        hit = SCAN_START + span * (m.y - SCAN_Y0) / (SCAN_Y1 - SCAN_Y0)
        dx = 5.5 * math.cos(math.radians(m.angle))
        dy = 5.5 * math.sin(math.radians(m.angle))
        pop = [
            (0.0, "opacity:0;transform:scale(0)"),
            (hit, "opacity:0;transform:scale(0)", EASE_BACK),
            (hit + 0.3, "opacity:1;transform:scale(1)"),
            (4.2, "opacity:1;transform:scale(1)", EASE_IN_OUT),
            (4.7, "opacity:0;transform:scale(.6)"),
        ]
        ring = [
            (0.0, "opacity:0;transform:scale(.4)"),
            (hit - 0.01, "opacity:0;transform:scale(.4)"),
            (hit, "opacity:.9;transform:scale(.4)", EASE_OUT),
            (hit + 0.6, "opacity:0;transform:scale(2.4)"),
        ]
        out.append(
            f'<g class="c" opacity="0" style="{tl.style(pop)}">'
            f'<circle cx="{m.x:.1f}" cy="{m.y:.1f}" r="2.4"/>'
            f'<path d="M{m.x:.1f} {m.y:.1f}l{dx:.1f} {dy:.1f}"/></g>'
            f'<circle class="c" cx="{m.x:.1f}" cy="{m.y:.1f}" r="3" fill="none" '
            f'stroke-width=".9" opacity="0" style="{tl.style(ring)}"/>'
        )
    return (
        f'<g fill="none" stroke="{AMBER}" stroke-width="1.5" stroke-linecap="round">'
        + "".join(out)
        + "</g>"
    )


def _scan(tl: Timeline) -> str:
    travel = SCAN_Y1 - SCAN_Y0
    frames = [
        (0.0, "opacity:0;transform:translateY(0)"),
        (SCAN_START - 0.1, "opacity:0;transform:translateY(0)", EASE_OUT),
        (SCAN_START + 0.1, "opacity:1;transform:translateY(0)", "linear"),
        (SCAN_END - 0.1, f"opacity:1;transform:translateY({travel - 6:g}px)", EASE_OUT),
        (SCAN_END + 0.1, f"opacity:0;transform:translateY({travel:g}px)"),
    ]
    return (
        f'<g opacity="0" style="{tl.style(frames)}">'
        f'<rect x="10" y="{SCAN_Y0 - 14:g}" width="108" height="14" fill="url(#sg)"/>'
        f'<rect x="10" y="{SCAN_Y0 - 1:g}" width="108" height="1.8" rx=".9" fill="{TEAL_GLOW}"/>'
        "</g>"
    )


def _viewfinder_lock(tl: Timeline) -> str:
    offsets = ((-5, -5), (5, -5), (5, 5), (-5, 5))
    out = []
    for path, (ox, oy) in zip(viewfinder_corners("url(#g)"), offsets, strict=True):
        far = f"transform:translate({ox}px,{oy}px);opacity:.3"
        frames = [
            (0.0, far),
            (LOCK, far, EASE_BACK),
            (LOCK + 0.45, "transform:translate(0,0);opacity:1"),
            (FADE_OUT, "transform:translate(0,0);opacity:1", EASE_IN_OUT),
            (LOCK + 4.0, far),
        ]
        out.append(path.replace("<path ", f'<path style="{tl.style(frames)}" ', 1))
    return (
        '<g fill="none" stroke="url(#g)" stroke-width="5" stroke-linecap="round" '
        'stroke-linejoin="round">' + "".join(out) + "</g>"
    )


def animated_logo_svg() -> str:
    """256 px animated mark (CSS keyframes only, works inside GitHub ``<img>``)."""

    tl = Timeline(LOOP, "m")
    bounce = [
        (0.0, "transform:scale(1)"),
        (LOCK + 0.1, "transform:scale(1)", EASE_OUT),
        (LOCK + 0.3, "transform:scale(1.035)", EASE_IN_OUT),
        (LOCK + 0.65, "transform:scale(1)"),
    ]
    body = (
        f'<g class="c" style="{tl.style(bounce)}">'
        f"{_glow(tl)}{_animated_ridges(tl)}{_minutiae(tl)}{_viewfinder_lock(tl)}</g>"
        f"{_scan(tl)}"
    )
    css = (
        ".c{transform-box:fill-box;transform-origin:center}"
        f"{tl.css()}{REDUCED_MOTION}"
    )
    return (
        f'<svg {SVG_NS} viewBox="-6 -6 140 140" width="256" height="256" role="img" '
        'aria-labelledby="t"><title id="t">Animated fingerprint scan</title>'
        f"<defs>{gradient('g')}"
        '<linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{TEAL}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{TEAL}" stop-opacity=".45"/></linearGradient>'
        '<filter id="blur" x="-20%" y="-20%" width="140%" height="140%">'
        '<feGaussianBlur stdDeviation="2.6"/></filter>'
        f"</defs><style>{css}</style>{body}</svg>\n"
    )
