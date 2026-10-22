"""GitHub social preview (1280 x 640): wordmark, tagline, ridge-field background.

The SVG is static (no animation) so it rasterises identically everywhere; the
PNG is rendered with ``resvg-py`` when it is installed (``pip install resvg-py``).
"""

from __future__ import annotations

from pathlib import Path

from make_brand_core import (
    FONT_MONO,
    FONT_SANS,
    INDIGO,
    RIDGES,
    SLATE_400,
    TEAL,
    TEAL_LIGHT,
    gradient,
    ridges,
)
from make_logo_assets import logo_svg

WIDTH, HEIGHT = 1280, 640
FIELD_RIDGES = (
    *RIDGES,
    *((41.0 + 5 * k, ((140 - 2 * k, 290 - 4 * k), (304 - 4 * k, 436 + k))) for k in range(1, 9)),
)
STEPS = ("hand landmarks", "upright crops", "embedding", "cosine match")


def social_svg() -> str:
    """Static 1280 x 640 social card."""

    scale, cx, cy = 5.0, 1010.0, 330.0
    field = "".join(
        f'<path d="{r.path(scale, cx - 64 * scale, cy - 64 * scale)}"/>'
        for r in ridges(FIELD_RIDGES, step_deg=2.0)
    )
    mark = logo_svg().split("</defs>", 1)[1].replace("</svg>\n", "")
    chips = []
    x = 96.0
    for index, step in enumerate(STEPS):
        width = 22 + 11.4 * len(step)
        chips.append(
            f'<rect x="{x:g}" y="444" width="{width:.0f}" height="38" rx="19" fill="#ffffff" '
            f'fill-opacity=".04" stroke="{TEAL_LIGHT}" stroke-opacity=".35"/>'
            f'<text x="{x + width / 2:.1f}" y="469" text-anchor="middle">{step}</text>'
        )
        x += width
        if index < len(STEPS) - 1:
            chips.append(
                f'<path d="M{x + 10:g} 463h16m-5-5 5 5-5 5" fill="none" stroke="{SLATE_400}" '
                'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            )
            x += 36
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t">'
        "<title id=\"t\">fingerprint-from-image — learned fingerprint recognition</title><defs>"
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#0B1120"/><stop offset="1" stop-color="#131D3A"/>'
        f"</linearGradient>{gradient('g')}{gradient('fg', a=TEAL, b=INDIGO)}"
        '<radialGradient id="fade" cx="1010" cy="330" r="420" gradientUnits="userSpaceOnUse">'
        '<stop offset="0" stop-color="#fff"/>'
        '<stop offset=".75" stop-color="#fff" stop-opacity=".5"/>'
        '<stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>'
        f'<mask id="m"><rect width="{WIDTH}" height="{HEIGHT}" fill="url(#fade)"/></mask>'
        "</defs>"
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="url(#bg)"/>'
        f'<g mask="url(#m)" fill="none" stroke="url(#fg)" stroke-width="7" stroke-linecap="round" '
        f'opacity=".22">{field}</g>'
        f'<g transform="translate(96 112) scale(.9)">{mark}</g>'
        f'<text x="92" y="330" font-family="{FONT_SANS}" font-size="72" font-weight="800" '
        'fill="#F8FAFC" letter-spacing="-1.5">fingerprint-from-image</text>'
        f'<text x="96" y="392" font-family="{FONT_SANS}" font-size="31" font-weight="500" '
        'fill="#CBD5E1">Learned fingerprint recognition from photos &amp; scans '
        f'<tspan fill="{TEAL_LIGHT}" font-weight="700">· ONNX · TFLite</tspan></text>'
        f'<g font-family="{FONT_MONO}" font-size="19" font-weight="600" fill="#E2E8F0">'
        f'{"".join(chips)}</g>'
        f'<text x="96" y="566" font-family="{FONT_MONO}" font-size="20" fill="{SLATE_400}">'
        "github.com/m-mutti/fingerprint-from-image</text>"
        "</svg>\n"
    )


def render_png(svg_text: str, out: Path) -> bool:
    """Rasterise with resvg-py; return False (and skip) when it is unavailable."""

    try:
        import resvg_py
    except ImportError:
        return False
    out.write_bytes(bytes(resvg_py.svg_to_bytes(svg_string=svg_text)))
    return True
