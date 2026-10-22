"""Hero animation: hand photo -> landmarks -> fingertip crops -> embedding -> match.

An 8 s CSS-keyframe loop on a self-contained rounded card (dark and light
themes). Every element's *static* styling is the final frame, so with
``prefers-reduced-motion: reduce`` (or in renderers without CSS animation) the
card shows the complete pipeline at rest.
"""

from __future__ import annotations

from make_brand_core import (
    AMBER,
    EASE_IN_OUT,
    EASE_OUT,
    FONT_MONO,
    FONT_SANS,
    INDIGO,
    REDUCED_MOTION,
    RIDGES,
    TEAL,
    TEAL_LIGHT,
    Ridge,
    Timeline,
    gradient,
    minutiae,
    ridges,
)
from make_logo_assets import logo_svg
from make_pipeline_hand import HandLayout, hand_panel
from make_pipeline_match import (
    ENCODE_START,
    MatchLayout,
    embeddings,
    encoder,
    gallery_rows,
    network_nodes,
    query_bar,
)

LOOP = 8.0
WIDTH, HEIGHT = 960, 470
FADE_OUT = 7.5
ZOOM, SCAN_START, SCAN_END = 2.2, 2.75, 3.3

THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "bg0": "#0B1120", "bg1": "#111B30", "border": "#22304A", "panel": "#0E1729",
        "panel_line": "#1C2A44", "text": "#E2E8F0", "muted": "#94A3B8", "faint": "#2A3A57",
        "hand_fill": "#15213A", "hand_line": "#3B5277", "neutral": "#1B2740",
        "track": "#1C2A44", "crop": "#0A1222", "raw": "#64748B", "on_accent": "#042F2E",
        "accent": "#2DD4BF",
    },
    "light": {
        "bg0": "#FFFFFF", "bg1": "#F3F6FB", "border": "#D9E1EC", "panel": "#F8FAFC",
        "panel_line": "#E3E9F2", "text": "#0F172A", "muted": "#526075", "faint": "#C7D2E1",
        "hand_fill": "#EEF3FA", "hand_line": "#9FB0C8", "neutral": "#E6ECF4",
        "track": "#E3E9F2", "crop": "#F1F5FA", "raw": "#94A3B8", "on_accent": "#042F2E",
        "accent": "#0D9488",
    },
}

# Columns: (x0, x1) for hand, crop, encoder, embedding/gallery.
COLS = ((20, 262), (282, 462), (482, 592), (612, 940))
PANEL_Y0, PANEL_Y1 = 92, 400
CROP = (300.0, 140.0, 144.0, 192.0)  # x, y, w, h
HAND = HandLayout(x=23, y=100, scale=0.97)
MATCH = MatchLayout(
    net_x=(502, 537, 572), net_cy=246, bar_x=628, query_y=128, row_y0=200, row_step=48,
    score_x=926,
)
# Crop ridges: the logo table extended with outer rings so the print fills 3:4.
CROP_RIDGES = (
    *RIDGES,
    (46.0, ((150, 268), (282, 420))),
    (51.0, ((142, 300), (314, 428))),
    (56.0, ((138, 250), (264, 434))),
    (61.0, ((134, 300), (312, 438))),
    (66.0, ((130, 442),)),
)
CROP_SCALE = 1.62


def _crop_ridges() -> list[Ridge]:
    return ridges(CROP_RIDGES, step_deg=4.0)


def _crop_offset() -> tuple[float, float]:
    x, y, w, h = CROP
    return x + w / 2 - 64 * CROP_SCALE, y + h * 0.53 - 64 * CROP_SCALE


def _crop_panel(tl: Timeline, c: dict[str, str]) -> str:
    """Zoom from the index fingertip, raw -> enhanced reveal, scan, minutiae."""

    x, y, w, h = CROP
    dx, dy = _crop_offset()
    box = HAND.boxes()[1]  # index finger
    sx, sy = box.w / w, box.h / h
    tx, ty = box.cx - (x + w / 2), box.cy - (y + h / 2)
    origin = f"transform-origin:{x + w / 2:g}px {y + h / 2:g}px"
    at_tip = f"transform:translate({tx:.1f}px,{ty:.1f}px) scale({sx:.3f},{sy:.3f})"
    zoom = [
        (0.0, f"opacity:0;{at_tip}"),
        (ZOOM, f"opacity:0;{at_tip}", EASE_OUT),
        (ZOOM + 0.08, f"opacity:1;{at_tip}", EASE_IN_OUT),
        (ZOOM + 0.55, "opacity:1;transform:translate(0,0) scale(1,1)"),
    ]
    # Raw layer: visible until the scan has passed; hidden in the static frame.
    raw_on = [(0.0, "opacity:1"), (SCAN_END, "opacity:1"), (SCAN_END + 0.05, "opacity:0")]
    paths = "".join(f'<path d="{r.path(CROP_SCALE, dx, dy)}"/>' for r in _crop_ridges())
    travel = h + 24
    reveal = [
        (0.0, f"transform:translateY({-travel:g}px)"),
        (SCAN_START, f"transform:translateY({-travel:g}px)", "linear"),
        (SCAN_END, "transform:translateY(0)"),
    ]
    hide_raw = [
        (0.0, "transform:translateY(0)"),
        (SCAN_START, "transform:translateY(0)", "linear"),
        (SCAN_END, f"transform:translateY({travel:g}px)"),
    ]
    scan = [
        (0.0, "opacity:0;transform:translateY(0)"),
        (SCAN_START - 0.05, "opacity:0;transform:translateY(0)", "linear"),
        (SCAN_START + 0.05, f"opacity:1;transform:translateY({travel * 0.09:.1f}px)", "linear"),
        (SCAN_END - 0.05, f"opacity:1;transform:translateY({travel * 0.91:.1f}px)", "linear"),
        (SCAN_END + 0.05, f"opacity:0;transform:translateY({travel:g}px)"),
    ]
    marks = []
    for m in minutiae(9):
        mx, my = m.x * CROP_SCALE + dx, m.y * CROP_SCALE + dy
        hit = SCAN_START + (SCAN_END - SCAN_START) * (my - y + 12) / travel
        pop = [
            (0.0, "opacity:0;transform:scale(0)"),
            (hit, "opacity:0;transform:scale(0)", "cubic-bezier(.34,1.56,.64,1)"),
            (hit + 0.25, "opacity:1;transform:scale(1)"),
        ]
        marks.append(
            f'<circle class="c" cx="{mx:.1f}" cy="{my:.1f}" r="3.6" style="{tl.style(pop)}"/>'
        )
    text_raw = [(0.0, "opacity:0"), (ZOOM + 0.35, "opacity:0"), (ZOOM + 0.55, "opacity:1"),
                (SCAN_END - 0.2, "opacity:1"), (SCAN_END - 0.05, "opacity:0")]
    text_enh = [(0.0, "opacity:0"), (SCAN_END - 0.05, "opacity:0"), (SCAN_END + 0.12, "opacity:1")]
    status_y = y + h + 26
    return (
        f'<defs><clipPath id="cc"><rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="10"/>'
        f'</clipPath><mask id="mr" maskUnits="userSpaceOnUse" x="{x - 4:g}" y="{y - 4:g}" '
        f'width="{w + 8:g}" height="{h + 8:g}">'
        f'<rect x="{x - 4:g}" y="{y - 4:g}" width="{w + 8:g}" height="{h + 24:g}" fill="url(#rv)" '
        f'style="{tl.style(reveal)}"/></mask>'
        f'<mask id="mh" maskUnits="userSpaceOnUse" x="{x - 4:g}" y="{y - 4:g}" '
        f'width="{w + 8:g}" height="{h + 8:g}">'
        f'<rect x="{x - 4:g}" y="{y - 4:g}" width="{w + 8:g}" height="{h + 24:g}" fill="url(#rh)" '
        f'style="{tl.style(hide_raw)}"/></mask></defs>'
        f'<g style="{tl.style(zoom)};transform-box:view-box;{origin}">'
        f'<g clip-path="url(#cc)"><rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
        f'fill="{c["crop"]}"/>'
        f'<g mask="url(#mh)" opacity="0" style="{tl.style(raw_on)}">'
        f'<g filter="url(#soft)" fill="none" stroke="{c["raw"]}" '
        f'stroke-width="4.2" stroke-linecap="round" opacity=".95">{paths}</g>'
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" filter="url(#grain)" '
        'opacity=".2"/></g>'
        f'<g mask="url(#mr)" fill="none" stroke="url(#cg)" stroke-width="3.4" '
        f'stroke-linecap="round">{paths}</g>'
        f'<g fill="none" stroke="{AMBER}" stroke-width="1.8">{"".join(marks)}</g>'
        f'<g opacity="0" style="{tl.style(scan)}">'
        f'<rect x="{x:g}" y="{y - 22:g}" width="{w:g}" height="22" fill="url(#sg)"/>'
        f'<rect x="{x:g}" y="{y - 1.5:g}" width="{w:g}" height="2" fill="{TEAL_LIGHT}"/></g></g>'
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="10" fill="none" '
        f'stroke="{TEAL_LIGHT}" stroke-width="1.5"/></g>'
        f'<g font-family="{FONT_MONO}" font-size="12.5" font-weight="600" text-anchor="middle">'
        f'<text x="{x + w / 2:g}" y="{status_y:g}" fill="{c["muted"]}" opacity="0" '
        f'style="{tl.style(text_raw)}">raw · low contrast</text>'
        f'<text x="{x + w / 2:g}" y="{status_y:g}" fill="{c["accent"]}" '
        f'style="{tl.style(text_enh)}">enhanced · 9 minutiae</text></g>'
    )


def _callout(tl: Timeline) -> str:
    """Dashed zoom lines from the index fingertip box to the crop frame."""

    x, y, w, h = CROP
    corners = HAND.boxes()[1].corners()
    right = sorted(corners, key=lambda p: p[0])[2:]
    top, bottom = sorted(right, key=lambda p: p[1])
    frames = [(0.0, "opacity:0"), (ZOOM + 0.2, "opacity:0", EASE_OUT), (ZOOM + 0.6, "opacity:.5")]
    return (
        f'<g fill="none" stroke="{TEAL_LIGHT}" stroke-width="1.2" stroke-dasharray="3 4" '
        f'style="{tl.style(frames)}">'
        f'<path d="M{top[0]:.1f} {top[1]:.1f}L{x:g} {y + 10:g}"/>'
        f'<path d="M{bottom[0]:.1f} {bottom[1]:.1f}L{x:g} {y + h - 10:g}"/></g>'
    )


def _particles(tl: Timeline) -> str:
    """Ridge features streaming from the crop into the encoder's input layer."""

    x, y, w, h = CROP
    out = []
    for i, (nx, ny) in enumerate(network_nodes(MATCH)[0]):
        sx, sy = x + w - 6, y + h * (0.2 + 0.12 * i)
        t0 = ENCODE_START - 0.32 + 0.03 * i
        frames = [
            (0.0, "opacity:0;transform:translate(0,0)"),
            (t0, "opacity:0;transform:translate(0,0)", EASE_IN_OUT),
            (t0 + 0.06, "opacity:1;transform:translate(0,0)", EASE_IN_OUT),
            (t0 + 0.34, f"opacity:1;transform:translate({nx - sx:.1f}px,{ny - sy:.1f}px)"),
            (t0 + 0.4, f"opacity:0;transform:translate({nx - sx:.1f}px,{ny - sy:.1f}px)"),
        ]
        out.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" opacity="0" style="{tl.style(frames)}"/>'
        )
    return f'<g fill="{TEAL_LIGHT}">{"".join(out)}</g>'


RAIL = (("Detect", 0.1), ("Landmarks", 0.62), ("Fingertips", 1.45), ("Enhance", ZOOM),
        ("Embed", ENCODE_START), ("Match", 4.45))


def _rail(tl: Timeline, c: dict[str, str]) -> str:
    """Numbered progress rail along the bottom edge."""

    x0, x1, y = 40.0, 920.0, 428.0
    step = (x1 - x0) / (len(RAIL) - 1)
    progress = [(0.0, "transform:scaleX(0)")]
    progress += [(t, f"transform:scaleX({i / (len(RAIL) - 1):.3f})", EASE_IN_OUT)
                 for i, (_, t) in enumerate(RAIL)]
    progress += [(FADE_OUT, "transform:scaleX(1)", EASE_IN_OUT), (7.9, "transform:scaleX(0)")]
    parts = [
        f'<path d="M{x0:g} {y:g}H{x1:g}" stroke="{c["faint"]}" stroke-width="2"/>',
        f'<path class="l" d="M{x0:g} {y:g}H{x1:g}" stroke="{TEAL_LIGHT}" stroke-width="2.5" '
        f'style="{tl.style(progress)}"/>',
    ]
    for i, (name, t) in enumerate(RAIL):
        cx = x0 + step * i
        on = [(0.0, "opacity:0"), (t, "opacity:0", EASE_OUT), (t + 0.2, "opacity:1"),
              (FADE_OUT, "opacity:1", EASE_IN_OUT), (7.9, "opacity:0")]
        anchor = "start" if i == 0 else "end" if i == len(RAIL) - 1 else "middle"
        parts.append(
            f'<circle cx="{cx:g}" cy="{y:g}" r="5" fill="{c["bg1"]}" stroke="{c["faint"]}" '
            'stroke-width="2"/>'
            f'<text x="{cx:g}" y="{y + 25:g}" text-anchor="{anchor}" fill="{c["muted"]}">'
            f"{i + 1:02d} {name}</text>"
            f'<g style="{tl.style(on)}"><circle cx="{cx:g}" cy="{y:g}" r="5" fill="{TEAL_LIGHT}"/>'
            f'<text x="{cx:g}" y="{y + 25:g}" text-anchor="{anchor}" fill="{c["text"]}">'
            f"{i + 1:02d} {name}</text></g>"
        )
    return (
        f'<g font-family="{FONT_MONO}" font-size="12.5" font-weight="600" fill="none">'
        + "".join(parts) + "</g>"
    )


def _frame(c: dict[str, str]) -> str:
    """Card, header, column panels and captions (all static)."""

    mark = logo_svg().split("</defs>", 1)[1].replace("</svg>\n", "")
    captions = ("hand landmarks", "fingertip crop", "encoder", "embedding · cosine match")
    panels = "".join(
        f'<rect x="{a:g}" y="{PANEL_Y0:g}" width="{b - a:g}" height="{PANEL_Y1 - PANEL_Y0:g}" '
        f'rx="14" fill="{c["panel"]}" stroke="{c["panel_line"]}"/>'
        f'<text x="{a + 14:g}" y="{PANEL_Y0 - 10:g}">{cap.upper()}</text>'
        for (a, b), cap in zip(COLS, captions, strict=True)
    )
    return (
        f'<rect x=".5" y=".5" width="{WIDTH - 1}" height="{HEIGHT - 1}" rx="20" fill="url(#bg)" '
        f'stroke="{c["border"]}"/>'
        f'<g transform="translate(22 17) scale(.27)">{mark}</g>'
        f'<text x="66" y="36" font-family="{FONT_SANS}" font-size="19" font-weight="700" '
        f'fill="{c["text"]}" letter-spacing="-.2">fingerprint-from-image</text>'
        f'<text x="{WIDTH - 22}" y="36" font-family="{FONT_SANS}" font-size="14" text-anchor="end" '
        f'fill="{c["muted"]}">learned pipeline · hand photo → identity</text>'
        f'<g font-family="{FONT_MONO}" font-size="12" font-weight="700" letter-spacing="1.1" '
        f'fill="{c["muted"]}">{panels}</g>'
    )


def pipeline_svg(theme: str = "dark") -> str:
    """Render the hero animation for ``theme`` in {"dark", "light"}."""

    c = THEMES[theme]
    tl = Timeline(LOOP, "p")
    q, gallery = embeddings()
    fade = [(0.0, "opacity:1"), (FADE_OUT, "opacity:1", EASE_IN_OUT), (7.9, "opacity:0")]
    hand_clip = (
        f'<clipPath id="hc"><rect x="{COLS[0][0]:g}" y="{PANEL_Y0:g}" '
        f'width="{COLS[0][1] - COLS[0][0]:g}" height="{PANEL_Y1 - PANEL_Y0:g}" rx="14"/></clipPath>'
    )
    hand = hand_panel(tl, HAND, fill=c["hand_fill"], line=c["hand_line"], label=c["text"])
    content = (
        f'<g style="{tl.style(fade)}">'
        f"{_callout(tl)}"
        f'<g clip-path="url(#hc)">{hand}</g>'
        f"{_crop_panel(tl, c)}{_particles(tl)}"
        f'{encoder(tl, MATCH, faint=c["faint"], panel=c["panel"])}'
        f'<text x="{MATCH.bar_x:g}" y="{MATCH.query_y - 8:g}" font-family="{FONT_MONO}" '
        f'font-size="12" font-weight="700" fill="{c["muted"]}" letter-spacing="1">'
        f"QUERY EMBEDDING · L2-NORMALISED</text>"
        f'{query_bar(tl, MATCH, q, c["neutral"])}'
        f'<text x="{MATCH.bar_x:g}" y="{MATCH.row_y0 - 18:g}" font-family="{FONT_MONO}" '
        f'font-size="12" font-weight="700" fill="{c["muted"]}" letter-spacing="1">'
        "ENROLLED GALLERY</text></g>"
    )
    rows = gallery_rows(
        tl, MATCH, gallery, text=c["text"], muted=c["muted"], neutral=c["neutral"],
        track=c["track"], on_accent=c["on_accent"], accent=c["accent"],
    )
    defs = (
        '<linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{c["bg1"]}"/>'
        f'<stop offset="1" stop-color="{c["bg0"]}"/></linearGradient>'
        f"{gradient('g')}{gradient('hg', horizontal=True)}{gradient('cg', a=TEAL_LIGHT, b=INDIGO)}"
        '<linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{TEAL}" stop-opacity="0"/>'
        f'<stop offset="1" stop-color="{TEAL}" stop-opacity=".45"/></linearGradient>'
        '<linearGradient id="rv" x1="0" y1="0" x2="0" y2="1"><stop offset=".85" stop-color="#fff"/>'
        '<stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>'
        '<linearGradient id="rh" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" '
        'stop-opacity="0"/><stop offset=".12" stop-color="#fff"/></linearGradient>'
        '<filter id="soft" x="-10%" y="-10%" width="120%" height="120%">'
        '<feGaussianBlur stdDeviation="1.7"/></filter>'
        '<filter id="grain" x="0" y="0" width="100%" height="100%">'
        '<feTurbulence type="fractalNoise" baseFrequency=".9" numOctaves="2" seed="4"/>'
        '<feColorMatrix values="0 0 0 0 .5 0 0 0 0 .55 0 0 0 0 .6 0 0 0 .9 0"/></filter>'
        f"{hand_clip}"
    )
    rail = _rail(tl, c)
    css = (
        ".c{transform-box:fill-box;transform-origin:center}"
        ".l{transform-box:fill-box;transform-origin:left center}"
        f"{tl.css()}{REDUCED_MOTION}"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t d">'
        '<title id="t">fingerprint-from-image pipeline</title>'
        '<desc id="d">A hand photo gets 21 learned landmarks and five upright fingertip crops; '
        "the index fingertip is enhanced, encoded into a unit embedding, and matched against "
        "an enrolled gallery by cosine similarity (best match 0.94).</desc>"
        f"<defs>{defs}</defs><style>{css}</style>"
        f"{_frame(c)}{content}{rows}{rail}</svg>\n"
    )
