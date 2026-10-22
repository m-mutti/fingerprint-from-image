"""Encoder, embedding bar, and gallery-matching panels of the pipeline animation.

The gallery scores shown on screen are the *actual* cosine similarities of the
drawn bars: each enrolled vector is built as ``a * q + sqrt(1 - a^2) * r`` with
``r`` orthogonal to the query ``q``, so the numbers and colours agree.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from make_brand_core import (
    EASE_BACK,
    EASE_IN_OUT,
    EASE_OUT,
    FONT_MONO,
    INDIGO,
    INDIGO_LIGHT,
    TEAL,
    TEAL_LIGHT,
    Timeline,
    lcg,
    mix,
)

DIM = 32
CELL = 7.6
NETWORK_LAYERS = (6, 5, 4)
ENCODE_START = 3.25
EMBED_START = 3.95
MATCH_START = 4.45
MATCH_STEP = 0.22
VERDICT = 5.75
GALLERY: tuple[tuple[str, float], ...] = (
    ("bob · left_thumb", 0.41),
    ("alice · right_index", 0.94),
    ("chen · right_ring", 0.12),
    ("dana · left_middle", 0.27),
)


def _unit(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def _centered(seed: int) -> list[float]:
    stream = lcg(seed)
    return [2 * stream[i] - 1 for i in range(DIM)]


def embeddings() -> tuple[list[float], list[list[float]]]:
    """Unit query vector and gallery vectors with the exact GALLERY cosines."""

    q = _unit(_centered(7))
    gallery = []
    for k, (_, cosine) in enumerate(GALLERY):
        r = _centered(101 + 13 * k)
        dot = sum(a * b for a, b in zip(r, q, strict=True))
        r = _unit([a - dot * b for a, b in zip(r, q, strict=True)])
        s = math.sqrt(1 - cosine**2)
        gallery.append([cosine * a + s * b for a, b in zip(q, r, strict=True)])
    return q, gallery


def cell_colour(value: float, scale: float, neutral: str) -> str:
    """Diverging map: negative -> indigo, zero -> neutral, positive -> teal."""

    t = min(1.0, abs(value) / scale) ** 0.75
    return mix(neutral, TEAL_LIGHT if value >= 0 else INDIGO_LIGHT, t)


def bar(values: Sequence[float], x: float, y: float, h: float, neutral: str) -> list[str]:
    """One ``<rect>`` per embedding dimension."""

    scale = max(abs(v) for v in values)
    return [
        f'<rect x="{x + i * CELL + 0.5:g}" y="{y:g}" width="{CELL - 1:g}" height="{h:g}" '
        f'rx="1.2" fill="{cell_colour(v, scale, neutral)}"/>'
        for i, v in enumerate(values)
    ]


@dataclass(frozen=True)
class MatchLayout:
    """Geometry of the encoder column and the embedding/gallery column."""

    net_x: tuple[float, float, float]
    net_cy: float
    bar_x: float
    query_y: float
    row_y0: float
    row_step: float
    score_x: float


def network_nodes(layout: MatchLayout) -> list[list[tuple[float, float]]]:
    """Node centres per layer."""

    layers = []
    for x, count in zip(layout.net_x, NETWORK_LAYERS, strict=True):
        span = (count - 1) * 22
        layers.append([(x, layout.net_cy - span / 2 + 22 * i) for i in range(count)])
    return layers


def encoder(tl: Timeline, layout: MatchLayout, *, faint: str, panel: str) -> str:
    """Faint static network plus travelling activations and lighting nodes."""

    layers = network_nodes(layout)
    lines, comets, nodes = [], [], []
    for li in range(len(layers) - 1):
        t0 = ENCODE_START + 0.22 * li
        for ai, (x1, y1) in enumerate(layers[li]):
            for bi, (x2, y2) in enumerate(layers[li + 1]):
                d = f"M{x1:g} {y1:g}L{x2:g} {y2:g}"
                lines.append(f'<path d="{d}"/>')
                if (ai + bi) % 2:
                    continue  # animate every other edge: busy enough, still legible
                start = t0 + 0.03 * ((ai + 2 * bi) % 4)
                frames = [
                    (0.0, "stroke-dashoffset:24;opacity:0"),
                    (start, "stroke-dashoffset:24;opacity:1", "linear"),
                    (start + 0.32, "stroke-dashoffset:-100;opacity:1"),
                    (start + 0.33, "stroke-dashoffset:-100;opacity:0"),
                ]
                colour = mix(TEAL_LIGHT, INDIGO_LIGHT, (li + 0.5) / 2)
                comets.append(
                    f'<path d="{d}" stroke="{colour}" pathLength="100" stroke-dasharray="24 200" '
                    f'opacity="0" style="{tl.style(frames)}"/>'
                )
    # Output nodes all feed the left end of the query bar.
    target = (layout.bar_x - 4, layout.query_y + 11)
    for x1, y1 in layers[-1]:
        tx, ty = target
        d = f"M{x1:g} {y1:g}C{x1 + 18:g} {y1:g} {tx - 18:g} {ty:g} {tx:g} {ty:g}"
        lines.append(f'<path d="{d}"/>')
        start = ENCODE_START + 0.5
        frames = [
            (0.0, "stroke-dashoffset:24;opacity:0"),
            (start, "stroke-dashoffset:24;opacity:1", "linear"),
            (start + 0.3, "stroke-dashoffset:-100;opacity:1"),
            (start + 0.31, "stroke-dashoffset:-100;opacity:0"),
        ]
        comets.append(
            f'<path d="{d}" stroke="{INDIGO_LIGHT}" pathLength="100" stroke-dasharray="24 200" '
            f'opacity="0" style="{tl.style(frames)}"/>'
        )
    for li, layer in enumerate(layers):
        colour = mix(TEAL, INDIGO, li / 2)
        lit = ENCODE_START + 0.22 * li + 0.12
        for x, y in layer:
            nodes.append(f'<circle cx="{x:g}" cy="{y:g}" r="5.5" fill="{panel}" stroke="{faint}"/>')
            frames = [
                (0.0, "opacity:0;transform:scale(.4)"),
                (lit, "opacity:0;transform:scale(.4)", EASE_BACK),
                (lit + 0.25, "opacity:1;transform:scale(1)"),
            ]
            nodes.append(
                f'<circle class="c" cx="{x:g}" cy="{y:g}" r="5.5" fill="{colour}" '
                f'style="{tl.style(frames)}"/>'
            )
    return (
        f'<g fill="none" stroke="{faint}" stroke-width="1">{"".join(lines)}</g>'
        f'<g fill="none" stroke-width="1.8" stroke-linecap="round">{"".join(comets)}</g>'
        f'<g stroke-width="1.5">{"".join(nodes)}</g>'
    )


def query_bar(tl: Timeline, layout: MatchLayout, q: Sequence[float], neutral: str) -> str:
    """The query embedding condensing out of the encoder, cell by cell."""

    cells = bar(q, layout.bar_x, layout.query_y, 22, neutral)
    out = []
    for i, cell in enumerate(cells):
        t = EMBED_START + 0.018 * i
        frames = [
            (0.0, "opacity:0;transform:scaleY(.1)"),
            (t, "opacity:0;transform:scaleY(.1)", EASE_BACK),
            (t + 0.3, "opacity:1;transform:scaleY(1)"),
        ]
        out.append(cell.replace("<rect ", f'<rect class="c" style="{tl.style(frames)}" ', 1))
    return "".join(out)


def _count_texts(tl: Timeline, value: float, start: float, x: float, y: float) -> str:
    """Stacked numbers shown one after another: a CSS-only count-up."""

    steps = 9
    out = []
    for s in range(1, steps + 1):
        shown = value * (s / steps) ** 0.6
        t_on = start + 0.07 * (s - 1)
        final = s == steps
        frames = [(0.0, "opacity:0"), (t_on, "opacity:0"), (t_on + 0.01, "opacity:1")]
        if not final:
            frames += [(t_on + 0.07, "opacity:1"), (t_on + 0.08, "opacity:0")]
        base = "" if final else ' opacity="0"'
        out.append(
            f'<text x="{x:g}" y="{y:g}"{base} style="{tl.style(frames)}">{shown:.2f}</text>'
        )
    return "".join(out)


def gallery_rows(
    tl: Timeline,
    layout: MatchLayout,
    gallery: Sequence[Sequence[float]],
    *,
    text: str,
    muted: str,
    neutral: str,
    track: str,
    on_accent: str,
    accent: str,
) -> str:
    """Enrolled rows: names, bars, similarity tracks, counted scores, verdict."""

    best = max(range(len(GALLERY)), key=lambda k: GALLERY[k][1])
    width = DIM * CELL
    out = []
    for k, ((name, cosine), vec) in enumerate(zip(GALLERY, gallery, strict=True)):
        y = layout.row_y0 + layout.row_step * k
        start = MATCH_START + MATCH_STEP * k
        is_best = k == best
        rest = 1 if is_best else 0.42
        dim = [
            (0.0, "opacity:.6"),
            (start - 0.1, "opacity:.6", EASE_OUT),
            (start + 0.1, "opacity:1"),
            (VERDICT, "opacity:1", EASE_OUT),
            (VERDICT + 0.35, f"opacity:{rest:g}"),
            (7.5, f"opacity:{rest:g}", EASE_IN_OUT),
            (7.9, "opacity:.6"),
        ]
        base_opacity = "" if is_best else ' opacity=".42"'
        fill = [
            (0.0, "transform:scaleX(0)"),
            (start, "transform:scaleX(0)", EASE_OUT),
            (start + 0.65, "transform:scaleX(1)"),
            (7.5, "transform:scaleX(1)", EASE_IN_OUT),
            (7.9, "transform:scaleX(0)"),
        ]
        score_colour = accent if is_best else text
        score_in = [(0.0, "opacity:0"), (start, "opacity:0"), (start + 0.01, "opacity:1"),
                    (7.5, "opacity:1", EASE_IN_OUT), (7.9, "opacity:0")]
        out.append(
            f'<g{base_opacity} style="{tl.style(dim)}">'
            f'<text x="{layout.bar_x:g}" y="{y + 10:g}" fill="{muted}" '
            f'font-family="{FONT_MONO}" font-size="13">{name}</text>'
            + "".join(bar(vec, layout.bar_x, y + 17, 12, neutral))
            + f'<rect x="{layout.bar_x + 0.5:g}" y="{y + 34:g}" width="{width - 1:g}" height="3" '
            f'rx="1.5" fill="{track}"/>'
            f'<rect class="l" x="{layout.bar_x + 0.5:g}" y="{y + 34:g}" '
            f'width="{(width - 1) * cosine:.1f}" height="3" rx="1.5" fill="url(#hg)" '
            f'style="{tl.style(fill)}"/>'
            f'<g font-family="{FONT_MONO}" font-size="18" font-weight="700" text-anchor="end" '
            f'fill="{score_colour}" style="{tl.style(score_in)}">'
            f"{_count_texts(tl, cosine, start, layout.score_x, y + 30)}</g></g>"
        )
        if is_best:
            out.append(_verdict(tl, layout, y, on_accent))
    return "".join(out)


def _verdict(tl: Timeline, layout: MatchLayout, y: float, on_accent: str) -> str:
    """Highlight frame + "MATCH" pill with a tick for the best gallery row."""

    width = DIM * CELL
    frame = [
        (0.0, "opacity:0"),
        (VERDICT, "opacity:0", EASE_OUT),
        (VERDICT + 0.35, "opacity:1"),
        (7.5, "opacity:1", EASE_IN_OUT),
        (7.9, "opacity:0"),
    ]
    pill = [
        (0.0, "opacity:0;transform:scale(.5)"),
        (VERDICT + 0.1, "opacity:0;transform:scale(.5)", EASE_BACK),
        (VERDICT + 0.5, "opacity:1;transform:scale(1)"),
        (7.5, "opacity:1;transform:scale(1)", EASE_IN_OUT),
        (7.9, "opacity:0;transform:scale(.8)"),
    ]
    x0 = layout.bar_x - 10
    px = layout.bar_x + width - 74
    return (
        f'<rect x="{x0:g}" y="{y - 6:g}" width="{layout.score_x - x0 + 10:g}" height="50" rx="10" '
        f'fill="{TEAL}" fill-opacity=".1" stroke="{TEAL_LIGHT}" stroke-width="1.5" '
        f'style="{tl.style(frame)}"/>'
        f'<g class="c" style="{tl.style(pill)}">'
        f'<rect x="{px:g}" y="{y - 2:g}" width="74" height="16" rx="8" fill="{TEAL_LIGHT}"/>'
        f'<path d="M{px + 9:g} {y + 6:g}l3 3 5.5-6" fill="none" stroke="{on_accent}" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<text x="{px + 22:g}" y="{y + 10:g}" font-family="{FONT_MONO}" font-size="12" '
        f'font-weight="800" letter-spacing=".8" fill="{on_accent}">MATCH</text></g>'
    )
