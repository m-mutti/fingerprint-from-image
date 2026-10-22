"""Hand panel of the pipeline animation: silhouette, 21 landmarks, fingertip boxes.

Landmarks follow the MediaPipe hand topology (0 = wrist, 4/8/12/16/20 = tips)
in a local 240 x 300 frame; :class:`HandLayout` maps them into the card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from make_brand_core import (
    EASE_BACK,
    EASE_OUT,
    FONT_MONO,
    INDIGO,
    TEAL,
    TEAL_LIGHT,
    Point,
    Timeline,
    mix,
)

LANDMARKS: tuple[Point, ...] = (
    (124, 300),  # 0 wrist
    (92, 270), (66, 238), (48, 206), (34, 178),  # thumb CMC, MCP, IP, tip
    (88, 172), (79, 124), (73, 94), (69, 66),  # index MCP, PIP, DIP, tip
    (118, 164), (117, 110), (116, 76), (115, 46),  # middle
    (147, 170), (153, 120), (157, 89), (160, 62),  # ring
    (173, 186), (185, 150), (192, 127), (198, 104),  # little
)
BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)
# name, landmark chain (base -> tip), silhouette width
FINGERS: tuple[tuple[str, tuple[int, ...], float], ...] = (
    ("thumb", (1, 2, 3, 4), 30),
    ("index", (5, 6, 7, 8), 27),
    ("middle", (9, 10, 11, 12), 28),
    ("ring", (13, 14, 15, 16), 26),
    ("little", (17, 18, 19, 20), 22),
)
PALM: tuple[Point, ...] = (
    (100, 330), (94, 268), (82, 176), (118, 166), (148, 172), (174, 190), (156, 330),
)

LANDMARK_START = 0.62
LANDMARK_STEP = 0.034
BOX_START = 1.45
BOX_STEP = 0.11
SELECTED = 1  # index finger is the one zoomed into
SELECT_AT = 2.0


def finger_color(index: int) -> str:
    """Per-finger hue along the brand gradient (thumb teal -> little indigo)."""

    return mix(TEAL, INDIGO, index / 4)


def landmark_finger(index: int) -> int:
    """Finger index (0-4) owning a landmark; the wrist counts as the thumb."""

    return 0 if index == 0 else (index - 1) // 4


@dataclass(frozen=True)
class FingerBox:
    """An upright fingertip crop: centre, size, and finger direction (degrees)."""

    name: str
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # rotation that maps "up" onto the finger direction

    def corners(self) -> list[Point]:
        """Rotated corners (top-left, top-right, bottom-right, bottom-left)."""

        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        out = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            x, y = sx * self.w / 2, sy * self.h / 2
            out.append((self.cx + x * cos - y * sin, self.cy + x * sin + y * cos))
        return out


@dataclass(frozen=True)
class HandLayout:
    """Placement of the local hand frame inside the card."""

    x: float
    y: float
    scale: float

    def map(self, p: Point) -> Point:
        return (self.x + p[0] * self.scale, self.y + p[1] * self.scale)

    def boxes(self) -> list[FingerBox]:
        """Oriented fingertip boxes in card coordinates."""

        out = []
        for name, chain, width in FINGERS:
            dip, tip = self.map(LANDMARKS[chain[2]]), self.map(LANDMARKS[chain[3]])
            ux, uy = tip[0] - dip[0], tip[1] - dip[1]
            norm = math.hypot(ux, uy)
            ux, uy = ux / norm, uy / norm
            w = (width + 8) * self.scale
            h = w * 1.3
            cx, cy = tip[0] - ux * h * 0.22, tip[1] - uy * h * 0.22
            angle = math.degrees(math.atan2(uy, ux)) + 90
            out.append(FingerBox(name, cx, cy, w, h, angle))
        return out


def _silhouette(fill: str, line: str) -> str:
    """Union-of-capsules hand: thick outline strokes, then a fill pass on top."""

    def shapes(extra: float, colour: str) -> str:
        palm = " ".join(f"{x:g},{y:g}" for x, y in PALM)
        parts = [
            f'<polygon points="{palm}" fill="{colour}" stroke="{colour}" '
            f'stroke-width="{24 + extra:g}" stroke-linejoin="round"/>'
        ]
        for _, chain, width in FINGERS:
            pts = " ".join(f"{LANDMARKS[i][0]:g},{LANDMARKS[i][1]:g}" for i in chain)
            parts.append(
                f'<polyline points="{pts}" fill="none" stroke="{colour}" '
                f'stroke-width="{width + extra:g}" stroke-linecap="round" stroke-linejoin="round"/>'
            )
        return "".join(parts)

    return shapes(4.5, line) + shapes(0, fill)


def hand_panel(tl: Timeline, layout: HandLayout, *, fill: str, line: str, label: str) -> str:
    """Silhouette + landmarks + skeleton + fingertip boxes for the first panel."""

    hand_in = [
        (0.0, "opacity:0;transform:translateY(14px) scale(.96)"),
        (0.12, "opacity:0;transform:translateY(14px) scale(.96)", EASE_OUT),
        (0.75, "opacity:1;transform:translateY(0) scale(1)"),
    ]
    local = f'transform="translate({layout.x:g} {layout.y:g}) scale({layout.scale:g})"'
    parts = [
        f"<g {local}>",
        f'<g class="c" style="{tl.style(hand_in)}">{_silhouette(fill, line)}</g>',
    ]

    bones = []
    for a, b in BONES:
        appear = LANDMARK_START + LANDMARK_STEP * b
        frames = [
            (0.0, "stroke-dashoffset:100"),
            (appear - 0.05, "stroke-dashoffset:100", EASE_OUT),
            (appear + 0.25, "stroke-dashoffset:0"),
        ]
        (x1, y1), (x2, y2) = LANDMARKS[a], LANDMARKS[b]
        bones.append(
            f'<path d="M{x1:g} {y1:g}L{x2:g} {y2:g}" stroke="{finger_color(landmark_finger(b))}" '
            f'pathLength="100" stroke-dasharray="100 101" style="{tl.style(frames)}"/>'
        )
    parts.append(
        '<g fill="none" stroke-width="2.4" stroke-linecap="round" opacity=".85">'
        + "".join(bones)
        + "</g>"
    )

    dots = []
    for index, (x, y) in enumerate(LANDMARKS):
        appear = LANDMARK_START + LANDMARK_STEP * index
        frames = [
            (0.0, "opacity:0;transform:scale(0)"),
            (appear, "opacity:0;transform:scale(0)", EASE_BACK),
            (appear + 0.28, "opacity:1;transform:scale(1)"),
        ]
        tip = index in (4, 8, 12, 16, 20)
        radius = 4.6 if tip else 3.6
        dots.append(
            f'<circle class="c" cx="{x:g}" cy="{y:g}" r="{radius:g}" '
            f'fill="{finger_color(landmark_finger(index))}" style="{tl.style(frames)}"/>'
        )
    parts.append(f'<g stroke="{fill}" stroke-width="1.6">{"".join(dots)}</g></g>')
    parts.append(_boxes(tl, layout, label))
    return "".join(parts)


def _bracket_path(w: float, h: float, arm: float) -> str:
    x0, y0, x1, y1 = -w / 2, -h / 2, w / 2, h / 2
    return (
        f"M{x0:.1f} {y0 + arm:.1f}V{y0:.1f}H{x0 + arm:.1f}"
        f"M{x1 - arm:.1f} {y0:.1f}H{x1:.1f}V{y0 + arm:.1f}"
        f"M{x1:.1f} {y1 - arm:.1f}V{y1:.1f}H{x1 - arm:.1f}"
        f"M{x0 + arm:.1f} {y1:.1f}H{x0:.1f}V{y1 - arm:.1f}"
    )


def _selection(tl: Timeline, box: FingerBox) -> str:
    """Ring that flashes on the fingertip chosen for the zoom, then stays."""

    flash = [
        (0.0, "opacity:0;transform:scale(1.5)"),
        (SELECT_AT, "opacity:0;transform:scale(1.5)", EASE_OUT),
        (SELECT_AT + 0.25, "opacity:1;transform:scale(1)"),
    ]
    return (
        f'<rect x="{-box.w / 2 - 3:.1f}" y="{-box.h / 2 - 3:.1f}" width="{box.w + 6:.1f}" '
        f'height="{box.h + 6:.1f}" rx="5" fill="none" stroke="{TEAL_LIGHT}" stroke-width="1.6" '
        f'style="{tl.style(flash)}"/>'
    )


def _boxes(tl: Timeline, layout: HandLayout, label: str) -> str:
    out = []
    for k, box in enumerate(layout.boxes()):
        start = BOX_START + BOX_STEP * k
        lock = [
            (0.0, "opacity:0;transform:scale(1.9) rotate(-12deg)"),
            (start, "opacity:0;transform:scale(1.9) rotate(-12deg)", EASE_BACK),
            (start + 0.38, "opacity:1;transform:scale(1) rotate(0deg)"),
        ]
        colour = finger_color(k)
        out.append(
            f'<g transform="translate({box.cx:.1f} {box.cy:.1f}) rotate({box.angle:.1f})">'
            f'<g style="{tl.style(lock)}">'
            f'<rect x="{-box.w / 2:.1f}" y="{-box.h / 2:.1f}" width="{box.w:.1f}" '
            f'height="{box.h:.1f}" rx="3" fill="{colour}" fill-opacity=".1"/>'
            f'<path d="{_bracket_path(box.w, box.h, box.w * 0.3)}" fill="none" '
            f'stroke="{colour}" stroke-width="2.2" stroke-linecap="round"/></g>'
            + (_selection(tl, box) if k == SELECTED else "")
            + "</g>"
        )
        # Label sits beyond the fingertip, along the finger direction.
        rad = math.radians(box.angle - 90)
        reach = box.h / 2 + 11
        lx, ly = box.cx + math.cos(rad) * reach, box.cy + math.sin(rad) * reach + 4
        text = [
            (0.0, "opacity:0;transform:translateY(4px)"),
            (start + 0.15, "opacity:0;transform:translateY(4px)", EASE_OUT),
            (start + 0.45, "opacity:1;transform:translateY(0)"),
        ]
        out.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{label}" style="{tl.style(text)}">'
            f"{box.name}</text>"
        )
    return (
        f'<g font-family="{FONT_MONO}" font-size="12.5" font-weight="600" '
        f'text-anchor="middle">{"".join(out)}</g>'
    )
