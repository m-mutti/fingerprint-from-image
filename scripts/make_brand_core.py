"""Shared building blocks for the brand generators: palette, ridges, keyframes.

Everything here is pure Python (standard library only) and deterministic, so
the generated SVG files are byte-stable across machines and Python versions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

TEAL = "#14B8A6"
TEAL_LIGHT = "#2DD4BF"
TEAL_GLOW = "#5EEAD4"
INDIGO = "#6366F1"
INDIGO_LIGHT = "#818CF8"
INK = "#0F172A"
PAPER = "#F8FAFC"
AMBER = "#F59E0B"  # detection accent (minutiae, match tick)
SLATE_400 = "#94A3B8"
SLATE_600 = "#475569"

FONT_SANS = (
    "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)
FONT_MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# (radius, [(start_deg, end_deg), ...]) — angles measured clockwise from +x in
# SVG coordinates. Gaps between segments mimic ridge endings.
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
# A bolder four-ridge variant that survives favicon sizes (16–32 px).
SMALL_RIDGES: tuple[tuple[float, tuple[tuple[float, float], ...]], ...] = (
    (9.0, ((200, 500),)),
    (24.5, ((168, 318), (340, 450))),
    (40.0, ((160, 412),)),
)
CENTER = (64.0, 64.0)
ROTATION = -25.0  # turns the ridge-ending gap toward the bottom, like a fingertip
ASPECT = 1.18  # fingertips are taller than they are wide

Point = tuple[float, float]


@dataclass(frozen=True)
class Ridge:
    """One ridge segment of the mark, in the 128 x 128 logo coordinate frame."""

    radius: float
    start: float
    end: float
    points: tuple[Point, ...]

    @property
    def length(self) -> float:
        """Approximate arc length, used for stroke-dash draw-on animations."""

        return math.radians(self.end - self.start) * self.radius * (1 + ASPECT) / 2

    def path(self, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> str:
        """SVG path data, optionally scaled and translated."""

        pts = [(x * scale + dx, y * scale + dy) for x, y in self.points]
        head = f"M{pts[0][0]:.1f} {pts[0][1]:.1f}"
        return head + "".join(f"L{x:.1f} {y:.1f}" for x, y in pts[1:])


def ellipse_point(radius: float, angle_deg: float) -> Point:
    """Point on the rotated, vertically stretched ridge ellipse."""

    cx, cy = CENTER
    angle = math.radians(ROTATION + angle_deg)
    return (cx + radius * math.cos(angle), cy + ASPECT * radius * math.sin(angle))


def ridges(
    table: Sequence[tuple[float, Sequence[tuple[float, float]]]] = RIDGES,
    step_deg: float = 6.0,
) -> list[Ridge]:
    """Expand a ridge table into polyline segments (inner ridges first)."""

    out = []
    for radius, segments in table:
        for start, end in segments:
            steps = max(6, int(abs(end - start) / step_deg))
            pts = tuple(
                ellipse_point(radius, start + (end - start) * i / steps) for i in range(steps + 1)
            )
            out.append(Ridge(radius, start, end, pts))
    return out


@dataclass(frozen=True)
class Minutia:
    """A ridge ending: position plus the direction the ridge runs out of."""

    x: float
    y: float
    angle: float  # degrees, SVG convention (clockwise from +x)


def minutiae(count: int = 10, *, min_radius: float = 14.0) -> list[Minutia]:
    """Pick ``count`` ridge endings spread evenly around the core.

    Endings are segment end points; the tangent points away from the ridge so
    a short tick drawn along ``angle`` reproduces a forensic minutiae plot.
    """

    candidates: list[tuple[float, Minutia]] = []
    for ridge in ridges():
        if ridge.radius < min_radius:
            continue
        for tip, prev in ((ridge.points[0], ridge.points[1]), (ridge.points[-1], ridge.points[-2])):
            angle = math.degrees(math.atan2(tip[1] - prev[1], tip[0] - prev[0]))
            polar = math.degrees(math.atan2(tip[1] - CENTER[1], tip[0] - CENTER[0])) % 360
            candidates.append((polar, Minutia(tip[0], tip[1], angle)))
    candidates.sort(key=lambda item: item[0])
    picked: list[Minutia] = []
    stride = len(candidates) / count
    for index in range(count):
        picked.append(candidates[int(index * stride)][1])
    return picked


def viewfinder(stroke: str, width: float = 5.0, *, lo: float = 8, hi: float = 120) -> str:
    """Camera viewfinder corners: the "from image" half of the mark."""

    return "".join(viewfinder_corners(stroke, width, lo=lo, hi=hi, wrap=True))


def viewfinder_corners(
    stroke: str,
    width: float = 5.0,
    *,
    lo: float = 8,
    hi: float = 120,
    corner: float = 22,
    radius: float = 6,
    wrap: bool = False,
) -> list[str]:
    """Return the four corner paths separately (top-left, clockwise)."""

    c, r = corner, radius
    paths = [
        f'<path d="M{lo:g} {lo + c:g}V{lo + r:g}Q{lo:g} {lo:g} {lo + r:g} {lo:g}H{lo + c:g}"/>',
        f'<path d="M{hi - c:g} {lo:g}H{hi - r:g}Q{hi:g} {lo:g} {hi:g} {lo + r:g}V{lo + c:g}"/>',
        f'<path d="M{hi:g} {hi - c:g}V{hi - r:g}Q{hi:g} {hi:g} {hi - r:g} {hi:g}H{hi - c:g}"/>',
        f'<path d="M{lo + c:g} {hi:g}H{lo + r:g}Q{lo:g} {hi:g} {lo:g} {hi - r:g}V{hi - c:g}"/>',
    ]
    if wrap:
        head = (
            f'<g fill="none" stroke="{stroke}" stroke-width="{width:g}" '
            'stroke-linecap="round" stroke-linejoin="round">'
        )
        return [head, *paths, "</g>"]
    return paths


def gradient(identifier: str, *, horizontal: bool = False, a: str = TEAL, b: str = INDIGO) -> str:
    """Brand gradient (teal -> indigo), diagonal by default."""

    end = 'x2="1" y2="0"' if horizontal else 'x2="1" y2="1"'
    return (
        f'<linearGradient id="{identifier}" x1="0" y1="0" {end}>'
        f'<stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/>'
        "</linearGradient>"
    )


def mix(a: str, b: str, t: float) -> str:
    """Linear interpolation between two ``#RRGGBB`` colours."""

    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb, strict=True))


def lcg(seed: int) -> list[float]:
    """Deterministic pseudo-random stream in [0, 1) (no global RNG state)."""

    state = seed & 0xFFFFFFFF
    out = []
    for _ in range(256):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        out.append(state / 2**32)
    return out


Frame = tuple[float, str] | tuple[float, str, str]


class Timeline:
    """Collects ``@keyframes`` for one looping master clock.

    Every animation shares the same duration and runs ``infinite``, so all
    elements stay phase-locked and the loop is seamless. Keyframe times are
    given in seconds and converted to percentages; identical keyframe bodies
    are emitted once and shared.
    """

    def __init__(self, duration: float, prefix: str = "k") -> None:
        self.duration = duration
        self.prefix = prefix
        self._names: dict[str, str] = {}

    def _pct(self, t: float) -> str:
        value = max(0.0, min(100.0, 100.0 * t / self.duration))
        return f"{value:.2f}".rstrip("0").rstrip(".") + "%"

    def keyframes(self, frames: Sequence[Frame]) -> str:
        """Register keyframes and return the generated animation name."""

        ordered = sorted(frames, key=lambda f: f[0])
        if ordered[0][0] > 0:
            ordered.insert(0, (0.0, ordered[0][1]))
        if ordered[-1][0] < self.duration:
            ordered.append((self.duration, ordered[-1][1]))
        parts = []
        for frame in ordered:
            props = frame[1]
            if len(frame) == 3:
                props += f";animation-timing-function:{frame[2]}"
            parts.append(f"{self._pct(frame[0])}{{{props}}}")
        body = "".join(parts)
        if body not in self._names:
            self._names[body] = f"{self.prefix}{len(self._names)}"
        return self._names[body]

    def style(self, frames: Sequence[Frame], easing: str = "linear") -> str:
        """Inline ``style`` value that runs ``frames`` on the master clock."""

        name = self.keyframes(frames)
        return f"animation:{name} {self.duration:g}s {easing} infinite both"

    def css(self) -> str:
        """All registered ``@keyframes`` rules."""

        return "".join(f"@keyframes {name}{{{body}}}" for body, name in self._names.items())


EASE_OUT = "cubic-bezier(.22,1,.36,1)"
EASE_IN_OUT = "cubic-bezier(.65,0,.35,1)"
EASE_BACK = "cubic-bezier(.34,1.56,.64,1)"

REDUCED_MOTION = (
    "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
)


def fade_window(t_in: float, t_out: float, fade: float = 0.3, *, peak: float = 1.0) -> list[Frame]:
    """Opacity keyframes: hidden, fade in at ``t_in``, fade out at ``t_out``."""

    return [
        (0.0, "opacity:0"),
        (t_in, "opacity:0", EASE_OUT),
        (t_in + fade, f"opacity:{peak:g}"),
        (t_out, f"opacity:{peak:g}"),
        (t_out + fade, "opacity:0"),
    ]
