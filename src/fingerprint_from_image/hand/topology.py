"""Hand landmark topology, finger names and anatomical label mapping.

MediaPipe's 21-point hand model: 0 wrist; thumb 1-4 (CMC, MCP, IP, tip);
index 5-8, middle 9-12, ring 13-16, little 17-20 (MCP, PIP, DIP, tip).

Handedness
----------
The landmark model emits one probability ``p`` that the hand is anatomically a
**right** hand, judged on the image *as given*. It is the score MediaPipe Tasks
(0.10.x) reports as ``"Right"``; on our test photographs (a right palm, a left
palm and the back of a left hand, all non-mirrored) ``p`` was 0.98, 0.12 and
0.01 respectively. A horizontally flipped image (front-camera selfie
*preview*, mirrored scan) shows the other hand's chirality, so callers declare
``mirrored=True`` for such images and the side is swapped. Whether *saved*
front-camera photos are mirrored depends on the device setting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = [
    "DIP_INDICES",
    "FINGERS",
    "HAND_CONNECTIONS",
    "MCP_INDICES",
    "PIP_INDICES",
    "TIP_INDICES",
    "Side",
    "finger_label",
    "palm_chirality",
    "palm_facing",
    "side_from_score",
]

Side = Literal["left", "right"]

FINGERS: tuple[str, ...] = ("thumb", "index", "middle", "ring", "little")
TIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)
DIP_INDICES: tuple[int, ...] = (3, 7, 11, 15, 19)  # IP joint for the thumb
PIP_INDICES: tuple[int, ...] = (2, 6, 10, 14, 18)  # MCP joint for the thumb
MCP_INDICES: tuple[int, ...] = (1, 5, 9, 13, 17)  # CMC joint for the thumb

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def side_from_score(score: float, *, mirrored: bool = False) -> tuple[Side, float]:
    """Map the landmark model's handedness probability to an anatomical side.

    Returns the side and the model's confidence in that side (``>= 0.5``).
    ``mirrored=True`` declares a horizontally flipped (selfie-preview) image.
    """

    is_right = score > 0.5
    confidence = score if is_right else 1.0 - score
    if mirrored:
        is_right = not is_right
    return ("right" if is_right else "left"), float(confidence)


def finger_label(side: Side | None, finger_index: int) -> str:
    """Return ``"<side>_<finger>"`` or ``"unknown"`` when the side is unknown."""

    if side is None:
        return "unknown"
    return f"{side}_{FINGERS[finger_index]}"


def palm_chirality(landmarks: np.ndarray) -> float:
    """Signed area of (wrist, index MCP, little MCP) in image coordinates.

    Negative when the index side is clockwise-before the little-finger side,
    i.e. the 2-D layout of a right palm (or a left hand's back) seen in a
    non-mirrored photograph. Normalised by the squared palm size.
    """

    pts = np.asarray(landmarks, dtype=np.float64)[:, :2]
    a, b = pts[5] - pts[0], pts[17] - pts[0]
    cross = a[0] * b[1] - a[1] * b[0]
    scale = max(float(np.dot(a, a) + np.dot(b, b)) / 2, 1e-9)
    return float(cross / scale)


def palm_facing(landmarks: np.ndarray, apparent_right: bool, *, min_chirality: float = 0.15
                ) -> bool | None:
    """Whether the palm (finger pads) faces the camera.

    ``apparent_right`` is the model's side for the image *as given* (before any
    mirroring correction; mirroring flips both cues, so the result does not
    depend on it). Returns ``None`` when the hand is seen edge-on and the 2-D
    layout is too degenerate to decide.
    """

    chirality = palm_chirality(landmarks)
    if abs(chirality) < min_chirality:
        return None
    return (chirality < 0) == apparent_right
