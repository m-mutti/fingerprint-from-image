"""Pillow drawing helpers for hand landmarks and fingertip crops (docs / demos)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..io import to_rgb
from ..types import FingerCrop
from .detector import HandResult
from .topology import HAND_CONNECTIONS, TIP_INDICES

__all__ = ["crop_mosaic", "draw_hands"]

_FINGER_COLOURS = ((230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (204, 121, 167))


def draw_hands(rgb: np.ndarray, hands: Sequence[HandResult],
               crops: Sequence[FingerCrop] = (), *, line_width: int | None = None) -> Image.Image:
    """Return an RGB image with skeletons, landmarks, crop polygons and labels."""

    image = Image.fromarray(to_rgb(rgb)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width = line_width or max(2, round(max(image.size) / 400))
    radius = 1.5 * width
    for hand in hands:
        pts = hand.landmarks[:, :2]
        for a, b in HAND_CONNECTIONS:
            draw.line([tuple(pts[a]), tuple(pts[b])], fill=(255, 255, 255), width=width)
        for index, (x, y) in enumerate(pts):
            colour = (255, 64, 64) if index in TIP_INDICES else (40, 40, 255)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=colour)
        x0, y0 = pts.min(axis=0)
        facing = {True: "palm", False: "back", None: "?"}[hand.palm_facing]
        draw.text((x0, max(0.0, y0 - 14 * width)),
                  f"{hand.side} ({hand.side_confidence:.2f}) {facing}", fill=(255, 255, 0))
    for index, crop in enumerate(crops):
        if not crop.polygon:
            continue
        colour = _FINGER_COLOURS[index % len(_FINGER_COLOURS)]
        polygon = [(float(x), float(y)) for x, y in crop.polygon]
        draw.line(polygon + [polygon[0]], fill=colour, width=width)
        draw.line([polygon[0], polygon[1]], fill=(255, 0, 0), width=width)  # tip edge
        draw.text(polygon[0], f"{crop.label} q={crop.quality:.2f}", fill=colour)
    return image


def crop_mosaic(crops: Sequence[FingerCrop], *, height: int = 256, pad: int = 6) -> Image.Image:
    """Tile crops side by side (each scaled to ``height``) with their labels."""

    if not crops:
        raise ValueError("no crops to tile")
    tiles = []
    for crop in crops:
        tile = Image.fromarray(crop.image)
        scale = height / tile.height
        tiles.append(tile.resize((max(1, round(tile.width * scale)), height),
                                 Image.Resampling.BICUBIC))
    caption = 14
    canvas = Image.new("L", (sum(t.width for t in tiles) + pad * (len(tiles) + 1),
                             height + caption + 2 * pad), 255)
    draw = ImageDraw.Draw(canvas)
    x = pad
    for crop, tile in zip(crops, tiles, strict=True):
        canvas.paste(tile, (x, pad + caption))
        draw.text((x, pad // 2), f"{crop.label} {crop.quality:.2f}", fill=0)
        x += tile.width + pad
    return canvas
