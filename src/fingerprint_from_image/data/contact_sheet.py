"""Render identity x impression contact sheets for visual label checks (Pillow only)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .quality import to_gray_uint8

__all__ = ["render_contact_sheet"]


def _thumb(path: str, cell: int) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        gray = Image.fromarray(to_gray_uint8(image))
    gray.thumbnail((cell, cell), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (cell, cell), 255)
    canvas.paste(gray, ((cell - gray.width) // 2, (cell - gray.height) // 2))
    return canvas


def render_contact_sheet(records: Sequence[Mapping[str, Any]], output: str | Path, *,
                         identities: int = 8, impressions: int = 10, cell: int = 128,
                         seed: int = 0, title: str = "",
                         label_key: str = "impression") -> Path:
    """Draw ``identities`` random identities (rows) x up to ``impressions`` images.

    Each row is labelled with the identity and split; each cell with
    ``label_key`` (and the sensor when a row mixes sensors).  Excluded rows
    are drawn with an ``X``.
    """

    by_identity: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        by_identity[row["identity"]].append(row)
    names = sorted(by_identity)
    chosen = sorted(random.Random(seed).sample(names, min(identities, len(names))))
    header, label_h, left = 24, 14, 0
    width = left + impressions * cell
    height = header + len(chosen) * (cell + 2 * label_h)
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title or str(output), fill=(0, 0, 0))
    for r, name in enumerate(chosen):
        rows = sorted(by_identity[name], key=lambda x: (str(x.get("sensor")),
                                                        x.get("impression", 0), x["path"]))
        top = header + r * (cell + 2 * label_h)
        split = rows[0].get("split", "?")
        draw.text((4, top), f"{name}  [{split}]  n={len(rows)}", fill=(0, 0, 160))
        sensors = {x.get("sensor") for x in rows}
        for c, row in enumerate(rows[:impressions]):
            x0 = left + c * cell
            try:
                sheet.paste(_thumb(row["path"], cell - 4).convert("RGB"),
                            (x0 + 2, top + label_h))
            except OSError:
                draw.rectangle((x0 + 2, top + label_h, x0 + cell - 2, top + label_h + cell),
                               outline=(255, 0, 0))
            label = str(row.get(label_key, ""))
            if len(sensors) > 1:
                label += f" {str(row.get('sensor', ''))[:14]}"
            draw.text((x0 + 3, top + label_h + cell - 4), label, fill=(200, 0, 0))
            if row.get("exclude"):
                draw.line((x0 + 2, top + label_h, x0 + cell - 2, top + label_h + cell - 4),
                          fill=(255, 0, 0), width=3)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target)
    return target
