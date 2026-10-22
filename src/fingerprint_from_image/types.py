"""Value types shared by the learned (v3) pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FINGER_LABELS", "FingerCrop"]

FINGER_LABELS: tuple[str, ...] = tuple(
    f"{side}_{finger}"
    for side in ("left", "right")
    for finger in ("thumb", "index", "middle", "ring", "little")
) + ("unknown",)


@dataclass(frozen=True)
class FingerCrop:
    """One upright fingertip image ready for embedding.

    ``image`` is ``(H, W) uint8`` grayscale. ``polygon`` lists the crop corners
    in source-image pixel coordinates (clockwise from top-left) when the crop
    came from a larger photograph.
    """

    image: np.ndarray = field(repr=False, compare=False)
    label: str = "unknown"
    quality: float = 1.0
    polygon: tuple[tuple[float, float], ...] = ()
    source: str | None = None

    def __post_init__(self) -> None:
        if self.label not in FINGER_LABELS:
            raise ValueError(f"Unknown finger label {self.label!r}")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")
        image = np.asarray(self.image)
        if image.ndim != 2 or image.dtype != np.uint8 or min(image.shape) < 1:
            raise ValueError("FingerCrop.image must be a non-empty (H, W) uint8 array")
