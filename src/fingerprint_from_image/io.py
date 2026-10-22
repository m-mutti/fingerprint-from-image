"""OpenCV-free image loading for the learned (v3) pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

__all__ = ["ImageLoadError", "load_image", "to_gray", "to_rgb"]


class ImageLoadError(ValueError):
    """Raised when an image cannot be decoded into a supported array."""


def load_image(path: str | Path, *, mode: Literal["L", "RGB"] = "L") -> np.ndarray:
    """Decode an image file into ``(H, W)`` grayscale or ``(H, W, 3)`` RGB uint8.

    EXIF orientation is applied so phone photographs come out upright.
    """

    source = Path(path)
    try:
        with Image.open(source) as handle:
            image = ImageOps.exif_transpose(handle)
            if image.mode in {"I;16", "I;16B", "I;16L", "I", "F"}:
                gray = _normalize_to_uint8(np.asarray(image, dtype=np.float32))
                return gray if mode == "L" else to_rgb(gray)
            return np.asarray(image.convert(mode), dtype=np.uint8).copy()
    except Image.DecompressionBombError as exc:
        raise ImageLoadError(f"Image is too large to decode safely: {source}") from exc
    except (OSError, ValueError) as exc:
        raise ImageLoadError(f"Could not read image: {source}") from exc


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(finite.shape, dtype=np.uint8)
    return ((finite - low) * (255.0 / (high - low))).round().astype(np.uint8)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA arrays (any numeric dtype) to ``(H, W) uint8``."""

    array = np.asarray(image)
    if array.size == 0:
        raise ImageLoadError("Image is empty")
    if array.dtype != np.uint8:
        array = _normalize_to_uint8(array)
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        rgb = array[..., :3].astype(np.float32)
        gray = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        return np.clip(gray.round(), 0, 255).astype(np.uint8)
    raise ImageLoadError(f"Expected a grayscale or RGB image, got shape {array.shape}")


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGB/RGBA arrays to ``(H, W, 3) uint8`` RGB."""

    array = np.asarray(image)
    if array.size == 0:
        raise ImageLoadError("Image is empty")
    if array.dtype != np.uint8:
        array = _normalize_to_uint8(array)
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[2] == 3:
        return np.ascontiguousarray(array)
    if array.ndim == 3 and array.shape[2] == 4:
        return np.ascontiguousarray(array[..., :3])
    raise ImageLoadError(f"Expected a grayscale or RGB image, got shape {array.shape}")
