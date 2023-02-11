"""Extract individual fingerprint crops from a photograph of a hand.

The original project implemented this pipeline directly at module import time.
This module keeps the same OpenCV approach, but exposes it as a reusable API so
that the matching framework can consume the crops without opening a GUI.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2 as cv
import numpy as np

EXTRACTION_VERSION = "hsv-contours-v2"


class FingerprintExtractionError(ValueError):
    """Raised when an input image cannot be used for fingerprint extraction."""


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for the hand segmentation pipeline."""

    skin_hsv_lower: tuple[int, int, int] = (0, 20, 70)
    skin_hsv_upper: tuple[int, int, int] = (20, 255, 255)
    max_fingers: int = 4
    min_contour_area_ratio: float = 0.005
    morphology_kernel_size: int = 3
    adaptive_block_size: int = 11
    adaptive_c: float = 2.0

    def __post_init__(self) -> None:
        if self.max_fingers < 1:
            raise ValueError("max_fingers must be at least 1")
        if not 0 <= self.min_contour_area_ratio < 1:
            raise ValueError("min_contour_area_ratio must be in [0, 1)")
        if self.morphology_kernel_size < 1:
            raise ValueError("morphology_kernel_size must be positive")
        if self.adaptive_block_size < 3 or self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be an odd integer >= 3")


@dataclass(frozen=True)
class FingerprintCrop:
    """One extracted fingerprint image and its location in the source image."""

    image: np.ndarray = field(repr=False, compare=False)
    bbox: tuple[int, int, int, int]
    contour_area: float
    quality: float


def _read_color_image(image: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(image, (str, Path)):
        path = Path(image)
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
        except OSError as exc:
            raise FingerprintExtractionError(f"Could not read image: {path}") from exc
        data = cv.imdecode(encoded, cv.IMREAD_COLOR) if encoded.size else None
        if data is None:
            raise FingerprintExtractionError(f"Could not read image: {path}")
        return data

    data = np.asarray(image)
    if data.size == 0:
        raise FingerprintExtractionError("Input image is empty")
    if data.dtype != np.uint8:
        data = cv.normalize(data, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    if data.ndim == 2:
        return cv.cvtColor(data, cv.COLOR_GRAY2BGR)
    if data.ndim != 3:
        raise FingerprintExtractionError(
            f"Expected a grayscale or color image, got shape {data.shape}"
        )
    if data.shape[2] == 4:
        return cv.cvtColor(data, cv.COLOR_BGRA2BGR)
    if data.shape[2] != 3:
        raise FingerprintExtractionError(
            f"Expected 3 or 4 color channels, got {data.shape[2]}"
        )
    return np.ascontiguousarray(data)


def extract_fingerprints(
    image: str | Path | np.ndarray,
    config: ExtractionConfig | None = None,
) -> list[FingerprintCrop]:
    """Extract fingerprint crops, ordered from left to right in the source.

    An empty list is returned when no contour passes the configured area filter.
    The output images are 8-bit, single-channel, adaptively thresholded crops.
    """

    config = config or ExtractionConfig()
    source = _read_color_image(image)

    hsv = cv.cvtColor(source, cv.COLOR_BGR2HSV)
    lower = np.asarray(config.skin_hsv_lower, dtype=np.uint8)
    upper = np.asarray(config.skin_hsv_upper, dtype=np.uint8)
    skin_mask = cv.inRange(hsv, lower, upper)

    kernel_size = config.morphology_kernel_size
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    skin_mask = cv.morphologyEx(skin_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(
        skin_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    minimum_area = source.shape[0] * source.shape[1] * config.min_contour_area_ratio
    contours = [contour for contour in contours if cv.contourArea(contour) >= minimum_area]
    contours = sorted(contours, key=cv.contourArea, reverse=True)[: config.max_fingers]
    # Area rank is useful for choosing the four relevant contours, but is not a
    # stable label. Spatial order makes repeated captures deterministic.
    contours.sort(key=lambda contour: cv.boundingRect(contour)[0])

    if not contours:
        return []

    selected_mask = np.zeros(source.shape[:2], dtype=np.uint8)
    cv.drawContours(selected_mask, contours, -1, 255, cv.FILLED)
    selected_mask = cv.GaussianBlur(selected_mask, (5, 5), 0)

    hand = cv.bitwise_and(source, source, mask=selected_mask)
    gray = cv.cvtColor(hand, cv.COLOR_BGR2GRAY)
    thresholded = cv.adaptiveThreshold(
        gray,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        config.adaptive_block_size,
        config.adaptive_c,
    )

    crops: list[FingerprintCrop] = []
    for contour in contours:
        x, y, width, height = cv.boundingRect(contour)
        crop = thresholded[y : y + height, x : x + width].copy()

        # Keep the exterior of the contour uniformly white. This prevents the
        # bounding rectangle from generating artificial feature points later.
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
        cv.drawContours(contour_mask, [shifted], -1, 255, cv.FILLED)
        crop[contour_mask == 0] = 255

        inside = crop[contour_mask != 0]
        contrast = float(np.std(inside) / 127.5) if inside.size else 0.0
        coverage = float(np.count_nonzero(contour_mask) / contour_mask.size)
        quality = float(np.clip(0.75 * contrast + 0.25 * coverage, 0.0, 1.0))
        crops.append(
            FingerprintCrop(
                image=crop,
                bbox=(x, y, width, height),
                contour_area=float(cv.contourArea(contour)),
                quality=quality,
            )
        )

    return crops


def extract_fingerprint_images(
    image: str | Path | np.ndarray,
    config: ExtractionConfig | None = None,
) -> list[np.ndarray]:
    """Convenience wrapper returning only the crop arrays."""

    return [crop.image for crop in extract_fingerprints(image, config)]


def save_fingerprints(crops: Sequence[FingerprintCrop], output_dir: str | Path) -> None:
    """Write extracted crops as lossless PNG files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, crop in enumerate(crops, start=1):
        output = destination / f"finger_{index}.png"
        encoded_ok, encoded = cv.imencode(".png", crop.image)
        try:
            if encoded_ok:
                encoded.tofile(output)
        except OSError as exc:
            raise FingerprintExtractionError(f"Could not write image: {output}") from exc
        if not encoded_ok or not output.is_file():
            raise FingerprintExtractionError(f"Could not write image: {output}")


def plot_fingerprints(crops: Sequence[FingerprintCrop]) -> None:
    """Display the extracted crops. Matplotlib is imported only when requested."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional GUI package
        raise FingerprintExtractionError(
            "Displaying crops requires matplotlib; install it or use --no-display"
        ) from exc

    columns = 2
    rows = max(1, (len(crops) + columns - 1) // columns)
    figure = plt.figure(figsize=(10, 5 * rows))
    for index, crop in enumerate(crops, start=1):
        axis = figure.add_subplot(rows, columns, index)
        axis.imshow(crop.image, cmap="gray")
        axis.set_title(f"Finger {index} (quality {crop.quality:.2f})")
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract individual fingerprints from a hand photograph"
    )
    parser.add_argument("image", help="path to the hand image")
    parser.add_argument(
        "--output-dir", help="directory in which to save finger_1.png, ..."
    )
    parser.add_argument(
        "--no-display", action="store_true", help="do not open a preview window"
    )
    parser.add_argument(
        "--max-fingers", type=int, default=4, help="maximum number of crops (default: 4)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        crops = extract_fingerprints(
            args.image, ExtractionConfig(max_fingers=args.max_fingers)
        )
        if not crops:
            raise FingerprintExtractionError("No fingerprints were found in the image")
        if args.output_dir:
            save_fingerprints(crops, args.output_dir)
        if not args.no_display:
            plot_fingerprints(crops)
    except (FingerprintExtractionError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
