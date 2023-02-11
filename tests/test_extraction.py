from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from fingerprint_extraction import (
    ExtractionConfig,
    FingerprintExtractionError,
    extract_fingerprint_images,
    extract_fingerprints,
    save_fingerprints,
)


def _synthetic_hand() -> np.ndarray:
    image = np.full((360, 760, 3), (180, 80, 30), dtype=np.uint8)
    skin_bgr = cv.cvtColor(
        np.asarray([[[10, 170, 210]]], dtype=np.uint8), cv.COLOR_HSV2BGR
    )[0, 0]
    for index, x in enumerate((30, 215, 400, 585)):
        cv.ellipse(
            image,
            (x + 70, 180),
            (68, 135 - 5 * index),
            0,
            0,
            360,
            tuple(int(channel) for channel in skin_bgr),
            cv.FILLED,
        )
        for offset in range(-90, 91, 12):
            cv.line(
                image,
                (x + 25, 180 + offset),
                (x + 115, 180 + offset + index),
                (45, 45, 45),
                2,
            )
    return image


def test_extracts_nonempty_crops_in_spatial_order() -> None:
    crops = extract_fingerprints(
        _synthetic_hand(), ExtractionConfig(min_contour_area_ratio=0.002)
    )

    assert len(crops) == 4
    assert [crop.bbox[0] for crop in crops] == sorted(crop.bbox[0] for crop in crops)
    assert all(crop.image.dtype == np.uint8 for crop in crops)
    assert all(crop.image.ndim == 2 and crop.image.size for crop in crops)
    assert all(0.0 <= crop.quality <= 1.0 for crop in crops)


def test_convenience_api_returns_arrays() -> None:
    images = extract_fingerprint_images(
        _synthetic_hand(), ExtractionConfig(min_contour_area_ratio=0.002)
    )
    assert len(images) == 4
    assert all(isinstance(image, np.ndarray) for image in images)


def test_no_skin_contours_returns_empty_list() -> None:
    image = np.full((200, 300, 3), (255, 0, 0), dtype=np.uint8)
    assert extract_fingerprints(image) == []


def test_unreadable_path_has_actionable_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(FingerprintExtractionError, match="Could not read image"):
        extract_fingerprints(missing)


def test_saves_lossless_png_crops(tmp_path: Path) -> None:
    crops = extract_fingerprints(
        _synthetic_hand(), ExtractionConfig(min_contour_area_ratio=0.002)
    )
    save_fingerprints(crops, tmp_path / "output")

    outputs = sorted((tmp_path / "output").glob("finger_*.png"))
    assert len(outputs) == len(crops)
    assert cv.imread(str(outputs[0]), cv.IMREAD_GRAYSCALE) is not None
