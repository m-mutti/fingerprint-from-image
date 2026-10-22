from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fingerprint_from_image import models
from fingerprint_from_image.io import ImageLoadError, load_image, to_gray, to_rgb
from fingerprint_from_image.types import FingerCrop


def test_load_image_gray_and_rgb(tmp_path: Path) -> None:
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    path = tmp_path / "red.png"
    Image.fromarray(rgb).save(path)
    assert load_image(path, mode="RGB").shape == (6, 8, 3)
    gray = load_image(path)
    assert gray.shape == (6, 8) and gray.dtype == np.uint8
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.png")


def test_conversions_handle_dtypes_and_channels() -> None:
    assert to_gray(np.ones((4, 4, 4), dtype=np.uint8)).shape == (4, 4)
    assert to_gray(np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)).max() == 255
    assert to_rgb(np.zeros((3, 3), dtype=np.uint8)).shape == (3, 3, 3)


def test_finger_crop_validation() -> None:
    crop = FingerCrop(np.zeros((5, 5), dtype=np.uint8), label="right_index", quality=0.5)
    assert crop.label == "right_index"
    with pytest.raises(ValueError):
        FingerCrop(np.zeros((5, 5), dtype=np.uint8), label="right_toe")
    with pytest.raises(ValueError):
        FingerCrop(np.zeros((5, 5, 3), dtype=np.uint8))


def test_model_registry_verifies_checksums(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(models.ENV_VAR, str(tmp_path))
    payload = b"model-bytes"
    models.register(models.ModelSpec("unit.onnx", hashlib.sha256(payload).hexdigest()))
    with pytest.raises(models.ModelUnavailableError):
        models.model_path("unit.onnx", download=False)
    (tmp_path / "unit.onnx").write_bytes(payload)
    assert models.model_path("unit.onnx") == tmp_path / "unit.onnx"
    (tmp_path / "unit.onnx").write_bytes(b"tampered")
    with pytest.raises(models.ModelUnavailableError):
        models.model_path("unit.onnx")
