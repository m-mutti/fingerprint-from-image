from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from fingerprint_from_image.embedding import preprocess as pp

EMBEDDING_DIR = Path(__file__).resolve().parents[1] / "src" / "fingerprint_from_image" / "embedding"


def _ridges(period: float, size: tuple[int, int] = (300, 260), angle: float = 0.6,
            seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: size[0], : size[1]].astype(np.float32)
    phase = 2 * np.pi * (np.cos(angle) * xx + np.sin(angle) * yy) / period
    ellipse = ((yy - size[0] / 2) / (0.42 * size[0])) ** 2 + (
        (xx - size[1] / 2) / (0.4 * size[1])) ** 2 <= 1
    image = np.full(size, 235.0, dtype=np.float32)
    image[ellipse] = 128 + 90 * np.cos(phase[ellipse])
    image += rng.normal(0, 4, size)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_runtime_package_does_not_import_torch_or_cv2() -> None:
    for path in EMBEDDING_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                assert name.split(".")[0] not in {"torch", "cv2", "tensorflow"}, (path, name)


@pytest.mark.parametrize("period", [3.2, 6.0, 9.5, 14.0])
def test_ridge_period_estimate(period: float) -> None:
    estimate = pp.estimate_ridge_period(_ridges(period))
    assert abs(estimate - period) / period < 0.08


def test_preprocess_normalises_scale_and_intensity() -> None:
    out_small = pp.preprocess(_ridges(4.0, (120, 110)))
    out_large = pp.preprocess(_ridges(10.0, (420, 380)))
    for out in (out_small, out_large):
        assert out.shape == (pp.INPUT_SIZE, pp.INPUT_SIZE) and out.dtype == np.float32
        assert np.isfinite(out).all() and np.abs(out).max() <= pp.CLIP + 1e-6
    # both are rescaled to TARGET_PERIOD: dominant period of the output matches
    for out in (out_small, out_large):
        assert abs(pp.estimate_ridge_period(out) - pp.TARGET_PERIOD) < 0.6


def test_preprocess_is_deterministic_and_handles_blank_images() -> None:
    image = _ridges(8.0)
    np.testing.assert_array_equal(pp.preprocess(image), pp.preprocess(image.copy()))
    blank = pp.preprocess(np.full((64, 64), 200, dtype=np.uint8))
    assert np.isfinite(blank).all()
    with pytest.raises(ValueError):
        pp.preprocess(np.zeros((4, 4), dtype=np.uint8))


def test_crop_around_pads_with_fill() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = pp.crop_around(image, (0.0, 0.0), 4, fill=-1.0)
    assert out[0, 0] == -1.0 and out[2, 2] == image[0, 0]


def _tiny_model(path: Path, dim: int = 8) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    size = pp.INPUT_SIZE
    pool = size // 4
    weight = np.random.default_rng(0).standard_normal((16, dim)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("AveragePool", ["input"], ["pooled"], kernel_shape=[pool, pool],
                             strides=[pool, pool]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "w"], ["embedding"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", 1, size, size])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["N", dim])],
        initializer=[numpy_helper.from_array(weight, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for key, value in {"model_version": "tiny-test", "dim": str(dim),
                       "input_size": str(size), "target_period": str(pp.TARGET_PERIOD)}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))


def test_embedder_with_tiny_onnx_graph(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    from fingerprint_from_image.embedding import FingerprintEmbedder

    model = tmp_path / "tiny.onnx"
    _tiny_model(model)
    embedder = FingerprintEmbedder(model, batch_size=2)
    assert embedder.model_version == "tiny-test" and embedder.dim == 8
    images = [_ridges(p, seed=i) for i, p in enumerate((5.0, 8.0, 11.0))]
    rgb = np.repeat(images[0][..., None], 3, axis=2)
    vectors = embedder.embed(images + [rgb])
    assert vectors.shape == (4, 8) and vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(vectors[0], vectors[3], atol=1e-5)  # RGB == gray input
    np.testing.assert_allclose(embedder.embed(images[1:2])[0], vectors[1], atol=1e-5)
    assert embedder.embed([]).shape == (0, 8)


def test_embedder_rejects_missing_or_mismatched_models(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    from fingerprint_from_image.embedding import FingerprintEmbedder
    from fingerprint_from_image.models import ModelUnavailableError

    with pytest.raises(ModelUnavailableError):
        FingerprintEmbedder(tmp_path / "missing.onnx")
    with pytest.raises(ModelUnavailableError):
        FingerprintEmbedder(model_name="not-registered.onnx")
