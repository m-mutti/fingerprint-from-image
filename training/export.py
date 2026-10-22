"""Export a checkpoint to ONNX FP32 / ORT-optimised / INT8 (static, dynamic) / FP16.

Each variant is checked for parity (cosine vs the PyTorch FP32 reference on
held-out FVC2002 images), timed on CPU (batch 1 and 32) and evaluated on
FVC2004 DB1-4 B. A variant is "accepted" when its mean EER(all) degrades by
≤ 10 % relative to the ONNX FP32 model. LiteRT files are produced separately
by ``training/export_litert.py`` (runs under WSL).

Usage::

    python -m training.export --checkpoint RUN/best.pt --out E:/.../runs/embedder/release \
        --version ffi-embed-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import torch

from training.data import calibration_inputs
from training.eval_fvc import (
    FVC_GROUPS,
    _onnx_embed,
    evaluate_sets,
    latency,
    load_fvc_inputs,
)
from training.train_utils import load_model

__all__ = ["export_onnx", "main"]



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_onnx(model: torch.nn.Module, path: Path, size: int, metadata: dict[str, str],
                opset: int = 18) -> None:
    """Dynamo-export with a dynamic batch axis, simplify, attach metadata, single file."""

    import onnxsim

    x = torch.zeros(2, 1, size, size)
    batch = torch.export.Dim("batch", min=1, max=1024)
    program = torch.onnx.export(model, (x,), dynamo=True, opset_version=opset,
                                input_names=["input"], output_names=["embedding"],
                                dynamic_shapes={"x": {0: batch}}, verbose=False)
    program.optimize()
    proto = program.model_proto
    simplified, ok = onnxsim.simplify(proto)
    if ok:
        proto = simplified
    del proto.metadata_props[:]
    for key, value in metadata.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, str(path))


class _Reader:
    """onnxruntime CalibrationDataReader over a fixed array (batch 1)."""

    def __init__(self, inputs: np.ndarray, name: str) -> None:
        self._iter = iter([{name: inputs[i : i + 1]} for i in range(len(inputs))])

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)


def quantize_static(src: Path, dst: Path, calib: np.ndarray, method: str) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quant_pre_process,
    )
    from onnxruntime.quantization import quantize_static as ort_quantize_static

    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    ort_quantize_static(
        str(pre), str(dst), _Reader(calib, "input"), quant_format=QuantFormat.QDQ,
        per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8,
        calibrate_method=getattr(CalibrationMethod, method),
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],  # keep LayerNorm/GRN/GELU in float
        extra_options={"CalibMovingAverage": method == "MinMax"},
    )
    pre.unlink(missing_ok=True)


def quantize_dynamic(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic

    ort_quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, per_channel=True)


def to_fp16(src: Path, dst: Path) -> None:
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(src)), keep_io_types=True)
    onnx.save(model, str(dst))


def ort_offline_optimize(src: Path, dst: Path) -> None:
    """Save ORT's extended graph optimisations (fusions, constant folding) as a model file."""

    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    options.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), options, providers=["CPUExecutionProvider"])


def _parity(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    cos = np.sum(reference * other, axis=1) / (
        np.linalg.norm(reference, axis=1) * np.linalg.norm(other, axis=1))
    return {"cos_min": float(cos.min()), "cos_mean": float(cos.mean())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", required=True, help="model_version string")
    parser.add_argument("--prefix", default="fingerprint_embedder_v1")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip", nargs="*", default=[], help="variants to skip")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.checkpoint, "cpu")
    size = cfg.input_size
    metadata = {"model_version": args.version, "dim": str(cfg.model.dim),
                "input_size": str(size), "target_period": "6.0",
                "backbone": cfg.model.backbone, "license": "CC-BY-NC-SA-4.0 (non-commercial)"}
    fp32 = args.out / f"{args.prefix}_fp32.onnx"
    export_onnx(model, fp32, size, metadata)

    variants: dict[str, Path] = {"onnx_fp32": fp32}
    builders = {
        "onnx_fp32_ort_opt": lambda p: ort_offline_optimize(fp32, p),
        "onnx_fp16": lambda p: to_fp16(fp32, p),
        "onnx_int8_dynamic": lambda p: quantize_dynamic(fp32, p),
        "onnx_int8_static_minmax": lambda p: quantize_static(fp32, p, calib, "MinMax"),
        "onnx_int8_static_percentile": lambda p: quantize_static(fp32, p, calib, "Percentile"),
        "onnx_int8_static_entropy": lambda p: quantize_static(fp32, p, calib, "Entropy"),
    }
    calib = calibration_inputs()
    for name, build in builders.items():
        if name in args.skip:
            continue
        path = args.out / f"{args.prefix}_{name.removeprefix('onnx_')}.onnx"
        build(path)
        variants[name] = path

    held_out = np.concatenate([load_fvc_inputs(s)[:16] for s in FVC_GROUPS["FVC2002"]])
    with torch.no_grad():
        reference = model(torch.from_numpy(held_out)).numpy()
    torch_results = evaluate_sets(lambda x: model(torch.from_numpy(x)).detach().numpy(),
                                  FVC_GROUPS["FVC2004"])
    report: dict[str, dict] = {"pytorch_fp32": {"fvc2004": torch_results}}
    for name, path in variants.items():
        embed = _onnx_embed(path, args.threads)
        entry = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2),
                 "sha256": _sha256(path), "parity": _parity(reference, embed(held_out)),
                 "ms_b1": latency(embed, size, batch=1),
                 "ms_b32": latency(embed, size, batch=32, repeats=3),
                 "fvc2004": evaluate_sets(embed, FVC_GROUPS["FVC2004"])}
        report[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "fvc2004"}),  # noqa: T201
              "EER(all) mean", round(entry["fvc2004"]["mean"]["eer_all"], 3))
    base = report["onnx_fp32"]["fvc2004"]["mean"]["eer_all"]
    for entry in report.values():
        if "parity" in entry:
            eer = entry["fvc2004"]["mean"]["eer_all"]
            ok = eer <= base * 1.10 + 1e-9 and entry["parity"]["cos_min"] >= 0.99
            entry["accepted"] = bool(ok)
    (args.out / f"{args.prefix}_export_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
