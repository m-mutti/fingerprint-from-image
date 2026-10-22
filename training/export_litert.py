"""LiteRT (formerly TensorFlow Lite) export: litert-torch FP32 → ai-edge-quantizer FP16/INT8.

Runs under WSL2 Linux (``ai-edge-litert`` has no Windows wheels); point
``FFI_ROOT`` at the Windows tree, e.g.::

    FFI_ROOT=/mnt/e/Personal_tries/Mutti_Finger_Prints PYTHONPATH=src:. \
      ~/.venvs/litert/bin/python -m training.export_litert \
      --checkpoint /mnt/e/.../best.pt --out /mnt/e/.../runs/embedder/release

Variants: ``fp32`` (litert-torch), ``fp16`` (FP16 weights via float casting, FP32
compute), ``int8_dynamic`` (INT8 weights, FP32 activations), ``int8_static``
(INT8 weights + activations) and ``int8_static_a16`` (INT8 weights, INT16
activations), both calibrated on training-split images only. Each is
checked for parity against PyTorch FP32 on held-out FVC2002 images, timed on CPU
and evaluated on FVC2004 DB1-4 B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from training.data import calibration_inputs
from training.eval_fvc import FVC_GROUPS, evaluate_sets, load_fvc_inputs
from training.train_utils import load_model

__all__ = ["main"]

_FP16_RECIPE = [{
    "regex": ".*",
    "operation": "*",
    "algorithm_key": "float_casting",
    "op_config": {
        "weight_tensor_config": {"num_bits": 16, "dtype": "FLOAT"},
        "compute_precision": "FLOAT",
        "explicit_dequantize": True,
    },
}]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Runner:
    """LiteRT interpreter wrapper: (N, 1, S, S) float32 → (N, D) float32 (one image per call)."""

    def __init__(self, path: Path, threads: int) -> None:
        from ai_edge_litert.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        rows = []
        in_scale, in_zero = self.inp["quantization"]
        out_scale, out_zero = self.out["quantization"]
        for i in range(len(x)):
            sample = np.ascontiguousarray(x[i : i + 1])
            if self.inp["dtype"] != np.float32:  # integer IO (fully quantised model)
                info = np.iinfo(self.inp["dtype"])
                sample = np.clip(np.round(sample / in_scale + in_zero), info.min, info.max)
                sample = sample.astype(self.inp["dtype"])
            self.interpreter.set_tensor(self.inp["index"], sample)
            self.interpreter.invoke()
            row = self.interpreter.get_tensor(self.out["index"]).reshape(1, -1)
            if self.out["dtype"] != np.float32:
                row = (row.astype(np.float32) - out_zero) * out_scale
            rows.append(row.astype(np.float32).copy())
        return np.concatenate(rows)


def _latency(run: _Runner, size: int, batch: int, repeats: int) -> float:
    x = np.random.default_rng(0).standard_normal((batch, 1, size, size)).astype(np.float32)
    run(x)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run(x)
        times.append((time.perf_counter() - start) * 1000.0 / batch)
    return float(np.median(times))


def _quantize(fp32: Path, dst: Path, recipe_obj: object, calib: np.ndarray | None) -> None:
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import quantizer

    qt = quantizer.Quantizer(float_model=str(fp32))
    qt.load_quantization_recipe(recipe_obj)
    result = None
    if calib is not None:
        signature = Interpreter(model_path=str(fp32)).get_signature_list()
        key = next(iter(signature))
        name = signature[key]["inputs"][0]
        data = {key: [{name: calib[i : i + 1]} for i in range(len(calib))]}
        result = qt.calibrate(data)
    qt.quantize(result).export_model(str(dst), overwrite=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="fingerprint_embedder_v1")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--calib", type=int, default=256)
    parser.add_argument("--reuse", action="store_true", help="keep existing .tflite files")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    import litert_torch
    from ai_edge_quantizer import recipe

    model, cfg = load_model(args.checkpoint, "cpu")
    size = cfg.input_size
    fp32 = args.out / f"{args.prefix}_litert_fp32.tflite"
    if not (args.reuse and fp32.is_file()):
        litert_torch.convert(model, (torch.zeros(1, 1, size, size),)).export(str(fp32))
    calib = calibration_inputs(args.calib)
    variants = {"litert_fp32": fp32}
    for name, recipe_obj, data in (
        ("litert_fp16", _FP16_RECIPE, None),
        ("litert_int8_dynamic", recipe.dynamic_wi8_afp32(), None),
        ("litert_int8_static", recipe.static_wi8_ai8(), calib),
        ("litert_int8_static_a16", recipe.static_wi8_ai16(), calib),
    ):
        path = args.out / f"{args.prefix}_{name}.tflite"
        if args.reuse and path.is_file():
            variants[name] = path
            continue
        try:
            _quantize(fp32, path, recipe_obj, data)
            variants[name] = path
        except Exception as exc:  # noqa: BLE001 - report and continue with other variants
            print(f"{name}: FAILED {type(exc).__name__}: {exc}")  # noqa: T201

    held_out = np.concatenate([load_fvc_inputs(s)[:16] for s in FVC_GROUPS["FVC2002"]])
    with torch.no_grad():
        reference = model(torch.from_numpy(held_out)).numpy()
    report: dict[str, dict] = {}
    for name, path in variants.items():
        run = _Runner(path, args.threads)
        out = run(held_out)
        cos = np.sum(reference * out, axis=1) / (
            np.linalg.norm(reference, axis=1) * np.linalg.norm(out, axis=1))
        entry = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 2),
                 "sha256": _sha256(path),
                 "parity": {"cos_min": float(cos.min()), "cos_mean": float(cos.mean())},
                 "ms_b1": _latency(run, size, 1, 20), "ms_b32": _latency(run, size, 32, 2),
                 "fvc2004": evaluate_sets(run, FVC_GROUPS["FVC2004"]),
                 "runtime": "ai-edge-litert (XNNPACK, WSL2 Linux)"}
        report[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k != "fvc2004"}),  # noqa: T201
              "EER(all) mean", round(entry["fvc2004"]["mean"]["eer_all"], 3))
    (args.out / f"{args.prefix}_litert_report.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
