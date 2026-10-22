"""Convert MediaPipe's hand_landmarker.task (Apache-2.0) to ONNX for runtime use.

Runs in a *separate* tools environment (TensorFlow + tf2onnx); the runtime
package only needs numpy, Pillow and onnxruntime. See README.md next to this
file for the pinned versions and expected SHA-256 digests.

    python tools/hand_models/convert.py --out-dir runs/hand_models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
TASK_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MEMBERS = {
    "hand_detector.tflite": "mediapipe_hand_detector.onnx",
    "hand_landmarks_detector.tflite": "mediapipe_hand_landmarks_detector.onnx",
}
OPSET = 17


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_task(destination: Path, local: Path | None) -> Path:
    """Download (or copy) the task bundle and verify its pinned digest."""

    if local is not None:
        shutil.copyfile(local, destination)
    else:
        request = urllib.request.Request(TASK_URL, headers={"User-Agent": "ffi-hand-convert"})
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            destination.write_bytes(response.read())
    digest = sha256(destination)
    if digest != TASK_SHA256:
        raise SystemExit(f"hand_landmarker.task digest {digest} != pinned {TASK_SHA256}")
    return destination


def canonicalize(model: Any) -> Any:
    """Rename nodes/tensors/initializers deterministically.

    tf2onnx numbers generated names from set-iteration order, so two runs give
    byte-different files. Renaming by topological position (graph inputs and
    outputs keep their names) makes the output reproducible.
    """

    graph = model.graph
    keep = {v.name for v in graph.input} | {v.name for v in graph.output}
    initializers = {t.name: t for t in graph.initializer}
    mapping: dict[str, str] = {}
    weight_order: list[str] = []
    for index, node in enumerate(graph.node):
        if any(attr.g.node or attr.graphs for attr in node.attribute):
            raise SystemExit("sub-graphs are not supported by canonicalize()")
        for name in node.input:
            if name in initializers and name not in mapping:
                mapping[name] = f"w{len(weight_order)}"
                weight_order.append(name)
        node.name = f"{node.op_type}_{index}"
        for k, name in enumerate(node.output):
            if name and name not in keep:
                mapping[name] = f"t{index}_{k}"
    for node in graph.node:
        node.input[:] = [mapping.get(n, n) for n in node.input]
        node.output[:] = [mapping.get(n, n) for n in node.output]
    ordered = [initializers[name] for name in weight_order]
    for tensor in ordered:
        tensor.name = mapping[tensor.name]
    del graph.initializer[:]
    graph.initializer.extend(ordered)
    del graph.value_info[:]
    return model


def convert(tflite: Path, onnx_path: Path) -> None:
    import onnx  # tools-only dependencies
    import tf2onnx

    model, _ = tf2onnx.convert.from_tflite(str(tflite), opset=OPSET)
    model = canonicalize(model)
    onnx.checker.check_model(model)
    onnx_path.write_bytes(model.SerializeToString(deterministic=True))


def parity(tflite: Path, onnx_path: Path, trials: int = 3) -> dict[str, float]:
    """Max absolute difference between TFLite and ONNX Runtime outputs."""

    import numpy as np
    import onnxruntime as ort
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite))
    interpreter.allocate_tensors()
    (inp,) = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]
    worst = dict.fromkeys(names, 0.0)
    rng = np.random.default_rng(0)
    for _ in range(trials):
        x = rng.uniform(0.0, 1.0, inp["shape"]).astype(np.float32)
        interpreter.set_tensor(inp["index"], x)
        interpreter.invoke()
        reference = {d["name"]: interpreter.get_tensor(d["index"]) for d in outputs}
        values = session.run(None, {session.get_inputs()[0].name: x})
        for name, value in zip(names, values, strict=True):
            worst[name] = max(worst[name], float(np.abs(reference[name] - value).max()))
    return worst


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task", type=Path, help="use a local hand_landmarker.task")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, object]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        task = fetch_task(work / "hand_landmarker.task", args.task)
        with zipfile.ZipFile(task) as bundle:
            bundle.extractall(work, members=list(MEMBERS))
        for member, onnx_name in MEMBERS.items():
            tflite = work / member
            onnx_path = args.out_dir / onnx_name
            convert(tflite, onnx_path)
            entry: dict[str, object] = {
                "source": f"{TASK_URL}#{member}",
                "source_sha256": sha256(tflite),
                "sha256": sha256(onnx_path),
                "bytes": onnx_path.stat().st_size,
                "opset": OPSET,
                "license": "Apache-2.0",
            }
            if not args.skip_parity:
                entry["parity_max_abs_diff"] = parity(tflite, onnx_path)
            manifest[onnx_name] = entry
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
