"""FVC DB_B verification evaluation for PyTorch checkpoints, ONNX and LiteRT models.

Protocol per FVC set (10 fingers x 8 impressions): genuine = 280 same-finger
pairs; impostor "fvc" = first impressions across fingers (45); impostor "all" =
every cross-finger pair (2880). Reports EER (both impostor sets), FMR100 and
FMR1000 (on "all"), plus embedding and cosine-scoring latency.

Usage::

    python -m training.eval_fvc --checkpoint runs/.../best.pt --sets FVC2004
    python -m training.eval_fvc --onnx model.onnx --sets FVC2004 --json out.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from training.config import ROOT
from training.data import model_input
from training.metrics import fvc_pairs, verification_metrics

__all__ = ["EmbedFn", "evaluate_sets", "fvc_scores", "load_fvc_inputs", "main"]

EmbedFn = Callable[[np.ndarray], np.ndarray]  # (N, 1, S, S) float32 → (N, D) unit norm
DATA = ROOT / "data" / "fvc"  # override the root with $FFI_ROOT (e.g. /mnt/e/... under WSL)
FVC_GROUPS = {
    "FVC2000": [f"FVC2000_DB{i}_B" for i in range(1, 5)],
    "FVC2002": [f"FVC2002_DB{i}_B" for i in range(1, 5)],
    "FVC2004": [f"FVC2004_DB{i}_B" for i in range(1, 5)],
}


def load_fvc_inputs(name: str, *, root: Path = DATA, naive: bool = False) -> np.ndarray:
    """Preprocessed ``(80, 1, S, S)`` inputs ordered finger-major (101_1 … 110_8)."""

    arrays = []
    for finger in range(101, 111):
        for impression in range(1, 9):
            with Image.open(root / name / f"{finger}_{impression}.tif") as handle:
                gray = np.asarray(handle.convert("L"), dtype=np.uint8)
            arrays.append(model_input(gray, naive=naive))
    return np.stack(arrays)[:, None].astype(np.float32)


def fvc_scores(embeddings: np.ndarray, impostor: str = "all") -> tuple[np.ndarray, np.ndarray]:
    """(genuine, impostor) cosine scores for one FVC set's (80, D) embeddings."""

    emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = emb @ emb.T
    gen, imp = fvc_pairs(impostor=impostor)  # type: ignore[arg-type]
    return scores[gen[:, 0], gen[:, 1]], scores[imp[:, 0], imp[:, 1]]


def evaluate_embeddings(
    embeddings: np.ndarray, collect: dict[str, np.ndarray] | None = None
) -> dict[str, float]:
    """Metrics for one FVC set from its (80, D) embeddings (optionally keep raw scores)."""

    out: dict[str, float] = {}
    for impostor in ("fvc", "all"):
        genuine, impostors = fvc_scores(embeddings, impostor)
        if collect is not None and impostor == "all":
            collect["genuine"], collect["impostor"] = genuine, impostors
        metrics = verification_metrics(genuine, impostors)
        out[f"eer_{impostor}"] = metrics.eer
        if impostor == "all":
            out["fmr100"] = metrics.fmr100
            out["fmr1000"] = metrics.fmr1000
            out["zero_fmr"] = metrics.zero_fmr
    return out


def evaluate_sets(
    embed: EmbedFn, sets: Sequence[str], *, naive: bool = False,
    cache: dict[str, np.ndarray] | None = None,
    scores: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate ``embed`` on each FVC set; adds a ``mean`` row (raw scores into ``scores``)."""

    results: dict[str, dict[str, float]] = {}
    for name in sets:
        inputs = cache[name] if cache is not None and name in cache else load_fvc_inputs(
            name, naive=naive)
        if cache is not None:
            cache[name] = inputs
        collect: dict[str, np.ndarray] | None = {} if scores is not None else None
        results[name] = evaluate_embeddings(embed(inputs), collect)
        if scores is not None and collect is not None:
            scores[name] = collect
    keys = next(iter(results.values())).keys()
    results["mean"] = {k: float(np.mean([r[k] for r in results.values()])) for k in keys}
    return results


def _torch_embed(checkpoint: Path, device: str) -> tuple[EmbedFn, bool]:
    import torch

    from training.train_utils import load_model

    model, cfg = load_model(checkpoint, device)

    @torch.no_grad()
    def embed(x: np.ndarray) -> np.ndarray:
        outs = [model(torch.from_numpy(x[i : i + 32]).to(device)).float().cpu().numpy()
                for i in range(0, len(x), 32)]
        return np.concatenate(outs)

    return embed, bool(cfg.data.naive_resize)


def _onnx_embed(path: Path, threads: int) -> EmbedFn:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    return lambda x: np.concatenate(
        [session.run(None, {name: x[i : i + 16]})[0] for i in range(0, len(x), 16)])


def _tflite_embed(path: Path, threads: int) -> EmbedFn:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # pragma: no cover - fallback for TF installs
        from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path), num_threads=threads)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    nhwc = inp["shape"][-1] == 1 and inp["shape"][1] != 1

    def embed(x: np.ndarray) -> np.ndarray:
        rows = []
        for i in range(len(x)):
            sample = x[i : i + 1]
            if nhwc:
                sample = sample.transpose(0, 2, 3, 1)
            interpreter.set_tensor(inp["index"], np.ascontiguousarray(sample))
            interpreter.invoke()
            rows.append(interpreter.get_tensor(out["index"]).reshape(1, -1).copy())
        return np.concatenate(rows)

    return embed


def with_tta(embed: EmbedFn, angles: Sequence[float]) -> EmbedFn:
    """Average unit embeddings over rotated copies of each input (rotation TTA)."""

    from fingerprint_from_image.embedding.preprocess import rotate_input

    def wrapped(x: np.ndarray) -> np.ndarray:
        total = None
        for angle in angles:
            rotated = x if angle == 0 else np.stack(
                [rotate_input(img[0], angle) for img in x])[:, None]
            emb = embed(np.ascontiguousarray(rotated))
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            total = emb if total is None else total + emb
        return total / np.linalg.norm(total, axis=1, keepdims=True)

    return wrapped


def latency(embed: EmbedFn, size: int, *, batch: int = 1, repeats: int = 20) -> float:
    """Median milliseconds per image for ``embed`` on random inputs."""

    x = np.random.default_rng(0).standard_normal((batch, 1, size, size)).astype(np.float32)
    embed(x)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        embed(x)
        times.append((time.perf_counter() - start) * 1000.0 / batch)
    return float(np.median(times))


def score_latency_us(dim: int, gallery: int = 10000) -> float:
    """Microseconds to score one probe against ``gallery`` templates (one mat-vec)."""

    rng = np.random.default_rng(0)
    g = rng.standard_normal((gallery, dim)).astype(np.float32)
    q = rng.standard_normal(dim).astype(np.float32)
    times = []
    for _ in range(50):
        start = time.perf_counter()
        _ = g @ q
        times.append((time.perf_counter() - start) * 1e6)
    return float(np.median(times))


def _format(results: dict[str, dict[str, float]]) -> str:
    lines = ["| set | EER fvc % | EER all % | FMR100 % | FMR1000 % | ZeroFMR % |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['eer_fvc']:.2f} | {r['eer_all']:.2f} | {r['fmr100']:.2f} "
                     f"| {r['fmr1000']:.2f} | {r['zero_fmr']:.2f} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--onnx", type=Path)
    group.add_argument("--tflite", type=Path)
    parser.add_argument("--sets", nargs="+", default=["FVC2004"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--latency", action="store_true", help="also time CPU embedding")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--tta", type=float, nargs="*", default=None,
                        help="rotation angles for test-time augmentation, e.g. 0 -10 10")
    args = parser.parse_args(argv)

    sets = [s for group_name in args.sets for s in FVC_GROUPS.get(group_name, [group_name])]
    naive = False
    if args.checkpoint:
        embed, naive = _torch_embed(args.checkpoint, args.device)
    elif args.onnx:
        embed = _onnx_embed(args.onnx, args.threads)
    else:
        embed = _tflite_embed(args.tflite, args.threads)
    if args.tta:
        embed = with_tta(embed, args.tta)
    results: dict[str, object] = dict(evaluate_sets(embed, sets, naive=naive))
    print(_format(results))  # noqa: T201 - CLI output
    if args.latency:
        size = load_fvc_inputs(sets[0])[0].shape[-1]
        timing = {"ms_per_image_b1": latency(embed, size, batch=1),
                  "ms_per_image_b32": latency(embed, size, batch=32, repeats=5)}
        dim = embed(load_fvc_inputs(sets[0])[:1]).shape[1]
        timing["score_us_per_probe_vs_10k"] = score_latency_us(dim)
        results["latency"] = timing
        print(json.dumps(timing, indent=2))  # noqa: T201
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
