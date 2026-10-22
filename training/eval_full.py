"""Held-out test on FVC2004 *full* (DB_A + DB_B, 110 fingers × 8 impressions per DB).

Per DB and subset (``full`` = fingers 1–110, ``A`` = 1–100, ``B`` = 101–110):
genuine = every same-finger impression pair (28 per finger); impostor ``fvc`` = first
impressions across fingers; impostor ``all`` = every cross-finger pair. Reports EER on
both and FMR100/FMR1000/ZeroFMR on ``all``. Images come from the shared manifest
(``split == "heldout"``) and are only read here.

Usage::

    python -m training.eval_full --checkpoint RUN/best.pt --json RUN/eval_fvc2004_full.json
    python -m training.eval_full --onnx release/model.onnx
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from training.config import ROOT
from training.data import load_gray, model_input
from training.eval_fvc import EmbedFn, _onnx_embed, _tflite_embed, _torch_embed
from training.manifest import load_rows
from training.metrics import fvc_pairs, verification_metrics

__all__ = ["evaluate_full", "main"]

SUBSETS = {"full": (1, 110), "A": (1, 100), "B": (101, 110)}


def _prep(args: tuple[str, bool, int]) -> np.ndarray:
    path, naive, size = args
    return model_input(load_gray(path, "fvc"), naive=naive, size=size)


def _metrics(emb: np.ndarray, n_fingers: int) -> dict[str, float]:
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    scores = emb @ emb.T
    out: dict[str, float] = {}
    for impostor in ("fvc", "all"):
        gen, imp = fvc_pairs(n_fingers, 8, impostor=impostor)  # type: ignore[arg-type]
        m = verification_metrics(scores[gen[:, 0], gen[:, 1]], scores[imp[:, 0], imp[:, 1]])
        out[f"eer_{impostor}"] = m.eer
        if impostor == "all":
            out.update(fmr100=m.fmr100, fmr1000=m.fmr1000, zero_fmr=m.zero_fmr)
    return out


def evaluate_full(embed: EmbedFn, *, naive: bool = False, size: int = 256,
                  workers: int = 8) -> dict:
    """Metrics per DB × subset plus means over the four DBs."""

    rows = [r for r in load_rows(ROOT / "data" / "manifests", ["FVC2004"])
            if r["split"] == "heldout"]
    results: dict[str, dict[str, dict[str, float]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for db in range(1, 5):
            chosen = sorted((r for r in rows if r["dataset"] == f"FVC2004_DB{db}"),
                            key=lambda r: (int(r["subject"]), int(r["impression"])))
            if len(chosen) != 880:
                raise ValueError(f"FVC2004_DB{db}: expected 880 images, got {len(chosen)}")
            inputs = np.stack(list(pool.map(_prep, [(r["path"], naive, size) for r in chosen],
                                            chunksize=16)))[:, None].astype(np.float32)
            emb = embed(inputs)
            results[f"FVC2004_DB{db}"] = {
                name: _metrics(emb[(lo - 1) * 8 : hi * 8], hi - lo + 1)
                for name, (lo, hi) in SUBSETS.items()
            }
    keys = next(iter(results.values()))["full"].keys()
    results["mean"] = {name: {k: float(np.mean([results[f"FVC2004_DB{d}"][name][k]
                                                for d in range(1, 5)])) for k in keys}
                       for name in SUBSETS}
    return results


def _format(results: dict) -> str:
    lines = ["| DB | subset | EER fvc % | EER all % | FMR100 % | FMR1000 % | ZeroFMR % |",
             "|---|---|---|---|---|---|---|"]
    for db, subsets in results.items():
        for name, r in subsets.items():
            lines.append(f"| {db} | {name} | {r['eer_fvc']:.2f} | {r['eer_all']:.2f} | "
                         f"{r['fmr100']:.2f} | {r['fmr1000']:.2f} | {r['zero_fmr']:.2f} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--onnx", type=Path)
    group.add_argument("--tflite", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    naive, size = False, 256
    if args.checkpoint:
        embed, naive, size = _torch_embed(args.checkpoint, args.device)
    elif args.onnx:
        embed = _onnx_embed(args.onnx, args.threads)
    else:
        embed = _tflite_embed(args.tflite, args.threads)
    results = evaluate_full(embed, naive=naive, size=size)
    print(_format(results))  # noqa: T201 - CLI output
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
