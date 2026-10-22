"""Automatic batch-size probe: largest P×K batch whose peak reserved VRAM ≤ 80 % of total.

The probe runs in a **subprocess** (complete teardown of compiled graphs, allocator
cache and CUDA context before training starts) and compiles the backbone exactly as
training does (static shapes, same ``torch.compile`` mode). Candidates are tried in
ascending order; once two points are measured, peak memory is extrapolated linearly and
a candidate predicted to exceed the budget is **not run**. The process is also capped
with ``set_per_process_memory_fraction`` so an over-budget allocation raises OOM
instead of silently spilling into WDDM shared memory. Results are cached in
``runs/probe/<key[:12]>/done.json``.

CLI (used internally)::

    python -m training.batch_probe --config cfg.yaml --classes 5585 --candidates 32 64 96
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import yaml

from training.config import TrainConfig, load_config
from training.keys import StepDir

__all__ = ["choose_batch", "probe"]

logger = logging.getLogger(__name__)
FRACTION = 0.8


def probe(cfg: TrainConfig, num_classes: int, candidates: Sequence[int], *,
          fraction: float = FRACTION, steps: int = 3) -> dict:
    """In-process measurement (call via :func:`choose_batch`, which isolates it)."""

    import torch

    from training.augment import Augmenter
    from training.losses import MarginLoss
    from training.model import EmbeddingNet

    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    budget = fraction * total
    torch.cuda.set_per_process_memory_fraction(min(0.95, fraction + 0.05))
    torch.backends.cudnn.benchmark = True
    net = EmbeddingNet(cfg.model, cfg.input_size).to(device).to(memory_format=torch.channels_last)
    if cfg.compile != "none":
        net.backbone.forward_features = torch.compile(net.backbone.forward_features,
                                                      mode=cfg.compile)
    criterion = MarginLoss(cfg.loss, cfg.model.dim, num_classes).to(device)
    params = list(net.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5, fused=True)
    augment = Augmenter(cfg.aug, cfg.input_size)
    table: list[dict] = []
    for batch in sorted(candidates):
        measured = [r for r in table if r.get("peak_reserved_gb")]
        if len(measured) >= 2:
            (b0, m0), (b1, m1) = [(r["batch"], r["peak_reserved_gb"] * 1e9) for r in measured[-2:]]
            predicted = m1 + (m1 - m0) / (b1 - b0) * (batch - b1)
            if predicted > budget:
                table.append({"batch": batch, "predicted_gb": round(predicted / 1e9, 2),
                              "fits": False, "skipped": True})
                break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        canvas = torch.randint(0, 256, (batch, 1, cfg.data.canvas, cfg.data.canvas),
                               dtype=torch.uint8, device=device)
        background = torch.full((batch,), 200.0, device=device)
        labels = torch.randint(0, num_classes, (batch,), device=device)
        try:
            for step in range(steps + 1):
                if step == 1:  # step 0 includes compilation
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                x = augment(canvas, background).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    emb = net(x)
                loss = criterion(emb.float(), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved()
            row = {"batch": batch, "peak_reserved_gb": round(peak / 1e9, 2),
                   "fits": peak <= budget,
                   "img_per_s": round(batch * steps / (time.perf_counter() - start), 1)}
        except torch.cuda.OutOfMemoryError:
            row = {"batch": batch, "fits": False, "oom": True}
        table.append(row)
        if not row["fits"]:
            break
    fitting = [r["batch"] for r in table if r["fits"]]
    return {"total_vram_gb": round(total / 1e9, 2), "budget_fraction": fraction,
            "table": table, "chosen": max(fitting) if fitting else None}


def candidates_for(max_batch: int, k: int, step: int = 32) -> list[int]:
    """Ascending multiples of ``k`` (every ``step`` images) up to ``max_batch``."""

    out = list(range(step, max_batch + 1, step))
    if max_batch not in out:
        out.append(max_batch)
    return sorted({max(k, b - b % k) for b in out})


def choose_batch(cfg: TrainConfig, num_classes: int, k: int, max_batch: int) -> dict:
    """Cached, subprocess-isolated probe; returns the probe record."""

    import torch

    candidates = candidates_for(max_batch, k)
    inputs = {"backbone": cfg.model.backbone, "head": cfg.model.head, "dim": cfg.model.dim,
              "input_size": cfg.input_size, "canvas": cfg.data.canvas, "classes": num_classes,
              "compile": cfg.compile, "loss": cfg.loss.kind, "subcenters": cfg.loss.subcenters,
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "candidates": candidates, "fraction": FRACTION}
    step = StepDir.for_inputs("probe", inputs)
    record = step.record()
    if record is not None and step.is_done():
        return record["probe"]
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "probe.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
        out = subprocess.run(
            [sys.executable, "-m", "training.batch_probe", "--config", str(cfg_path),
             "--classes", str(num_classes), "--candidates", *map(str, candidates)],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[1])
    result = json.loads(out.stdout.strip().splitlines()[-1])
    step.path.mkdir(parents=True, exist_ok=True)
    step.mark_done(inputs, [], {"probe": result}, started)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--candidates", type=int, nargs="+", required=True)
    args = parser.parse_args(argv)
    result = probe(load_config(args.config), args.classes, args.candidates)
    print(json.dumps(result))  # noqa: T201 - read by the parent process
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
