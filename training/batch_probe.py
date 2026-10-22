"""Automatic batch-size probe: largest P×K batch whose peak reserved VRAM ≤ a budget.

Runs a few full training steps (GPU augmentation → bf16 fwd/bwd with the backbone
compiled exactly as in training → fused AdamW) for increasing batch sizes and keeps
the largest one whose ``torch.cuda.max_memory_reserved`` stays ≤ ``fraction`` of total
VRAM (default 80 %; on Windows/WDDM, exceeding physical VRAM silently spills to shared
memory instead of raising OOM, which slows and destabilises training). Results are
cached per (backbone, input size, classes, compile mode, GPU, torch) in
``runs/probe/<key>/done.json``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from training.config import TrainConfig
from training.keys import StepDir

__all__ = ["choose_batch", "probe"]

logger = logging.getLogger(__name__)


def probe(cfg: TrainConfig, num_classes: int, candidates: Sequence[int], *,
          fraction: float = 0.8, steps: int = 3) -> dict:
    """Measure peak reserved VRAM and img/s for each candidate (ascending, stop when over)."""

    import torch

    from training.augment import Augmenter
    from training.losses import MarginLoss
    from training.model import EmbeddingNet

    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory
    budget = fraction * total
    net = EmbeddingNet(cfg.model, cfg.input_size).to(device).to(memory_format=torch.channels_last)
    if cfg.compile != "none":
        net.backbone.forward_features = torch.compile(
            net.backbone.forward_features, mode=cfg.compile, dynamic=True)
    criterion = MarginLoss(cfg.loss, cfg.model.dim, num_classes).to(device)
    params = list(net.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5, fused=True)
    augment = Augmenter(cfg.aug, cfg.input_size)
    canvas_side = cfg.data.canvas
    table: list[dict] = []
    for batch in sorted(candidates):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        canvas = torch.randint(0, 256, (batch, 1, canvas_side, canvas_side), dtype=torch.uint8,
                               device=device)
        background = torch.full((batch,), 200.0, device=device)
        labels = torch.randint(0, num_classes, (batch,), device=device)
        try:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(steps + 1):
                if step == 1:  # first step includes (re)compilation
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
            seconds = (time.perf_counter() - start) / steps
            peak = torch.cuda.max_memory_reserved()
            row = {"batch": batch, "peak_reserved_gb": round(peak / 1e9, 2),
                   "fits": peak <= budget, "img_per_s": round(batch / seconds, 1)}
        except torch.cuda.OutOfMemoryError:
            row = {"batch": batch, "peak_reserved_gb": None, "fits": False, "oom": True}
        table.append(row)
        logger.info("batch probe: %s", row)
        del canvas
        if not row["fits"]:
            break
    torch.cuda.empty_cache()
    fitting = [r["batch"] for r in table if r["fits"]]
    return {"total_vram_gb": round(total / 1e9, 2), "budget_fraction": fraction,
            "table": table, "chosen": max(fitting) if fitting else None}


def choose_batch(cfg: TrainConfig, num_classes: int, k: int) -> dict:
    """Cached probe over multiples of ``k`` (32…320); returns the probe record."""

    import torch

    candidates = [k * p for p in range(max(1, 32 // k), 320 // k + 1, max(1, 32 // k))]
    inputs = {"backbone": cfg.model.backbone, "head": cfg.model.head, "dim": cfg.model.dim,
              "input_size": cfg.input_size, "canvas": cfg.data.canvas, "classes": num_classes,
              "compile": cfg.compile, "gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "candidates": candidates, "fraction": 0.8}
    step = StepDir.for_inputs("probe", inputs)
    record = step.record()
    if record is not None and step.is_done():
        return record["probe"]
    started = time.time()
    result = probe(cfg, num_classes, candidates)
    step.path.mkdir(parents=True, exist_ok=True)
    step.mark_done(inputs, [], {"probe": result}, started)
    return result
