"""Train a fingerprint embedder as a content-addressed, resumable pipeline step.

Steps: gather data (held-out guard) → keyed canvas cache → batch-size probe (largest
P×K batch with peak VRAM ≤ 80 %) → keyed ``train`` step in ``runs/train/<key[:12]>/``.
The train key covers the resolved config, cache key, code version, seed and held-out
digest, so an identical request is skipped and an interrupted one resumes from
``state.pt``. Selection: mean all-pairs EER over contact FVC2000/2002 DB1–3 (fingers
101–110) + neurotech **val** identities; FVC2004 is never read here.

Usage::

    python -m training.train --config training/configs/m1_fvcfull.yaml
    python -m training.train --config ... --set optim.epochs=2 --force
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from training.config import TrainConfig, load_config
from training.data import build_cache
from training.keys import StepDir, code_version
from training.licences import lineage
from training.plots import write_plots
from training.registry import Registry
from training.sources import gather, source_group, train_shares
from training.train_utils import environment, save_checkpoint, set_seed, setup_logging
from training.trainer import TrainData, Trainer, TrainingAborted
from training.validate import Validator

logger = logging.getLogger("training.train")
_VOLATILE = ("name", "out_dir", "log_every", "num_threads")


def train_inputs(cfg: TrainConfig, cache_key: str, heldout_digest: str, batch: int,
                 accum: int) -> dict:
    """Everything that determines the trained weights (the ``train`` content key)."""

    config = {k: v for k, v in cfg.to_dict().items() if k not in _VOLATILE}
    return {"config": config, "cache": cache_key, "code": code_version(),
            "heldout_digest": heldout_digest, "batch": batch, "accum": accum}


def _resolve_batch(cfg: TrainConfig, num_classes: int) -> tuple[int, int, dict | None]:
    k = cfg.optim.k
    probe = None
    batch = cfg.optim.batch_size
    if batch <= 0:
        from training.batch_probe import choose_batch

        probe = choose_batch(cfg, num_classes, k)
        if not probe["chosen"]:
            raise RuntimeError(f"no batch size fits the VRAM budget: {probe['table']}")
        batch = probe["chosen"]
    batch = max(k, batch - batch % k)
    effective = max(cfg.optim.effective_batch, batch)
    return batch, math.ceil(effective / batch), probe


def train(cfg: TrainConfig, *, force: bool = False) -> Path:
    """Run (or skip / resume) one training step; returns its directory."""

    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    samples, val_sets, stats = gather(cfg)
    classes = sorted({s.label for s in samples})
    index = {c: i for i, c in enumerate(classes)}
    arrays, cache_key = build_cache(samples, canvas=cfg.data.canvas, naive=cfg.data.naive_resize)
    batch, accum, probe = _resolve_batch(cfg, len(classes))
    inputs = train_inputs(cfg, cache_key, stats["heldout_guard"]["heldout_digest"], batch, accum)
    step = StepDir.for_inputs("train", inputs)
    registry = Registry()
    shares = train_shares(cfg)
    licence = lineage(shares, cfg.model.backbone, cfg.model.pretrained)
    if step.is_done() and not force:
        logger.info("train step %s already done: %s", step.key[:12], step.path)
        return step.path
    if force:
        step.invalidate()
    run_dir = step.path
    setup_logging(run_dir)
    (run_dir.parent.parent / "LATEST").write_text(str(run_dir), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment(), indent=2))
    registry.upsert("train", step.key, "running", name=cfg.name, config=cfg.to_dict(),
                    git_commit=inputs["code"]["commit"], licence=licence,
                    artifacts={"dir": str(run_dir)})
    logger.info("train %s: images=%d classes=%d batch=%d accum=%d lr=%.2e data=%s val=%s",
                step.key[:12], len(samples), len(classes), batch, accum,
                cfg.optim.scaled_lr(batch * accum), stats, {k: len(v) for k, v in val_sets.items()})
    started = time.time()
    data = TrainData(torch.from_numpy(arrays["canvases"]), torch.from_numpy(arrays["background"]),
                     torch.from_numpy(arrays["valid_box"].astype(np.int64)),
                     torch.tensor([index[s.label] for s in samples]),
                     [source_group(s) for s in samples])
    heartbeat = Path(os.environ["FFI_HEARTBEAT"]) if os.environ.get("FFI_HEARTBEAT") else None
    trainer = Trainer(cfg, run_dir, data, batch, accum, shares, heartbeat)
    trainer.load_state()
    validator = Validator(val_sets, naive=cfg.data.naive_resize)
    try:
        while trainer.epoch < cfg.optim.epochs:
            record = trainer.train_epoch()
            scores: dict = {}
            if trainer.epoch % cfg.eval_every == 0 or trainer.epoch == cfg.optim.epochs:
                val, scores = validator(trainer.embed)
                record["val"] = val
                if val["select"] < trainer.best:
                    trainer.best = val["select"]
                    save_checkpoint(run_dir / "best.pt", trainer.ema.module, cfg,
                                    {"epoch": trainer.epoch, "val": val, "train_key": step.key})
            save_checkpoint(run_dir / "last.pt", trainer.ema.module, cfg,
                            {"epoch": trainer.epoch, "train_key": step.key})
            trainer.save_state()
            trainer._log(record)
            logger.info(json.dumps(record))
            write_plots(run_dir, scores or None)
    except TrainingAborted as exc:
        registry.upsert("train", step.key, "failed", reason=exc.reason)
        raise
    except Exception as exc:
        reason = "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else type(exc).__name__
        registry.upsert("train", step.key, "failed", reason=f"{reason}: {exc}"[:500])
        raise
    outputs = ["best.pt", "last.pt", "metrics.jsonl"]
    extra = {"best_val_select": trainer.best, "batch": batch, "accum": accum, "probe": probe,
             "data": stats, "licence": licence, "classes": len(classes)}
    step.mark_done(inputs, outputs, extra, started)
    registry.upsert("train", step.key, "done", metrics={"val_select": trainer.best},
                    artifacts={"dir": str(run_dir), "best": str(run_dir / "best.pt")})
    logger.info("best val selection EER = %.3f%% (%s)", trainer.best, run_dir)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--force", action="store_true", help="retrain even if already done")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    train(load_config(args.config, overrides), force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
