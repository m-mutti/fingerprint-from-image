"""Train a fingerprint embedder with a margin loss and GPU acquisition augmentation.

Model selection uses the shared manifests' ``val`` identities: mean all-pairs EER over
contact FVC2000/2002 DB1-3 (fingers 101-110) and neurotech; per-sensor EERs (incl.
FVC DB4, l3sf, SOCOFing) are logged. FVC2004 is never touched here; evaluate it
afterwards with ``python -m training.eval_fvc``.

Usage::

    python -m training.train --config training/configs/convnext_heavy.yaml
    python -m training.train --config ... --set optim.epochs=2 --set name=smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from timm.utils import ModelEmaV3

from training.augment import Augmenter
from training.config import TrainConfig, load_config
from training.data import build_cache
from training.losses import MarginLoss
from training.model import EmbeddingNet
from training.plots import write_plots
from training.sources import gather, sampling_weights
from training.train_utils import (
    GpuMonitor,
    cosine_lr,
    environment,
    load_model,
    save_checkpoint,
    set_seed,
    setup_logging,
)
from training.validate import Validator

logger = logging.getLogger("training.train")


def _optimizer(net: EmbeddingNet, criterion: MarginLoss, cfg: TrainConfig) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, param in net.backbone.named_parameters():
        (no_decay if param.ndim <= 1 or name.endswith(".bias") else decay).append(param)
    head_params = list(net.head.parameters()) + list(criterion.parameters())
    return torch.optim.AdamW([
        {"params": decay, "lr": cfg.optim.lr, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay, "lr": cfg.optim.lr, "weight_decay": 0.0},
        {"params": head_params, "lr": cfg.optim.lr * cfg.optim.head_lr_mult,
         "weight_decay": cfg.optim.weight_decay},
    ], fused=True)


def _distill_loss(teacher: torch.nn.Module, x: torch.Tensor, emb: torch.Tensor,
                  cfg: TrainConfig) -> torch.Tensor:
    """Cosine + relational (in-batch similarity) distillation towards a frozen teacher."""

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        target = teacher(x.contiguous(memory_format=torch.channels_last)).float()
    student = F.normalize(emb.float(), dim=1)
    target = F.normalize(target, dim=1)
    cosine = (1.0 - (student * target).sum(dim=1)).mean()
    relational = F.mse_loss(student @ student.T, target @ target.T)
    return cfg.distill.w_cos * cosine + cfg.distill.w_rel * relational


def _append(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def train(cfg: TrainConfig) -> Path:
    """Run one experiment; returns the run directory."""

    set_seed(cfg.seed)
    run_dir = Path(cfg.out_dir) / f"{cfg.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    setup_logging(run_dir)
    (Path(cfg.out_dir) / "LATEST").write_text(str(run_dir), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment(), indent=2))
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    samples, val_sets, data_stats = gather(cfg)
    logger.info("data: %s", data_stats)
    classes = sorted({s.label for s in samples})
    class_index = {c: i for i, c in enumerate(classes)}
    labels = torch.tensor([class_index[s.label] for s in samples])
    cache = build_cache(samples, Path(cfg.data.cache_dir), canvas=cfg.data.canvas,
                        naive=cfg.data.naive_resize)
    canvases = torch.from_numpy(cache["canvases"])
    backgrounds = torch.from_numpy(cache["background"])
    boxes = torch.from_numpy(cache["valid_box"].astype(np.int64))
    weights = sampling_weights(samples, cfg)
    logger.info("train images=%d classes=%d; val sets=%s", len(samples), len(classes),
                {k: len(v) for k, v in val_sets.items()})

    net = EmbeddingNet(cfg.model, cfg.input_size).to(device).to(memory_format=torch.channels_last)
    criterion = MarginLoss(cfg.loss, cfg.model.dim, len(classes)).to(device)
    ema = ModelEmaV3(net, decay=cfg.optim.ema)  # deep copy BEFORE compiling (see below)
    if cfg.compile != "none":
        # Compile the backbone only: compiling backbone+head as one Inductor graph gives
        # NaN gradients on torch 2.11 / triton-windows 3.7 / sm_120 (EXPERIMENTS.md).
        net.backbone.forward_features = torch.compile(net.backbone.forward_features,
                                                      mode=cfg.compile)
    forward = net
    optimizer = _optimizer(net, criterion, cfg)
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    augment = Augmenter(cfg.aug, cfg.input_size)
    batch = cfg.optim.batch_size
    steps_per_epoch = cfg.optim.samples_per_epoch // batch
    total = steps_per_epoch * cfg.optim.epochs
    warmup = int(cfg.optim.warmup_epochs * steps_per_epoch)
    margin_ramp = cfg.loss.margin_warmup_epochs * steps_per_epoch  # linear per-step ramp
    params = list(net.parameters())
    gpu = GpuMonitor()
    teacher = None
    if cfg.distill.teacher:
        teacher, teacher_cfg = load_model(Path(cfg.distill.teacher), "cuda")
        teacher = teacher.to(memory_format=torch.channels_last)
        if teacher_cfg.model.dim != cfg.model.dim:
            raise ValueError("student and teacher embedding dims must match")

    validator = Validator(val_sets, naive=cfg.data.naive_resize)

    @torch.no_grad()
    def embed(x: np.ndarray) -> np.ndarray:
        model = ema.module.eval()
        out = []
        for i in range(0, len(x), 64):
            chunk = torch.from_numpy(x[i : i + 64]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = model(chunk.contiguous(memory_format=torch.channels_last))
            out.append(F.normalize(emb.float(), dim=1).cpu().numpy())
        return np.concatenate(out)

    best, step, bad_steps = float("inf"), 0, 0
    log_path = run_dir / "metrics.jsonl"
    for epoch in range(cfg.optim.epochs):
        net.train()
        order = torch.multinomial(weights, steps_per_epoch * batch, replacement=True)
        epoch_start = window_start = time.time()
        losses: list[float] = []
        for it in range(steps_per_epoch):
            idx = order[it * batch : (it + 1) * batch]
            canvas = canvases[idx].pin_memory().to(device, non_blocking=True)[:, None]
            x = augment(canvas, backgrounds[idx].to(device, non_blocking=True),
                        boxes[idx].to(device, non_blocking=True))
            y = labels[idx].to(device, non_blocking=True)
            factor = cosine_lr(step, total, warmup)
            criterion.margin_factor = min(1.0, step / max(1.0, margin_ramp))
            for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = base * factor
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = forward(x.contiguous(memory_format=torch.channels_last))
            loss = criterion(emb.float(), y)
            if teacher is not None:
                loss = cfg.distill.w_arc * loss + _distill_loss(teacher, x, emb, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.optim.grad_clip)
            if not torch.isfinite(grad_norm):  # hard guard: skip the update, abort if persistent
                bad_steps += 1
                _append(log_path, {"kind": "nonfinite", "step": step, "loss": float(loss.item()),
                                   "x_finite": bool(torch.isfinite(x).all())})
                if bad_steps > 20:
                    raise FloatingPointError(f"{bad_steps} non-finite steps; see {log_path}")
                step += 1
                continue
            optimizer.step()
            ema.update(net, step=step)
            step += 1
            if step % cfg.log_every == 0:
                value = float(loss.item())
                losses.append(value)
                now = time.time()
                _append(log_path, {
                    "kind": "step", "step": step, "epoch": epoch + 1, "loss": value,
                    "lr": base_lrs[0] * factor, "margin": cfg.loss.margin * criterion.margin_factor,
                    "grad_norm": float(grad_norm), "gpu_util": gpu.utilisation(),
                    "img_per_s": round(cfg.log_every * batch / (now - window_start), 1),
                })
                window_start = now
        record: dict[str, object] = {
            "kind": "epoch", "epoch": epoch + 1, "step": step,
            "loss": float(np.mean(losses)) if losses else None,
            "sec": round(time.time() - epoch_start, 1),
            "img_per_s": round(steps_per_epoch * batch / (time.time() - epoch_start), 1),
        }
        last = epoch + 1 == cfg.optim.epochs
        scores: dict[str, dict[str, np.ndarray]] = {}
        if (epoch + 1) % cfg.eval_every == 0 or last:
            val, scores = validator(embed)
            record["val"] = val
            if val["select"] < best:
                best = val["select"]
                save_checkpoint(run_dir / "best.pt", ema.module, cfg,
                                {"epoch": epoch + 1, "val": val})
        logger.info(json.dumps(record))
        _append(log_path, record)
        write_plots(run_dir, scores or None)
        save_checkpoint(run_dir / "last.pt", ema.module, cfg, {"epoch": epoch + 1})
    logger.info("best val selection EER = %.3f%%", best)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = yaml.safe_load(value)
    train(load_config(args.config, overrides))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
