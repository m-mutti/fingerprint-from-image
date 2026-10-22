"""Training-step throughput benchmark (data → GPU aug → fwd/bwd → AdamW → EMA).

Times steps with CUDA events after warm-up and samples GPU utilisation with
NVML. Used for the "Throughput" table in ``training/EXPERIMENTS.md``.

Usage::

    python -m training.bench --backbone convnextv2_tiny --batch 64 --compile none
"""

from __future__ import annotations

import argparse
import json
import threading
import time

import numpy as np
import torch
from timm.utils import ModelEmaV3

from training.augment import Augmenter
from training.config import AugConfig, LossConfig, ModelConfig
from training.losses import MarginLoss
from training.model import EmbeddingNet

__all__ = ["main"]


class _UtilSampler(threading.Thread):
    """Background NVML sampler of GPU utilisation (%)."""

    def __init__(self, period: float = 0.05) -> None:
        super().__init__(daemon=True)
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.period = period
        self.samples: list[int] = []
        self.active = False
        self.stop_flag = False

    def run(self) -> None:
        while not self.stop_flag:
            if self.active:
                self.samples.append(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
            time.sleep(self.period)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", default="convnextv2_tiny.fcmae_ft_in22k_in1k")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--classes", type=int, default=5440)
    parser.add_argument("--compile", default="none", choices=["none", "default", "max-autotune",
                                                              "max-autotune-no-cudagraphs"])
    parser.add_argument("--no-fused", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-aug", action="store_true")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=15)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    canvases = torch.randint(0, 256, (4000, 320, 320), dtype=torch.uint8).pin_memory()
    background = torch.full((4000,), 200.0)
    labels = torch.randint(0, args.classes, (4000,))
    net = EmbeddingNet(ModelConfig(backbone=args.backbone, pretrained=False), args.size).to(device)
    if not args.no_channels_last:
        net = net.to(memory_format=torch.channels_last)
    criterion = MarginLoss(LossConfig(), 256, args.classes).to(device)
    ema = ModelEmaV3(net, decay=0.999)
    params = list(net.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, fused=not args.no_fused)
    augment = Augmenter(AugConfig(), args.size)
    if args.compile != "none":  # backbone only, as in train.py
        net.backbone.forward_features = torch.compile(net.backbone.forward_features,
                                                      mode=args.compile)
    forward = net
    sampler = _UtilSampler()
    sampler.start()
    start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = 0.0
    fixed = torch.randn(args.batch, 1, args.size, args.size, device=device)
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sampler.active = True
            start_evt.record()
            wall = time.perf_counter()
        idx = torch.randint(0, 4000, (args.batch,))
        canvas = canvases[idx].pin_memory().to(device, non_blocking=True)[:, None]
        x = fixed if args.no_aug else augment(canvas, background[idx].to(device))
        if not args.no_channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = forward(x)
        loss = criterion(emb.float(), labels[idx].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        ema.update(net, step=step)
    end_evt.record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - wall
    sampler.active = False
    sampler.stop_flag = True
    step_ms = start_evt.elapsed_time(end_evt) / args.steps
    result = {
        "backbone": args.backbone, "batch": args.batch, "compile": args.compile,
        "fused_adamw": not args.no_fused, "channels_last": not args.no_channels_last,
        "aug": not args.no_aug, "step_ms": round(step_ms, 1),
        "img_per_s": round(args.batch * args.steps / wall, 1),
        "gpu_util_pct": round(float(np.mean(sampler.samples)), 1) if sampler.samples else None,
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
    }
    print(json.dumps(result))  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
