"""Generate a multi-impression synthetic identity set on CPU.

Usage::

    python -m fingerprint_from_image.data.synth.generate --out data/synth_procedural \\
        --identities 5000 --impressions 8 --seed 0 --workers 8

Every identity ``i`` uses master seed ``seed * 1_000_003 + i`` and impression
seeds ``master_seed * 100 + k``; files are ``{i:05d}/{k}.png`` and
``meta.json`` lists every file with its seed.  Re-running skips files that
already exist (resumable) and always rewrites ``meta.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from .master import make_master
from .render import render_impression

__all__ = ["GENERATOR_VERSION", "generate_identity", "main"]

GENERATOR_VERSION = "1.0"
LICENSE = "generated-in-repo (procedural, no third-party data or weights)"
logger = logging.getLogger(__name__)


def generate_identity(job: tuple[str, int, int, int]) -> list[dict[str, object]]:
    """Render all impressions of one identity; return its meta entries."""

    out_dir, identity, master_seed, impressions = job
    folder = Path(out_dir) / f"{identity:05d}"
    targets = [folder / f"{k}.png" for k in range(1, impressions + 1)]
    entries = [{"file": f"{identity:05d}/{k}.png", "identity": identity, "impression": k,
                "seed": master_seed * 100 + k} for k in range(1, impressions + 1)]
    if all(t.is_file() for t in targets):
        return entries
    master = make_master(master_seed)
    folder.mkdir(parents=True, exist_ok=True)
    for k, target in enumerate(targets, 1):
        if not target.is_file():
            image = render_impression(master, master_seed * 100 + k)
            temporary = target.with_suffix(".tmp.png")
            Image.fromarray(image).save(temporary, optimize=False)
            temporary.replace(target)
    for entry in entries:
        entry["pattern"] = master.pattern
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic fingerprint identities.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--identities", type=int, default=2000)
    parser.add_argument("--impressions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.out), i, args.seed * 1_000_003 + i, args.impressions)
            for i in range(args.identities)]
    images: list[dict[str, object]] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for done, entries in enumerate(pool.map(generate_identity, jobs, chunksize=4), 1):
            images.extend(entries)
            if done % 250 == 0:
                rate = done / (time.time() - start)
                logger.info("%d / %d identities (%.1f/s)", done, len(jobs), rate)
    meta = {"generator": "fingerprint_from_image.data.synth", "version": GENERATOR_VERSION,
            "seed": args.seed, "identities": args.identities, "impressions": args.impressions,
            "dpi": 500, "license": LICENSE, "images": images}
    (args.out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    logger.info("wrote %d images to %s", len(images), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
