"""Run the hand -> fingertip pipeline on images and save annotated outputs.

    python tools/hand_models/demo.py --model-dir runs/hand_models \
        --out-dir runs/hand_demo tests/data/*.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from fingerprint_from_image.hand import (
    DetectorConfig,
    HandFingerDetector,
    crop_mosaic,
    draw_hands,
)
from fingerprint_from_image.io import load_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mirrored", action="store_true")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5, help="timing repetitions")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = HandFingerDetector(
        args.model_dir, config=DetectorConfig(max_hands=args.max_hands, mirrored=args.mirrored))
    report = {}
    for path in args.images:
        rgb = load_image(path, mode="RGB")
        detector(rgb)  # warm-up
        timings: dict[str, list[float]] = {"hands_ms": [], "crops_ms": []}
        for _ in range(args.repeats):
            start = time.perf_counter()
            hands = detector.detect_hands(rgb)
            middle = time.perf_counter()
            crops = [c for hand in hands for c in detector.finger_crops(rgb, hand)]
            timings["hands_ms"].append((middle - start) * 1e3)
            timings["crops_ms"].append((time.perf_counter() - middle) * 1e3)
        stem = path.stem
        draw_hands(rgb, hands, crops).save(args.out_dir / f"{stem}_annotated.jpg", quality=90)
        if crops:
            crop_mosaic(crops).save(args.out_dir / f"{stem}_crops.png")
        for index, crop in enumerate(crops):
            Image.fromarray(crop.image).save(args.out_dir / f"{stem}_{index}_{crop.label}.png")
        report[path.name] = {
            "size": list(rgb.shape[:2]),
            "median_hands_ms": float(np.median(timings["hands_ms"])),
            "median_crops_ms": float(np.median(timings["crops_ms"])),
            "hands": [
                {"side": h.side, "side_confidence": round(h.side_confidence, 3),
                 "handedness_score": round(h.handedness_score, 3),
                 "presence": round(h.presence, 3), "palm_score": round(h.palm_score, 3),
                 "palm_facing": h.palm_facing}
                for h in hands
            ],
            "crops": [
                {"label": c.label, "quality": round(c.quality, 3), "shape": list(c.image.shape),
                 "center": [round(float(v), 1) for v in np.mean(c.polygon, axis=0)]}
                for c in crops
            ],
        }
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
