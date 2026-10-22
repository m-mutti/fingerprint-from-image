"""``fingerprint-extract`` command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..extraction import (
    ExtractionConfig,
    FingerprintExtractionError,
    extract_fingerprints,
    plot_fingerprints,
    save_fingerprints,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract individual fingerprints from a hand photograph"
    )
    parser.add_argument("image", help="path to the hand image")
    parser.add_argument(
        "--output-dir", help="directory in which to save finger_1.png, ..."
    )
    parser.add_argument(
        "--no-display", action="store_true", help="do not open a preview window"
    )
    parser.add_argument(
        "--max-fingers", type=int, default=4, help="maximum number of crops (default: 4)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        crops = extract_fingerprints(
            args.image, ExtractionConfig(max_fingers=args.max_fingers)
        )
        if not crops:
            raise FingerprintExtractionError("No fingerprints were found in the image")
        if args.output_dir:
            save_fingerprints(crops, args.output_dir)
        if not args.no_display:
            plot_fingerprints(crops)
    except (FingerprintExtractionError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
