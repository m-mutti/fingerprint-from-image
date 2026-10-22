"""Ablation-table rows from run directories (best-epoch FVC2002 val + FVC2004 test).

For each run the selected checkpoint (``best.pt``, chosen on FVC2002 during
training) is evaluated once on FVC2002 and FVC2004 (cached in
``eval_best.json``)::

    python -m training.summarize RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from training.eval_fvc import main as eval_main

__all__ = ["main"]


def _mean(results: dict, year: str, key: str) -> float:
    return float(np.mean([v[key] for k, v in results.items() if k.startswith(f"FVC{year}")]))


def main(argv: list[str] | None = None) -> int:
    runs = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    print("| run | best epoch | SOCOFing-val EER | FVC2002 EER all | FVC2004 EER fvc | "  # noqa: T201
          "FVC2004 EER all | FVC2004 FMR100 | FVC2004 FMR1000 | FVC2004 per-DB EER all |")
    print("|---|---|---|---|---|---|---|---|---|")  # noqa: T201
    for run in runs:
        out = run / "eval_best.json"
        if not out.is_file():
            eval_main(["--checkpoint", str(run / "best.pt"), "--sets", "FVC2002", "FVC2004",
                       "--json", str(out)])
        results = json.loads(out.read_text(encoding="utf-8"))
        records = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        epochs = [r for r in records if r.get("kind") == "epoch" and "val_fvc2002" in r]
        best = min(epochs, key=lambda r: r["val_fvc2002"]["eer_all"])
        per_db = " / ".join(f"{results[f'FVC2004_DB{i}_B']['eer_all']:.1f}" for i in range(1, 5))
        cells = [
            f"`{run.name}`", str(best["epoch"]), f"{best.get('val_soco_eer', float('nan')):.2f}",
            f"{_mean(results, '2002', 'eer_all'):.2f}", f"{_mean(results, '2004', 'eer_fvc'):.2f}",
            f"**{_mean(results, '2004', 'eer_all'):.2f}**",
            f"{_mean(results, '2004', 'fmr100'):.1f}", f"{_mean(results, '2004', 'fmr1000'):.1f}",
            per_db,
        ]
        print("| " + " | ".join(cells) + " |")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
