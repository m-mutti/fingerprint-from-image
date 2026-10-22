"""Validation-only ablation rows from run directories (``metrics.jsonl``).

Shows the epoch with the best pooled val metric (FNMR @ FMR 0.1 %, one global threshold
over the selection sets), its mean val EER and the per-sensor val EERs. No test
(FVC2004) numbers: those exist only for frozen release candidates
(``python -m training.evaluate --release-candidate``).

Usage::

    python -m training.summarize RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

__all__ = ["main"]

_SETS = ("FVC2000_DB1", "FVC2000_DB2", "FVC2000_DB3", "FVC2002_DB1", "FVC2002_DB2",
         "FVC2002_DB3", "neurotech", "FVC2000_DB4", "FVC2002_DB4", "l3sf", "SOCOFing")


def main(argv: list[str] | None = None) -> int:
    runs = [Path(p) for p in (argv if argv is not None else sys.argv[1:])]
    header = ["run", "best epoch", "val FNMR@FMR0.1% (pooled)", "val mean EER",
              *[f"{s} EER" for s in _SETS]]
    print("| " + " | ".join(header) + " |")  # noqa: T201
    print("|" + "---|" * len(header))  # noqa: T201
    for run in runs:
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        evals = [r for r in rows if r.get("kind") == "epoch" and "val" in r]
        best = min(evals, key=lambda r: (r["val"]["select"], r["val"]["select_eer"]))
        sets = best["val"]["sets"]
        cells = [f"`{run.name}`", str(best["epoch"]), f"{best['val']['select']:.2f}",
                 f"{best['val']['select_eer']:.2f}",
                 *[f"{sets[s]['eer']:.2f}" if s in sets else "-" for s in _SETS]]
        print("| " + " | ".join(cells) + " |")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
