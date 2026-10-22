"""Per-sensor validation on manifest ``val`` identities (model selection signal).

Each val set (FVC2000/2002 DB1–4 DB_B fingers 101–110, neurotech, l3sf, SOCOFing) is
preprocessed once with the runtime pipeline; every evaluation embeds it and reports the
all-pairs EER and FMR100. Selection = mean EER over :data:`SELECTION_SETS` (contact FVC
DB1–3 of 2000/2002 + neurotech).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from training.data import Sample, load_gray, model_input
from training.manifest import SELECTION_SETS, all_pair_scores
from training.metrics import verification_metrics

__all__ = ["Validator"]


def _prep(args: tuple[str, str, bool]) -> np.ndarray:
    path, source, naive = args
    return model_input(load_gray(path, source), naive=naive)


class Validator:
    """Holds preprocessed val inputs; ``__call__(embed)`` returns metrics and raw scores."""

    def __init__(self, groups: dict[str, list[Sample]], *, naive: bool = False,
                 workers: int = 8, selection: Sequence[str] = SELECTION_SETS) -> None:
        self.selection = [s for s in selection if s in groups]
        self.sets: dict[str, tuple[np.ndarray, list[str]]] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, samples in groups.items():
                jobs = [(s.path, s.source, naive) for s in samples]
                inputs = np.stack(list(pool.map(_prep, jobs, chunksize=16)))[:, None]
                self.sets[name] = (inputs.astype(np.float32), [s.label for s in samples])

    def __call__(
        self, embed: Callable[[np.ndarray], np.ndarray]
    ) -> tuple[dict[str, object], dict[str, dict[str, np.ndarray]]]:
        metrics: dict[str, dict[str, float]] = {}
        scores: dict[str, dict[str, np.ndarray]] = {}
        for name, (inputs, labels) in self.sets.items():
            genuine, impostor = all_pair_scores(embed(inputs), labels)
            result = verification_metrics(genuine, impostor)
            metrics[name] = {"eer": result.eer, "fmr100": result.fmr100}
            if name in self.selection:
                scores[name] = {"genuine": genuine, "impostor": impostor}
        select = float(np.mean([metrics[s]["eer"] for s in self.selection]))
        return {"select": select, "sets": metrics}, scores
