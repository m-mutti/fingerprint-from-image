from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import affine_variant, synthetic_fingerprint
from fingerprint_from_image.evaluation import (
    ClassicScorer,
    EmbeddingScorer,
    FingerprintRecord,
    load_fvc_set,
)
from fingerprint_from_image.evaluation.runner import main, run_fvc_set


class ToyEmbedder:
    """Mean-centred, downsampled pixels; raises on blank images to simulate FTE."""

    model_version = "toy-1"

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def embed(self, images: list[np.ndarray]) -> np.ndarray:
        rows = []
        for image in images:
            if int(image.max()) == int(image.min()):
                raise ValueError("blank image")
            small = image.astype(np.float64)[::4, ::4].reshape(-1)
            small -= small.mean()
            rows.append(self.scale * small / np.linalg.norm(small))
        return np.stack(rows)


def _toy_images(n_fingers: int, n_impressions: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    images = []
    for _finger in range(n_fingers):
        base = rng.integers(0, 256, (32, 32)).astype(np.float64)
        for _impression in range(n_impressions):
            noisy = base + rng.normal(0.0, 20.0, base.shape)
            images.append(np.clip(noisy, 0, 255).astype(np.uint8))
    return images


def _records(n_fingers: int, n_impressions: int) -> list[FingerprintRecord]:
    return [
        FingerprintRecord(Path(f"{f}_{i}.tif"), f"toy/{f}", f"toy/{f}", i, "toy")
        for f in range(101, 101 + n_fingers)
        for i in range(1, n_impressions + 1)
    ]


def _write_fvc_set(folder: Path, images: list[np.ndarray], n_impressions: int) -> None:
    folder.mkdir(parents=True)
    for index, image in enumerate(images):
        finger, impression = 101 + index // n_impressions, 1 + index % n_impressions
        Image.fromarray(image).save(folder / f"{finger}_{impression}.tif")


def test_embedding_run_separates_toy_fingers() -> None:
    images = _toy_images(5, 4)
    scorer = EmbeddingScorer(ToyEmbedder(), batch_size=3)
    result = run_fvc_set(scorer, _records(5, 4), images=images)
    assert result.n_fte == 0 and result.n_failed_comparisons == 0
    assert len(result.genuine_scores) == 5 * 6
    assert len(result.impostor_scores) == 190 - 30
    assert result.verification["fvc"].n_impostor == 10
    assert result.verification["all"].eer == 0.0
    assert result.identification is not None and result.identification.rank1 == 1.0
    assert result.identification.n_queries == 15
    assert scorer.dim == 64 and scorer.describe()["embedder_model_version"] == "toy-1"
    payload = result.to_dict()
    assert payload["scores"]["genuine"][0] == pytest.approx(result.genuine_scores[0])


def test_failure_to_enrol_counts_as_worst_score() -> None:
    images = _toy_images(3, 3)
    images[4] = np.zeros_like(images[4])  # finger 2, impression 2 cannot be embedded
    scorer = EmbeddingScorer(ToyEmbedder(scale=3.0), batch_size=4)
    result = run_fvc_set(scorer, _records(3, 3), images=images)
    assert result.n_fte == 1 and result.fte_rate == pytest.approx(1 / 9)
    assert scorer.failures and "enroll[4]" in scorer.failures[0]
    assert scorer.max_norm_deviation == pytest.approx(2.0)  # renormalised, but reported
    assert np.isneginf(result.genuine_scores).sum() == 2  # (3,4) and (4,5)
    # Two of nine genuine pairs are forced non-matches: FNMR at FMR=0 >= 2/9.
    assert result.verification["all"].zero_fmr >= 2 / 9
    assert result.to_dict()["scores"]["genuine"].count(None) == 2


def test_fvc_only_protocol_skips_identification() -> None:
    result = run_fvc_set(
        EmbeddingScorer(ToyEmbedder()), _records(3, 2), images=_toy_images(3, 2), impostor="fvc"
    )
    assert set(result.verification) == {"fvc"} and result.identification is None
    assert len(result.impostor_scores) == 3


def test_run_rejects_ungrouped_records() -> None:
    records = _records(2, 2)
    with pytest.raises(ValueError):
        run_fvc_set(EmbeddingScorer(ToyEmbedder()), records[::-1][1:] + records[:1])


def test_cli_writes_json_and_markdown(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for name, seed in (("FVC2000_DB1_B", 0), ("FVC2004_DB3_B", 1)):
        _write_fvc_set(tmp_path / "data" / "fvc" / name, _toy_images(3, 3, seed), 3)
    out = tmp_path / "results"
    code = main(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--sets",
            "FVC2000_DB1_B,FVC2004_DB3_B",
            "--scorer",
            "embedding",
            "--embedder",
            "test_evaluation_runner:ToyEmbedder",
            "--embedder-kwargs",
            '{"scale": 1.0}',
            "--tag",
            "toy",
            "--output-dir",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    table = capsys.readouterr().out
    assert "| FVC2004_DB3_B |" in table and "**Average**" in table
    per_set = json.loads((out / "toy_FVC2004_DB3_B.json").read_text(encoding="utf-8"))
    assert per_set["dpi"] == 512 and per_set["scorer"]["name"] == "toy"
    assert per_set["verification"]["fvc"]["n_impostor"] == 3
    summary = json.loads((out / "toy_summary.json").read_text(encoding="utf-8"))
    assert [row["set"] for row in summary["sets"]] == ["FVC2000_DB1_B", "FVC2004_DB3_B"]
    assert summary["average"]["eer_fvc_protocol"] == pytest.approx(0.0)
    assert (out / "toy_summary.md").read_text(encoding="utf-8") == table
    assert load_fvc_set(tmp_path / "data" / "fvc" / "FVC2000_DB1_B")[0].impression == 1


def test_cli_argument_errors(tmp_path: Path) -> None:
    assert main(["--sets", "FVC1999_DB1_B", "-q"]) == 2
    assert main(["--scorer", "embedding", "-q"]) == 2
    assert main(["--data-root", str(tmp_path), "--sets", "FVC2000_DB1_B", "-q"]) == 1


def test_classic_scorer_smoke() -> None:
    images = []
    for seed in (3, 11):
        base = synthetic_fingerprint(seed, size=320)
        images.extend([base, affine_variant(base, angle=4.0, noise_seed=seed)])
    images.append(np.full((320, 320), 128, dtype=np.uint8))  # blank: fails to enrol
    images.append(synthetic_fingerprint(29, size=320))
    scorer = ClassicScorer()
    result = run_fvc_set(scorer, _records(3, 2), images=images, name="synthetic")
    assert result.n_fte == 1
    assert result.genuine_scores[0] > 0.3 and result.genuine_scores[1] > 0.3
    assert np.isneginf(result.genuine_scores[2])
    assert result.compare_us_per_pair > 0 and result.enroll_ms_per_image > 0
    assert scorer.describe()["feature_version"] == "rootsift-v1"
