from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2 as cv
from conftest import affine_variant, synthetic_fingerprint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHER = PROJECT_ROOT / "fingerprint_matching.py"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MATCHER), *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_build_identify_and_stats_commands(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    alice = synthetic_fingerprint(701)
    bob = synthetic_fingerprint(702)
    (dataset / "alice" / "index").mkdir(parents=True)
    (dataset / "bob" / "index").mkdir(parents=True)
    assert cv.imwrite(str(dataset / "alice" / "index" / "a.png"), alice)
    assert cv.imwrite(str(dataset / "bob" / "index" / "b.png"), bob)
    query = tmp_path / "query.png"
    assert cv.imwrite(str(query), affine_variant(alice, noise_seed=703))
    database = tmp_path / "fingerprints.db"

    built = _run("build", str(dataset), "--db", str(database))
    assert built.returncode == 0, built.stderr
    assert json.loads(built.stdout)["enrollment"]["added"] == 2

    identified = _run(
        "identify",
        str(query),
        "--cropped",
        "--top-k",
        "2",
        "--db",
        str(database),
    )
    assert identified.returncode == 0, identified.stderr
    payload = json.loads(identified.stdout)
    assert payload["matches"][0]["identity"] == "alice"
    assert payload["matches"][0]["accepted"]

    stats = _run("stats", "--db", str(database))
    assert stats.returncode == 0, stats.stderr
    assert json.loads(stats.stdout)["stats"]["templates"] == 2


def test_read_command_does_not_create_a_missing_database(tmp_path: Path) -> None:
    database = tmp_path / "missing.db"
    query = tmp_path / "query.png"
    assert cv.imwrite(str(query), synthetic_fingerprint(801))

    result = _run("identify", str(query), "--cropped", "--db", str(database))

    assert result.returncode == 2
    assert "does not exist" in result.stderr
    assert not database.exists()


def test_failed_replacement_preserves_the_existing_database(tmp_path: Path) -> None:
    original = tmp_path / "original"
    (original / "alice" / "index").mkdir(parents=True)
    assert cv.imwrite(
        str(original / "alice" / "index" / "a.png"), synthetic_fingerprint(901)
    )
    database = tmp_path / "replace.db"
    assert _run("build", str(original), "--db", str(database)).returncode == 0

    invalid = tmp_path / "invalid"
    (invalid / "bob" / "index").mkdir(parents=True)
    assert cv.imwrite(
        str(invalid / "bob" / "index" / "blank.png"),
        synthetic_fingerprint(902) * 0 + 255,
    )
    replacement = _run(
        "build", str(invalid), "--replace", "--db", str(database)
    )
    assert replacement.returncode == 1
    assert json.loads(replacement.stdout)["replacement_committed"] is False

    stats_result = _run("stats", "--db", str(database))
    assert stats_result.returncode == 0
    stats = json.loads(stats_result.stdout)["stats"]
    assert stats["templates_per_identity"] == {"alice": 1}

    empty = tmp_path / "empty"
    (empty / "charlie").mkdir(parents=True)
    empty_replacement = _run(
        "build", str(empty), "--replace", "--db", str(database)
    )
    assert empty_replacement.returncode == 2
    stats = json.loads(_run("stats", "--db", str(database)).stdout)["stats"]
    assert stats["templates_per_identity"] == {"alice": 1}


def test_successful_replacement_commits_the_staged_database(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "alice" / "index").mkdir(parents=True)
    (second / "bob" / "thumb").mkdir(parents=True)
    assert cv.imwrite(
        str(first / "alice" / "index" / "a.png"), synthetic_fingerprint(910)
    )
    assert cv.imwrite(
        str(second / "bob" / "thumb" / "b.png"), synthetic_fingerprint(911)
    )
    database = tmp_path / "successful-replace.db"
    assert _run("build", str(first), "--db", str(database)).returncode == 0

    replacement = _run(
        "build", str(second), "--replace", "--db", str(database)
    )

    assert replacement.returncode == 0, replacement.stderr
    assert json.loads(replacement.stdout)["replacement_committed"] is True
    stats = json.loads(_run("stats", "--db", str(database)).stdout)["stats"]
    assert stats["templates_per_identity"] == {"bob": 1}
