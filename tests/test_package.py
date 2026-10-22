from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import fingerprint_extraction
import fingerprint_matching

import fingerprint_from_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_legacy_modules_re_export_the_package_api() -> None:
    for name in fingerprint_from_image.__all__:
        if name == "__version__":
            continue
        assert getattr(fingerprint_matching, name) is getattr(fingerprint_from_image, name)
    legacy_extract = fingerprint_extraction.extract_fingerprints
    assert legacy_extract is fingerprint_from_image.extract_fingerprints


def test_legacy_script_runs_from_an_uninstalled_checkout(tmp_path: Path) -> None:
    # -I ignores site-packages paths set through the environment and the CWD,
    # so the shim must locate ``src/`` itself.
    completed = subprocess.run(
        [sys.executable, "-I", str(PROJECT_ROOT / "fingerprint_matching.py"), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "identify" in completed.stdout


def test_version_is_exposed() -> None:
    assert fingerprint_from_image.__version__.startswith("2.")
