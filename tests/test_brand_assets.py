"""Brand-asset generators: well-formed, accessible, small, and deterministic."""

from __future__ import annotations

import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_brand_assets as brand  # noqa: E402
from make_pipeline_match import GALLERY, embeddings  # noqa: E402


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("brand")
    return {path.name: path for path in brand.write_svgs(out)}


def test_every_svg_is_well_formed_xml(generated: dict[str, Path]) -> None:
    for path in generated.values():
        root = ET.parse(path).getroot()
        assert root.tag == "{http://www.w3.org/2000/svg}svg", path.name
        assert root.find("{http://www.w3.org/2000/svg}title") is not None, path.name


def test_size_limits(generated: dict[str, Path]) -> None:
    assert set(generated) == set(brand.SIZE_LIMITS)
    for name, path in generated.items():
        assert path.stat().st_size < brand.SIZE_LIMITS[name], name


@pytest.mark.parametrize("name", brand.ANIMATED)
def test_animations_honour_reduced_motion(generated: dict[str, Path], name: str) -> None:
    text = generated[name].read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in text
    assert "@keyframes" in text


@pytest.mark.parametrize("name", brand.ANIMATED)
def test_every_animation_has_keyframes(generated: dict[str, Path], name: str) -> None:
    text = generated[name].read_text(encoding="utf-8")
    used = set(re.findall(r"animation:([\w-]+) ", text))
    defined = set(re.findall(r"@keyframes ([\w-]+)\{", text))
    assert used, name
    assert used <= defined, sorted(used - defined)


def test_svgs_are_self_contained(generated: dict[str, Path]) -> None:
    for path in generated.values():
        text = path.read_text(encoding="utf-8")
        assert "<script" not in text, path.name
        assert "@import" not in text and "url(http" not in text, path.name
        assert not re.search(r'href="(?!#)', text), path.name


def test_generation_is_deterministic() -> None:
    assert brand.build_svgs() == brand.build_svgs()


def test_committed_svgs_are_up_to_date() -> None:
    for name, content in brand.build_svgs().items():
        committed = ROOT / "assets" / name
        assert committed.read_text(encoding="utf-8") == content, (
            f"{name} is stale: run python scripts/make_brand_assets.py"
        )


def test_displayed_scores_are_real_cosines() -> None:
    query, gallery = embeddings()
    assert math.isclose(sum(x * x for x in query), 1.0, rel_tol=1e-9)
    for (_, shown), vector in zip(GALLERY, gallery, strict=True):
        cosine = sum(a * b for a, b in zip(query, vector, strict=True))
        assert round(cosine, 2) == shown
