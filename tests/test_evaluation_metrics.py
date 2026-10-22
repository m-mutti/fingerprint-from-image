from __future__ import annotations

import math

import numpy as np
import pytest

from fingerprint_from_image.evaluation.metrics import (
    det_points,
    identification_metrics,
    operating_points,
    verification_metrics,
)


def test_operating_points_hand_computed() -> None:
    thresholds, fmr, fnmr = operating_points([0.9, 0.6, 0.35], [0.5, 0.4, 0.3, 0.2])
    assert thresholds.tolist() == [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.9, math.inf]
    assert fmr.tolist() == pytest.approx([1, 0.75, 0.5, 0.5, 0.25, 0, 0, 0])
    assert fnmr.tolist() == pytest.approx([0, 0, 0, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 1])


def test_eer_interpolates_between_operating_points() -> None:
    # Crossing lies between (FMR .5, FNMR 1/3) and (FMR .25, FNMR 1/3): EER = 1/3.
    metrics = verification_metrics([0.9, 0.6, 0.35], [0.5, 0.4, 0.3, 0.2])
    assert metrics.eer == pytest.approx(1 / 3)
    assert metrics.eer_threshold == pytest.approx(0.5)
    # FVC rule: t1 sum 5/6 > t2 sum 7/12, so [EERlow, EERhigh] = [FMR(t2), FNMR(t2)].
    assert (metrics.eer_low, metrics.eer_high) == pytest.approx((0.25, 1 / 3))
    assert metrics.eer_fvc == pytest.approx(7 / 24)
    assert metrics.zero_fmr == pytest.approx(1 / 3)
    assert metrics.zero_fnmr == pytest.approx(0.5)
    assert metrics.fmr100 == pytest.approx(1 / 3)
    assert metrics.n_genuine == 3 and metrics.n_impostor == 4


def test_perfect_separation_and_total_overlap() -> None:
    perfect = verification_metrics([0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
    assert perfect.eer == 0.0 and perfect.eer_fvc == 0.0
    assert perfect.fmr100 == perfect.fmr1000 == perfect.zero_fmr == perfect.zero_fnmr == 0.0
    assert perfect.eer_threshold == pytest.approx(0.7)
    tied = verification_metrics([0.5, 0.5], [0.5, 0.5])
    assert tied.eer == pytest.approx(0.5)
    assert tied.zero_fmr == 1.0 and tied.zero_fnmr == 1.0
    inverted = verification_metrics([0.1, 0.2], [0.8, 0.9])
    assert inverted.eer == pytest.approx(1.0)


def test_exact_crossing_at_an_operating_point() -> None:
    metrics = verification_metrics([0.6, 0.4], [0.5, 0.3])
    assert metrics.eer == pytest.approx(0.5)
    assert metrics.eer_fvc == pytest.approx(0.5)


def test_failures_as_negative_infinity_floor_fnmr() -> None:
    metrics = verification_metrics([-np.inf, 0.9], [0.1])
    assert metrics.eer == pytest.approx(0.5)
    assert metrics.zero_fmr == pytest.approx(0.5)
    payload = metrics.to_dict()
    assert payload["roc"]["threshold"][-1] is None  # +inf serialised as null


def test_fmr_targets_and_resolution() -> None:
    rng = np.random.default_rng(0)
    genuine = rng.normal(2.0, 1.0, 2000)
    impostor = rng.normal(0.0, 1.0, 5000)
    metrics = verification_metrics(genuine, impostor)
    assert metrics.fmr100 == metrics.fnmr_at_fmr[1e-2]
    assert metrics.fmr1000 == metrics.fnmr_at_fmr[1e-3]
    assert metrics.tar_at_far[1e-3] == pytest.approx(1 - metrics.fmr1000)
    assert metrics.fmr100 <= metrics.fmr1000 <= metrics.zero_fmr
    assert metrics.resolvable(1e-3) and not metrics.resolvable(1e-4)
    assert metrics.fmr_resolution == pytest.approx(1 / 5000)
    # EER of two unit Gaussians 2 sigma apart is Phi(-1) ~ 0.1587.
    assert metrics.eer == pytest.approx(0.1587, abs=0.015)
    # FMR100 is attained at some point with FMR <= 1% and no such point does better.
    eligible = metrics.fmr <= 0.01
    assert np.any(eligible & (metrics.fnmr == metrics.fmr100))
    assert metrics.fnmr[eligible].min() == metrics.fmr100


def test_to_dict_subsamples_roc() -> None:
    rng = np.random.default_rng(1)
    metrics = verification_metrics(rng.random(3000), rng.random(3000) - 0.2)
    payload = metrics.to_dict(max_points=64)
    assert len(payload["roc"]["fmr"]) <= 66
    assert payload["roc"]["fmr"][0] == 1.0 and payload["roc"]["fmr"][-1] == 0.0
    assert set(payload["fnmr_at_fmr"]) == {"0.01", "0.001", "0.0001"}


def test_degenerate_inputs_raise() -> None:
    with pytest.raises(ValueError):
        verification_metrics([], [0.1])
    with pytest.raises(ValueError):
        verification_metrics([0.2], [])
    with pytest.raises(ValueError):
        verification_metrics([np.nan], [0.1])
    with pytest.raises(ValueError):
        verification_metrics([np.inf], [0.1])


def test_det_points_are_probit_and_finite() -> None:
    metrics = verification_metrics([0.9, 0.8], [0.1, 0.2])
    x, y = det_points(metrics, eps=1e-3)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert x[0] == pytest.approx(3.0902, abs=1e-3)  # probit(1 - 1e-3)
    with pytest.raises(ValueError):
        det_points(metrics, eps=0.0)


def test_identification_cmc_hand_computed() -> None:
    gallery = ["a", "b", "c"]
    scores = np.array(
        [
            [0.9, 0.1, 0.2],  # a: rank 1
            [0.5, 0.4, 0.1],  # b: rank 2
            [0.3, 0.6, 0.2],  # c: rank 3
            [0.7, 0.7, 0.1],  # a, tied with b: rank 1 or 2
            [0.1, 0.2, 0.3],  # d: not in gallery -> excluded
        ]
    )
    queries = ["a", "b", "c", "a", "d"]
    pessimistic = identification_metrics(scores, queries, gallery)
    assert pessimistic.n_queries == 4 and pessimistic.n_excluded == 1
    assert pessimistic.cmc.tolist() == pytest.approx([0.25, 0.75, 1.0])
    optimistic = identification_metrics(scores, queries, gallery, ties="optimistic")
    assert optimistic.cmc.tolist() == pytest.approx([0.5, 0.75, 1.0])
    expected = identification_metrics(scores, queries, gallery, ties="expected")
    assert expected.cmc.tolist() == pytest.approx([0.375, 0.75, 1.0])
    assert expected.rank1 == pytest.approx(0.375)
    assert expected.rank5 == expected.rank10 == 1.0  # saturates at gallery size


def test_identification_fuses_gallery_templates_per_identity() -> None:
    scores = np.array([[0.2, 0.8, 0.5], [np.nan, 0.1, 0.4]])
    result = identification_metrics(scores, ["a", "b"], ["a", "a", "b"])
    assert result.n_gallery_identities == 2
    assert result.cmc.tolist() == pytest.approx([1.0, 1.0])
    with pytest.raises(ValueError):
        identification_metrics(scores, ["x", "y"], ["a", "a", "b"])
    with pytest.raises(ValueError):
        identification_metrics(scores, ["a"], ["a", "a", "b"])
    with pytest.raises(ValueError):
        identification_metrics(scores, ["a", "b"], ["a", "a", "b"], ties="random")  # type: ignore[arg-type]
