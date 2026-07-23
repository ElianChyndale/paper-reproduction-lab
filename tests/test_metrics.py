from __future__ import annotations

import math

import numpy as np
import pytest

from paper_reproduction_lab.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    ndcg_at_k,
    negative_log_likelihood,
    recall_at_k,
    reciprocal_rank,
    risk_coverage,
    sigmoid,
)


def test_retrieval_metrics_are_manually_checkable() -> None:
    ranking = ["distractor", "relevant-a", "relevant-b"]
    relevant = {"relevant-a", "relevant-b"}
    assert recall_at_k(ranking, relevant, 2) == 0.5
    assert reciprocal_rank(ranking, relevant) == 0.5
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranking, relevant, 2) == pytest.approx(expected)


def test_duplicate_ranked_ids_are_ignored() -> None:
    ranking = ["other", "other", "relevant"]
    assert reciprocal_rank(ranking, {"relevant"}) == 0.5
    assert recall_at_k(ranking, {"relevant"}, 2) == 1.0


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_retrieval_metrics_reject_nonpositive_k(metric: object) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        metric(["doc"], {"doc"}, 0)  # type: ignore[operator]


def test_retrieval_metrics_reject_empty_relevance() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        reciprocal_rank(["doc"], set())


def test_binary_probability_metrics() -> None:
    probabilities = np.asarray([0.9, 0.2], dtype=np.float64)
    labels = np.asarray([1, 0], dtype=np.int64)
    assert accuracy(probabilities, labels) == 1.0
    assert brier_score(probabilities, labels) == pytest.approx(0.025)
    expected_nll = -(math.log(0.9) + math.log(0.8)) / 2
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(expected_nll)
    assert expected_calibration_error(probabilities, labels, bins=2) == pytest.approx(0.15)


def test_sigmoid_is_finite_at_extremes() -> None:
    values = sigmoid(np.asarray([-1e9, 0.0, 1e9], dtype=np.float64))
    assert np.all(np.isfinite(values))
    assert values[0] < 1e-20
    assert values[1] == 0.5
    assert values[2] == pytest.approx(1.0)


def test_risk_coverage_points_and_aurc() -> None:
    probabilities = np.asarray([0.99, 0.9, 0.6, 0.55], dtype=np.float64)
    labels = np.asarray([1, 1, 0, 1], dtype=np.int64)
    points, aurc = risk_coverage(probabilities, labels, (0.5, 1.0))
    assert points == {"risk-at-50-coverage": 0.0, "risk-at-100-coverage": 0.25}
    assert aurc == pytest.approx((0.0 + 0.0 + 1 / 3 + 0.25) / 4)


@pytest.mark.parametrize(
    ("probabilities", "labels", "message"),
    [
        (np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int64), "empty"),
        (np.asarray([1.2]), np.asarray([1], dtype=np.int64), r"\[0, 1\]"),
        (np.asarray([0.2]), np.asarray([2], dtype=np.int64), "binary"),
        (np.asarray([np.nan]), np.asarray([1], dtype=np.int64), "finite"),
    ],
)
def test_probability_validation(
    probabilities: np.ndarray,
    labels: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        accuracy(probabilities, labels)


def test_risk_coverage_rejects_invalid_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        risk_coverage(
            np.asarray([0.8], dtype=np.float64),
            np.asarray([1], dtype=np.int64),
            (0.0,),
        )
