"""Small, manually testable metrics used by the three studies."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def recall_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("relevant documents must not be empty")
    unique = list(dict.fromkeys(ranking))
    return len(set(unique[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranking: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        raise ValueError("relevant documents must not be empty")
    for rank, document_id in enumerate(dict.fromkeys(ranking), start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranking: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("relevant documents must not be empty")
    unique = list(dict.fromkeys(ranking))
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(unique[:k], start=1)
        if document_id in relevant
    )
    ideal_count = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal


def sigmoid(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def accuracy(probabilities: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    _check_probabilities(probabilities, labels)
    return float(np.mean((probabilities >= 0.5) == labels))


def negative_log_likelihood(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    _check_probabilities(probabilities, labels)
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def brier_score(probabilities: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    _check_probabilities(probabilities, labels)
    return float(np.mean((probabilities - labels) ** 2))


def expected_calibration_error(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    bins: int = 10,
) -> float:
    _check_probabilities(probabilities, labels)
    if bins <= 0:
        raise ValueError("bins must be positive")
    predictions = probabilities >= 0.5
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = predictions == labels
    boundaries = np.linspace(0.5, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == bins - 1 else confidence < upper
        )
        if not np.any(selected):
            continue
        weight = float(np.mean(selected))
        gap = abs(float(np.mean(correctness[selected])) - float(np.mean(confidence[selected])))
        error += weight * gap
    return error


def risk_coverage(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
) -> tuple[dict[str, float], float]:
    _check_probabilities(probabilities, labels)
    if any(coverage <= 0.0 or coverage > 1.0 for coverage in coverages):
        raise ValueError("coverage values must be in (0, 1]")
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    correctness = (probabilities >= 0.5) == labels
    order = np.argsort(-confidence, kind="stable")
    points: dict[str, float] = {}
    for coverage in coverages:
        count = max(1, math.ceil(coverage * len(labels)))
        risk = 1.0 - float(np.mean(correctness[order[:count]]))
        points[f"risk-at-{int(coverage * 100)}-coverage"] = risk
    all_risks = [
        1.0 - float(np.mean(correctness[order[:count]]))
        for count in range(1, len(labels) + 1)
    ]
    return points, float(np.mean(all_risks))


def _check_probabilities(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> None:
    if probabilities.ndim != 1 or labels.ndim != 1 or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must be equally sized vectors")
    if len(labels) == 0:
        raise ValueError("probabilities and labels must not be empty")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("labels must be binary")
