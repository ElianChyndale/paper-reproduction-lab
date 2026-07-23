"""Synthetic post-hoc calibration and selective prediction study."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from paper_reproduction_lab.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    risk_coverage,
    sigmoid,
)
from paper_reproduction_lab.models import CalibrationExample, TrialResult


def fit_temperature(logits: NDArray[np.float64], labels: NDArray[np.int64]) -> float:
    if len(logits) != len(labels) or len(logits) == 0:
        raise ValueError("calibration logits and labels must have equal non-zero length")
    candidates = np.geomspace(0.25, 6.0, num=500)
    losses = [
        negative_log_likelihood(sigmoid(logits / temperature), labels)
        for temperature in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def fit_platt(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> tuple[float, float]:
    if len(np.unique(labels)) != 2:
        raise ValueError("Platt scaling requires both binary classes")
    model = LogisticRegression(
        C=1_000_000.0,
        random_state=42,
        solver="lbfgs",
        max_iter=1_000,
    )
    model.fit(logits.reshape(-1, 1), labels)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def calibration_metrics(
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> dict[str, float]:
    risks, aurc = risk_coverage(probabilities, labels)
    return {
        "accuracy": accuracy(probabilities, labels),
        "negative-log-likelihood": negative_log_likelihood(probabilities, labels),
        "brier-score": brier_score(probabilities, labels),
        "ece-10": expected_calibration_error(probabilities, labels, bins=10),
        **risks,
        "aurc": aurc,
        "examples": float(len(labels)),
    }


def run_calibration(
    examples: list[CalibrationExample],
) -> tuple[list[TrialResult], list[dict[str, object]]]:
    identifiers = [example.example_id for example in examples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("calibration example IDs must be unique")
    calibration = [example for example in examples if example.split == "calibration"]
    test = [example for example in examples if example.split == "test"]
    if not calibration or not test:
        raise ValueError("calibration and test splits must both be non-empty")

    calibration_logits = np.asarray([example.logit for example in calibration])
    calibration_labels = np.asarray([example.label for example in calibration], dtype=np.int64)
    test_logits = np.asarray([example.logit for example in test])
    test_labels = np.asarray([example.label for example in test], dtype=np.int64)

    temperature = fit_temperature(calibration_logits, calibration_labels)
    slope, intercept = fit_platt(calibration_logits, calibration_labels)
    probabilities = {
        "raw-confidence": sigmoid(test_logits),
        "temperature-scaling": sigmoid(test_logits / temperature),
        "platt-scaling": sigmoid(slope * test_logits + intercept),
    }
    results: list[TrialResult] = []
    for method, values in probabilities.items():
        metrics = calibration_metrics(values, test_labels)
        if method == "temperature-scaling":
            metrics["fitted-temperature"] = temperature
        if method == "platt-scaling":
            metrics["fitted-slope"] = slope
            metrics["fitted-intercept"] = intercept
        results.append(
            TrialResult(
                result_id=f"calibration-{method}",
                study_id="calibration",
                method=method,
                split="test",
                metrics=metrics,
                synthetic=True,
            )
        )

    predictions: list[dict[str, object]] = [
        {
            "study_id": "calibration",
            "example_id": example.example_id,
            "label": example.label,
            "raw_probability": float(probabilities["raw-confidence"][index]),
            "temperature_probability": float(
                probabilities["temperature-scaling"][index]
            ),
            "platt_probability": float(probabilities["platt-scaling"][index]),
            "synthetic": True,
        }
        for index, example in enumerate(test)
    ]
    return results, predictions
