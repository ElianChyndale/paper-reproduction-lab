from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paper_reproduction_lab.calibration import (
    fit_platt,
    fit_temperature,
    run_calibration,
)
from paper_reproduction_lab.io import read_jsonl
from paper_reproduction_lab.models import CalibrationExample


def test_temperature_fit_is_positive_and_deterministic() -> None:
    logits = np.asarray([-4.0, -1.0, 1.0, 4.0], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    first = fit_temperature(logits, labels)
    second = fit_temperature(logits, labels)
    assert first == second
    assert 0.25 <= first <= 6.0


def test_temperature_fit_rejects_empty_or_mismatched_data() -> None:
    with pytest.raises(ValueError, match="equal non-zero"):
        fit_temperature(
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="equal non-zero"):
        fit_temperature(
            np.asarray([1.0], dtype=np.float64),
            np.asarray([0, 1], dtype=np.int64),
        )


def test_platt_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        fit_platt(
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([1, 1], dtype=np.int64),
        )


def test_calibration_fixture_improves_at_least_one_ece(repository_root: Path) -> None:
    examples = read_jsonl(
        repository_root / "datasets/calibration/examples.jsonl", CalibrationExample
    )
    first_results, first_predictions = run_calibration(examples)
    second_results, second_predictions = run_calibration(examples)
    assert [item.model_dump() for item in first_results] == [
        item.model_dump() for item in second_results
    ]
    assert first_predictions == second_predictions
    by_method = {result.method: result.metrics for result in first_results}
    raw = by_method["raw-confidence"]["ece-10"]
    assert min(
        by_method["temperature-scaling"]["ece-10"],
        by_method["platt-scaling"]["ece-10"],
    ) < raw
    assert len(first_predictions) == 200


def test_calibration_rejects_duplicate_ids(repository_root: Path) -> None:
    examples = read_jsonl(
        repository_root / "datasets/calibration/examples.jsonl", CalibrationExample
    )
    with pytest.raises(ValueError, match="IDs must be unique"):
        run_calibration([*examples, examples[0]])


def test_calibration_requires_both_splits() -> None:
    example = CalibrationExample(
        example_id="only-test",
        logit=1.0,
        label=1,
        split="test",
        synthetic=True,
    )
    with pytest.raises(ValueError, match="both be non-empty"):
        run_calibration([example])
