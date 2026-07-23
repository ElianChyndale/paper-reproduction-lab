from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_reproduction_lab.cli import run
from paper_reproduction_lab.models import StudyId
from paper_reproduction_lab.reporting import generate_reports
from paper_reproduction_lab.runner import count_csv_rows, load_results, run_studies


def _tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_runner_writes_all_machine_artifacts(
    repository_root: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results"
    results, predictions, manifest = run_studies(repository_root, output)
    assert len(results) == 10
    assert manifest.result_count == 10
    assert set(predictions) == set(StudyId)
    assert sum(len(rows) for rows in predictions.values()) == 380
    assert len(load_results(output)) == 10
    assert count_csv_rows(output / "trial_metrics.csv") > 0
    for path in output.iterdir():
        assert path.stat().st_size > 0


def test_selected_runner_only_emits_selected_study(
    repository_root: Path, tmp_path: Path
) -> None:
    results, predictions, manifest = run_studies(
        repository_root,
        tmp_path / "results",
        [StudyId.RETRIEVAL],
    )
    assert len(results) == 4
    assert set(predictions) == {StudyId.RETRIEVAL}
    assert manifest.studies == [StudyId.RETRIEVAL]


def test_reports_are_byte_deterministic(repository_root: Path, tmp_path: Path) -> None:
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    first = generate_reports(repository_root, results, reports)
    first_hash = _tree_hash(tmp_path)
    second = generate_reports(repository_root, results, reports)
    second_hash = _tree_hash(tmp_path)
    assert first == second
    assert first_hash == second_hash
    assert first["release_gate_passed"] is True
    assert len(list(reports.glob("*.md"))) == 6


def test_cli_validate_run_report_and_audit(
    repository_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["validate", "--root", str(repository_root)]) == 0
    results = tmp_path / "results"
    reports = tmp_path / "reports"
    assert (
        run(
            [
                "run",
                "--root",
                str(repository_root),
                "--study",
                "retrieval",
                "--output",
                str(results),
            ]
        )
        == 0
    )
    assert (
        run(
            [
                "report",
                "--root",
                str(repository_root),
                "--results",
                str(results),
                "--output",
                str(reports),
            ]
        )
        == 0
    )
    assert run(["audit", "--root", str(repository_root)]) == 0
    output = capsys.readouterr().out
    assert '"synthetic": true' in output


def test_cli_returns_error_for_missing_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["validate", "--root", str(tmp_path / "missing")]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is False


def test_load_results_rejects_empty_array(tmp_path: Path) -> None:
    (tmp_path / "trial_results.json").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_results(tmp_path)
