"""Unified deterministic runner for the three scoped studies."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from paper_reproduction_lab.calibration import run_calibration
from paper_reproduction_lab.io import (
    content_hash,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from paper_reproduction_lab.models import (
    AgentTask,
    CalibrationExample,
    RetrievalDocument,
    RetrievalQuery,
    RunManifest,
    StudyId,
    TrialResult,
)
from paper_reproduction_lab.retrieval import run_retrieval
from paper_reproduction_lab.validation import audit_repository, load_manifests
from paper_reproduction_lab.workflow_agent import run_workflow_agent


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_studies(
    root: Path,
    output: Path,
    selected: list[StudyId] | None = None,
) -> tuple[list[TrialResult], dict[StudyId, list[dict[str, object]]], RunManifest]:
    audit = audit_repository(root)
    if not audit.passed:
        raise ValueError("repository validation failed: " + "; ".join(audit.issues))
    studies = selected or list(StudyId)
    manifests = {manifest.study_id: manifest for manifest in load_manifests(root)}
    all_results: list[TrialResult] = []
    predictions: dict[StudyId, list[dict[str, object]]] = {}

    if StudyId.RETRIEVAL in studies:
        documents = read_jsonl(
            root / "datasets/retrieval/documents.jsonl",
            RetrievalDocument,
        )
        queries = read_jsonl(
            root / "datasets/retrieval/queries.jsonl",
            RetrievalQuery,
        )
        results, rows = run_retrieval(documents, queries)
        all_results.extend(results)
        predictions[StudyId.RETRIEVAL] = rows
    if StudyId.CALIBRATION in studies:
        examples = read_jsonl(
            root / "datasets/calibration/examples.jsonl",
            CalibrationExample,
        )
        results, rows = run_calibration(examples)
        all_results.extend(results)
        predictions[StudyId.CALIBRATION] = rows
    if StudyId.WORKFLOW_AGENT in studies:
        tasks = read_jsonl(
            root / "datasets/workflow-agent/tasks.jsonl",
            AgentTask,
        )
        results, rows = run_workflow_agent(tasks)
        all_results.extend(results)
        predictions[StudyId.WORKFLOW_AGENT] = rows

    expected_methods = {
        method
        for study in studies
        for method in manifests[study].methods
    }
    observed_methods = {result.method for result in all_results}
    if observed_methods != expected_methods:
        raise ValueError("runner output does not cover all declared methods")

    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "trial_results.json",
        [result.model_dump(mode="json") for result in all_results],
    )
    flattened = [
        {
            "study_id": result.study_id.value,
            "method": result.method,
            "metric": metric,
            "value": value,
            "synthetic": True,
        }
        for result in all_results
        for metric, value in sorted(result.metrics.items())
    ]
    write_csv(
        output / "trial_metrics.csv",
        flattened,
        ["study_id", "method", "metric", "value", "synthetic"],
    )
    for study, rows in predictions.items():
        write_jsonl(output / f"{study.value}_predictions.jsonl", rows)

    dataset_paths = {
        "retrieval-documents": root / "datasets/retrieval/documents.jsonl",
        "retrieval-queries": root / "datasets/retrieval/queries.jsonl",
        "calibration-examples": root / "datasets/calibration/examples.jsonl",
        "workflow-agent-tasks": root / "datasets/workflow-agent/tasks.jsonl",
    }
    manifest = RunManifest(
        studies=studies,
        result_count=len(all_results),
        dataset_hashes={name: _file_hash(path) for name, path in dataset_paths.items()},
        config_hashes={
            study.value: content_hash(
                read_json(root / "papers" / study.value / "manifest.json")
            )
            for study in studies
        },
    )
    write_json(output / "run_manifest.json", manifest.model_dump(mode="json"))
    return all_results, predictions, manifest


def load_results(path: Path) -> list[TrialResult]:
    raw = read_json(path / "trial_results.json")
    if not isinstance(raw, list) or not raw:
        raise ValueError("trial_results.json must contain a non-empty array")
    return [TrialResult.model_validate(item) for item in raw]


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))
