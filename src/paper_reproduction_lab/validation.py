"""Manifest, schema, dataset, and claim-scope validation."""

from __future__ import annotations

import csv
import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from paper_reproduction_lab.io import DataError, read_json, read_jsonl
from paper_reproduction_lab.models import (
    AgentTask,
    AuditSummary,
    CalibrationExample,
    RetrievalDocument,
    RetrievalQuery,
    RunManifest,
    StudyId,
    StudyManifest,
    TrialResult,
)

EXPECTED_METHODS = {
    StudyId.RETRIEVAL: {"bm25", "lsa", "rrf-hybrid", "feature-reranker"},
    StudyId.CALIBRATION: {
        "raw-confidence",
        "temperature-scaling",
        "platt-scaling",
    },
    StudyId.WORKFLOW_AGENT: {
        "deterministic-workflow",
        "heuristic-single-agent",
        "constrained-multi-agent",
    },
}


def load_manifests(root: Path) -> list[StudyManifest]:
    manifests = []
    for study_id in StudyId:
        path = root / "papers" / study_id.value / "manifest.json"
        manifests.append(StudyManifest.model_validate(read_json(path)))
    return manifests


def _unique_ids(identifiers: list[str], label: str) -> list[str]:
    duplicates = sorted(
        identifier for identifier in set(identifiers) if identifiers.count(identifier) > 1
    )
    return [f"duplicate {label} IDs: {', '.join(duplicates)}"] if duplicates else []


def audit_repository(root: Path) -> AuditSummary:
    issues: list[str] = []
    records = 0
    try:
        manifests = load_manifests(root)
        documents = read_jsonl(
            root / "datasets" / "retrieval" / "documents.jsonl",
            RetrievalDocument,
        )
        queries = read_jsonl(
            root / "datasets" / "retrieval" / "queries.jsonl",
            RetrievalQuery,
        )
        calibration = read_jsonl(
            root / "datasets" / "calibration" / "examples.jsonl",
            CalibrationExample,
        )
        tasks = read_jsonl(
            root / "datasets" / "workflow-agent" / "tasks.jsonl",
            AgentTask,
        )
    except (DataError, ValueError) as exc:
        return AuditSummary(
            manifests=0,
            datasets=0,
            records=0,
            issues=[str(exc)],
            passed=False,
        )

    if len(manifests) != 3 or {manifest.study_id for manifest in manifests} != set(StudyId):
        issues.append("exactly one manifest per declared study is required")
    for manifest in manifests:
        if set(manifest.methods) != EXPECTED_METHODS[manifest.study_id]:
            issues.append(f"{manifest.study_id}: declared methods differ from release contract")
        if not manifest.not_reproduced:
            issues.append(f"{manifest.study_id}: not_reproduced must not be empty")
        if any(reference.url.scheme != "https" for reference in manifest.references):
            issues.append(f"{manifest.study_id}: source URLs must use HTTPS")

    schema = read_json(
        Path(str(files("paper_reproduction_lab").joinpath("schemas")))
        / "study-manifest.schema.json"
    )
    validator = Draft202012Validator(schema)
    for manifest in manifests:
        for error in validator.iter_errors(manifest.model_dump(mode="json")):
            issues.append(f"{manifest.study_id}: {error.message}")

    issues.extend(_unique_ids([item.document_id for item in documents], "document"))
    issues.extend(_unique_ids([item.query_id for item in queries], "query"))
    issues.extend(
        _unique_ids([item.example_id for item in calibration], "calibration example")
    )
    issues.extend(_unique_ids([item.task_id for item in tasks], "workflow task"))
    known_documents = {document.document_id for document in documents}
    for query in queries:
        if not set(query.relevant_document_ids) <= known_documents:
            issues.append(f"{query.query_id}: missing relevant document")
        if len(query.relevant_document_ids) != len(set(query.relevant_document_ids)):
            issues.append(f"{query.query_id}: duplicate relevant document")
    if {example.split for example in calibration} != {"calibration", "test"}:
        issues.append("calibration dataset requires calibration and test splits")
    if not any(task.injection_present for task in tasks):
        issues.append("workflow dataset requires prompt-injection cases")
    all_synthetic = (
        all(record.synthetic for record in documents)
        and all(record.synthetic for record in queries)
        and all(record.synthetic for record in calibration)
        and all(record.synthetic for record in tasks)
    )
    if not all_synthetic:
        issues.append("all release fixture records must be synthetic")

    records = len(documents) + len(queries) + len(calibration) + len(tasks)
    return AuditSummary(
        manifests=len(manifests),
        datasets=4,
        records=records,
        issues=issues,
        passed=not issues,
    )


def audit_release(root: Path) -> AuditSummary:
    """Audit the repository plus every tracked v0.1 release artifact."""
    repository = audit_repository(root)
    issues = list(repository.issues)
    results = root / "research" / "results" / "v0.1"
    reports = root / "reports" / "v0.1"
    required_results = [
        "trial_results.json",
        "trial_metrics.csv",
        "retrieval_predictions.jsonl",
        "calibration_predictions.jsonl",
        "workflow-agent_predictions.jsonl",
        "run_manifest.json",
        "local_findings.json",
    ]
    required_reports = [
        "RETRIEVAL_REPORT.md",
        "CALIBRATION_REPORT.md",
        "WORKFLOW_AGENT_REPORT.md",
        "FAILURE_ANALYSIS.md",
        "REPRODUCTION_SUMMARY.md",
        "LIMITATIONS.md",
        "release_summary.json",
    ]
    for directory, names in [(results, required_results), (reports, required_reports)]:
        for name in names:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(f"missing or empty release artifact: {path.relative_to(root)}")

    if issues:
        return repository.model_copy(update={"issues": issues, "passed": False})

    try:
        raw_results = read_json(results / "trial_results.json")
        if not isinstance(raw_results, list):
            raise ValueError("trial_results.json must contain an array")
        trial_results = [TrialResult.model_validate(value) for value in raw_results]
        run_manifest = RunManifest.model_validate(read_json(results / "run_manifest.json"))
        release_summary = read_json(reports / "release_summary.json")
        findings = read_json(results / "local_findings.json")
    except (DataError, ValidationError, ValueError) as exc:
        issues.append(f"invalid JSON release artifact: {exc}")
        return repository.model_copy(update={"issues": issues, "passed": False})

    expected_results = sum(len(methods) for methods in EXPECTED_METHODS.values())
    if len(trial_results) != expected_results:
        issues.append(f"expected {expected_results} trial results")
    if len({result.result_id for result in trial_results}) != len(trial_results):
        issues.append("trial result IDs must be unique")
    if {result.study_id for result in trial_results} != set(StudyId):
        issues.append("trial results must cover all three studies")
    if any(not result.synthetic or result.api_cost_usd != 0.0 for result in trial_results):
        issues.append("all trial results must be synthetic with zero API cost")
    if run_manifest.result_count != len(trial_results):
        issues.append("run manifest result_count differs from trial results")
    if set(run_manifest.studies) != set(StudyId):
        issues.append("run manifest must cover all three studies")
    if not isinstance(release_summary, dict) or release_summary.get(
        "release_gate_passed"
    ) is not True:
        issues.append("release summary gate did not pass")
    if not isinstance(findings, dict) or findings.get("synthetic") is not True:
        issues.append("local findings must be marked synthetic")

    try:
        with (results / "trial_metrics.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            csv_rows = list(csv.DictReader(handle))
        if not csv_rows:
            issues.append("trial_metrics.csv contains no data rows")
        for name in required_results:
            if not name.endswith(".jsonl"):
                continue
            path = results / name
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not rows:
                issues.append(f"{name} contains no records")
            elif any(row.get("synthetic") is not True for row in rows):
                issues.append(f"{name} contains a non-synthetic record")
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"invalid CSV/JSONL release artifact: {exc}")

    prohibited = [
        "the paper was reproduced",
        "confirmed the paper",
        "state-of-the-art",
        "production-ready",
        "guaranteed",
    ]
    for name in required_reports:
        if not name.endswith(".md"):
            continue
        text = (reports / name).read_text(encoding="utf-8").lower()
        for phrase in prohibited:
            if phrase in text:
                issues.append(f"{name} contains prohibited claim: {phrase}")

    return repository.model_copy(update={"issues": issues, "passed": not issues})
