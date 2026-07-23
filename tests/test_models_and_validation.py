from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from paper_reproduction_lab.io import canonical_json, read_json
from paper_reproduction_lab.models import AgentTask, StudyManifest, TrialResult
from paper_reproduction_lab.validation import (
    audit_release,
    audit_repository,
    load_manifests,
)


def test_manifests_match_public_draft_2020_schema(repository_root: Path) -> None:
    schema = read_json(repository_root / "schemas/study-manifest.schema.json")
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    validator = Draft202012Validator(schema)
    manifests = load_manifests(repository_root)
    assert len(manifests) == 3
    for manifest in manifests:
        validator.validate(manifest.model_dump(mode="json"))


def test_manifest_rejects_duplicate_methods(repository_root: Path) -> None:
    payload = read_json(repository_root / "papers/retrieval/manifest.json")
    payload["methods"] = ["bm25", "bm25"]
    with pytest.raises(ValidationError, match="methods must be unique"):
        StudyManifest.model_validate(payload)


def test_models_forbid_unknown_fields(repository_root: Path) -> None:
    payload = read_json(repository_root / "papers/retrieval/manifest.json")
    payload["undeclared"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudyManifest.model_validate(payload)


def test_agent_contract_rejects_disallowed_expected_tool() -> None:
    with pytest.raises(ValidationError, match="expected_tool must be in allowed_tools"):
        AgentTask(
            task_id="task-one",
            operation="lookup-document",
            allowed_tools=["validate-schema"],
            prompt="bounded fixture",
            injection_present=False,
            expected_tool="lookup-document",
            split="test",
            synthetic=True,
        )


def test_trial_result_requires_zero_api_cost() -> None:
    with pytest.raises(ValidationError):
        TrialResult(
            result_id="retrieval-bm25",
            study_id="retrieval",
            method="bm25",
            split="test",
            metrics={"mrr": 1.0},
            api_cost_usd=0.01,
            synthetic=True,
        )


def test_repository_and_release_audits_pass(repository_root: Path) -> None:
    repository = audit_repository(repository_root)
    release = audit_release(repository_root)
    assert repository.passed
    assert repository.manifests == 3
    assert repository.datasets == 4
    assert repository.records == 508
    assert release.passed


def test_audit_reports_malformed_jsonl(repository_root: Path, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    shutil.copytree(repository_root, candidate)
    query_path = candidate / "datasets/retrieval/queries.jsonl"
    query_path.write_text("{malformed\n", encoding="utf-8")
    audit = audit_repository(candidate)
    assert not audit.passed
    assert "cannot read JSONL" not in audit.issues[0]
    assert "queries.jsonl:1" in audit.issues[0]


def test_audit_reports_duplicate_ids(repository_root: Path, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    shutil.copytree(repository_root, candidate)
    query_path = candidate / "datasets/retrieval/queries.jsonl"
    first = query_path.read_text(encoding="utf-8").splitlines()[0]
    with query_path.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")
    audit = audit_repository(candidate)
    assert not audit.passed
    assert any("duplicate query IDs" in issue for issue in audit.issues)


def test_release_audit_requires_all_artifacts(repository_root: Path, tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    shutil.copytree(repository_root, candidate)
    (candidate / "reports/v0.1/release_summary.json").unlink()
    audit = audit_release(candidate)
    assert not audit.passed
    assert any("release_summary.json" in issue for issue in audit.issues)


def test_all_public_json_files_parse(repository_root: Path) -> None:
    paths = [
        *repository_root.glob("schemas/*.json"),
        *repository_root.glob("papers/*/manifest.json"),
        *repository_root.glob("research/results/v0.1/*.json"),
        *repository_root.glob("reports/v0.1/*.json"),
    ]
    assert paths
    for path in paths:
        assert json.loads(path.read_text(encoding="utf-8"))


def test_numeric_serialization_removes_last_bit_variation() -> None:
    assert canonical_json({"value": 0.32397977677114403}) == canonical_json(
        {"value": 0.3239797767711441}
    )
    with pytest.raises(ValueError, match="finite"):
        canonical_json({"value": float("nan")})
