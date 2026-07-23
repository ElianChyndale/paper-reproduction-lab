"""Pydantic contracts for studies, fixtures, and results."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints, model_validator

StableId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StudyId(StrEnum):
    RETRIEVAL = "retrieval"
    CALIBRATION = "calibration"
    WORKFLOW_AGENT = "workflow-agent"


class EvidenceLabel(StrEnum):
    METHOD_BEHAVIOR = "method-behavior-reproduction"
    SCOPED_CLAIM = "scoped-claim-check"
    NOT_REPRODUCED = "not-reproduced"
    METHODOLOGICAL_REFERENCE = "methodological-reference"


class PaperReference(StrictModel):
    reference_id: StableId
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    year: int = Field(ge=1900, le=2100)
    url: HttpUrl
    claim_summary: str = Field(min_length=1)


class StudyManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    study_id: StudyId
    title: str = Field(min_length=1)
    evidence_labels: list[EvidenceLabel] = Field(min_length=1)
    references: list[PaperReference] = Field(min_length=1)
    local_hypothesis: str = Field(min_length=1)
    reproduced_mechanics: list[str] = Field(min_length=1)
    not_reproduced: list[str] = Field(min_length=1)
    dataset_id: StableId
    methods: list[StableId] = Field(min_length=2)
    metrics: list[StableId] = Field(min_length=1)
    seed: Literal[42] = 42
    rerun_command: str = Field(min_length=1)
    synthetic: Literal[True]

    @model_validator(mode="after")
    def unique_lists(self) -> StudyManifest:
        for name, values in [
            ("methods", self.methods),
            ("metrics", self.metrics),
            ("reference IDs", [reference.reference_id for reference in self.references]),
        ]:
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class RetrievalDocument(StrictModel):
    document_id: StableId
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    topic: StableId
    synthetic: Literal[True]


class RetrievalQuery(StrictModel):
    query_id: StableId
    text: str = Field(min_length=1)
    relevant_document_ids: list[StableId] = Field(min_length=1)
    split: Literal["dev", "test"]
    synthetic: Literal[True]


class CalibrationExample(StrictModel):
    example_id: StableId
    logit: float = Field(allow_inf_nan=False)
    label: Literal[0, 1]
    split: Literal["calibration", "test"]
    synthetic: Literal[True]


class AgentTask(StrictModel):
    task_id: StableId
    operation: StableId
    allowed_tools: list[StableId] = Field(min_length=1)
    prompt: str = Field(min_length=1)
    injection_present: bool
    expected_tool: StableId
    split: Literal["test"]
    synthetic: Literal[True]

    @model_validator(mode="after")
    def expected_is_allowed(self) -> AgentTask:
        if self.expected_tool not in self.allowed_tools:
            raise ValueError("expected_tool must be in allowed_tools")
        if self.operation != self.expected_tool:
            raise ValueError("operation must equal expected_tool")
        return self


class TrialResult(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    result_id: StableId
    study_id: StudyId
    method: StableId
    split: Literal["test"]
    metrics: dict[StableId, float] = Field(min_length=1)
    api_cost_usd: float = Field(default=0.0, ge=0.0, le=0.0)
    seed: Literal[42] = 42
    synthetic: Literal[True] = True


class RunManifest(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    run_id: Literal["v0-1-seed-42"] = "v0-1-seed-42"
    studies: list[StudyId] = Field(min_length=1)
    result_count: int = Field(ge=1)
    dataset_hashes: dict[str, str] = Field(min_length=1)
    config_hashes: dict[str, str] = Field(min_length=1)
    release_claim: Literal[
        "synthetic-scoped-observations-not-original-paper-results"
    ] = "synthetic-scoped-observations-not-original-paper-results"
    seed: Literal[42] = 42


class AuditSummary(StrictModel):
    schema_version: Literal["0.1.0"] = "0.1.0"
    manifests: int = Field(ge=0)
    datasets: int = Field(ge=0)
    records: int = Field(ge=0)
    issues: list[str] = Field(default_factory=list)
    passed: bool
