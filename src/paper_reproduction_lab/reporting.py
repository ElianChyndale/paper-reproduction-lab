"""Deterministic study reports with explicit source/local scope separation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from paper_reproduction_lab.io import write_json
from paper_reproduction_lab.models import StudyId, TrialResult
from paper_reproduction_lab.runner import run_studies
from paper_reproduction_lab.validation import audit_repository, load_manifests


def _as_float(value: object) -> float:
    if not isinstance(value, int | float):
        raise TypeError("expected a numeric report value")
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, int):
        raise TypeError("expected an integer report value")
    return value


def _write_markdown(path: Path, title: str, sections: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n" + "\n\n".join(section.rstrip() for section in sections) + "\n",
        encoding="utf-8",
    )


def _method_table(results: list[TrialResult], metrics: list[str]) -> str:
    header = "| Method | " + " | ".join(metrics) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in metrics) + " |"
    rows = [header, separator]
    for result in results:
        values = [
            f"{result.metrics[metric]:.6f}" if metric in result.metrics else "?"
            for metric in metrics
        ]
        rows.append(f"| `{result.method}` | " + " | ".join(values) + " |")
    return "\n".join(rows)


def _manifest_scope(root: Path, study: StudyId) -> tuple[str, str, str]:
    manifest = next(item for item in load_manifests(root) if item.study_id == study)
    sources = "\n".join(
        f"- [{reference.title}]({reference.url}) ? {reference.claim_summary}"
        for reference in manifest.references
    )
    reproduced = "\n".join(f"- {item}" for item in manifest.reproduced_mechanics)
    excluded = "\n".join(f"- {item}" for item in manifest.not_reproduced)
    return sources, reproduced, excluded


def _retrieval_comparisons(rows: list[dict[str, object]]) -> dict[str, object]:
    by_query: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_query[str(row["query_id"])][str(row["method"])] = _as_float(
            row["reciprocal_rank"]
        )
    comparisons: dict[str, object] = {}
    for method in ["lsa", "rrf-hybrid", "feature-reranker"]:
        wins = losses = ties = 0
        for values in by_query.values():
            candidate = values[method]
            baseline = values["bm25"]
            if candidate > baseline:
                wins += 1
            elif candidate < baseline:
                losses += 1
            else:
                ties += 1
        comparisons[method] = {"wins": wins, "losses": losses, "ties": ties}
    return comparisons


def generate_reports(
    root: Path,
    results_path: Path,
    output: Path,
) -> dict[str, object]:
    audit = audit_repository(root)
    if not audit.passed:
        raise ValueError("repository audit failed: " + "; ".join(audit.issues))
    results, predictions, run_manifest = run_studies(root, results_path)
    grouped = {
        study: [result for result in results if result.study_id == study]
        for study in StudyId
    }

    retrieval_rows = predictions[StudyId.RETRIEVAL]
    comparisons = _retrieval_comparisons(retrieval_rows)
    calibration = {result.method: result for result in grouped[StudyId.CALIBRATION]}
    workflow = {result.method: result for result in grouped[StudyId.WORKFLOW_AGENT]}
    raw_ece = calibration["raw-confidence"].metrics["ece-10"]
    best_calibrated_ece = min(
        calibration["temperature-scaling"].metrics["ece-10"],
        calibration["platt-scaling"].metrics["ece-10"],
    )
    unsafe_single = workflow["heuristic-single-agent"].metrics["unsafe-action-rate"]
    unsafe_constrained = workflow["constrained-multi-agent"].metrics[
        "unsafe-action-rate"
    ]
    accuracy_single = workflow["heuristic-single-agent"].metrics["task-accuracy"]
    accuracy_constrained = workflow["constrained-multi-agent"].metrics["task-accuracy"]

    findings = {
        "schema_version": "0.1.0",
        "synthetic": True,
        "scope_statement": "local observations, not original paper results",
        "retrieval": {
            "query_comparisons_against_bm25": comparisons,
            "interpretation": (
                "Rankings and aggregate metrics are deterministic on the synthetic "
                "fixture; no DPR or paper benchmark result was tested."
            ),
        },
        "calibration": {
            "raw_ece": raw_ece,
            "best_posthoc_ece": best_calibrated_ece,
            "local_hypothesis_supported": best_calibrated_ece < raw_ece,
            "interpretation": (
                "At least one post-hoc method lowered fixture ECE; this is not a "
                "general claim about neural networks or datasets."
            ),
        },
        "workflow_agent": {
            "single_agent_unsafe_action_rate": unsafe_single,
            "constrained_unsafe_action_rate": unsafe_constrained,
            "single_agent_accuracy": accuracy_single,
            "constrained_accuracy": accuracy_constrained,
            "local_hypothesis_supported": (
                unsafe_constrained <= unsafe_single
                and accuracy_constrained >= accuracy_single
            ),
            "interpretation": (
                "The deterministic verifier controlled a heuristic proxy; no LLM "
                "agent or external action was evaluated."
            ),
        },
    }
    write_json(results_path / "local_findings.json", findings)

    retrieval_sources, retrieval_reproduced, retrieval_excluded = _manifest_scope(
        root, StudyId.RETRIEVAL
    )
    _write_markdown(
        output / "RETRIEVAL_REPORT.md",
        "Retrieval Study",
        [
            "## Source context\n\n" + retrieval_sources,
            (
                "## Local hypothesis\n\nThe declared methods should produce "
                "deterministic rankings, and RRF should combine sparse and "
                "latent-semantic rankings on this fixture."
            ),
            "## Reproduced mechanics\n\n" + retrieval_reproduced,
            "## Not reproduced\n\n" + retrieval_excluded,
            "## Synthetic test results\n\n"
            + _method_table(
                grouped[StudyId.RETRIEVAL],
                ["recall-at-5", "mrr", "ndcg-at-5"],
            ),
            "## Query-level comparisons against BM25\n\n```json\n"
            + json.dumps(comparisons, indent=2, sort_keys=True)
            + "\n```",
            (
                "## Interpretation\n\nThese are local method-behavior observations. "
                "LSA is not DPR, the feature reranker is not a neural cross-encoder, "
                "and the original benchmark values were not tested."
            ),
            (
                "## Rerun\n\n```text\npaper-repro run --study retrieval "
                "--output research/results/v0.1\n```"
            ),
        ],
    )

    calibration_sources, calibration_reproduced, calibration_excluded = (
        _manifest_scope(root, StudyId.CALIBRATION)
    )
    _write_markdown(
        output / "CALIBRATION_REPORT.md",
        "Calibration and Abstention Study",
        [
            "## Source context\n\n" + calibration_sources,
            (
                "## Local hypothesis\n\nAt least one post-hoc calibrator should "
                "lower test ECE relative to the raw synthetic logits."
            ),
            "## Reproduced mechanics\n\n" + calibration_reproduced,
            "## Not reproduced\n\n" + calibration_excluded,
            "## Synthetic test results\n\n"
            + _method_table(
                grouped[StudyId.CALIBRATION],
                [
                    "accuracy",
                    "negative-log-likelihood",
                    "brier-score",
                    "ece-10",
                    "aurc",
                ],
            ),
            (
                "## Scoped finding\n\n"
                f"Raw ECE was `{raw_ece:.6f}` and the better fitted post-hoc ECE "
                f"was `{best_calibrated_ece:.6f}`. The local hypothesis was "
                f"{'supported' if best_calibrated_ece < raw_ece else 'not supported'} "
                "on this fixture only."
            ),
            (
                "## Rerun\n\n```text\npaper-repro run --study calibration "
                "--output research/results/v0.1\n```"
            ),
        ],
    )

    workflow_sources, workflow_reproduced, workflow_excluded = _manifest_scope(
        root, StudyId.WORKFLOW_AGENT
    )
    _write_markdown(
        output / "WORKFLOW_AGENT_REPORT.md",
        "Workflow and Agent-Proxy Study",
        [
            "## Source context\n\n" + workflow_sources,
            (
                "## Local hypothesis\n\nAn allowlisted proposer/verifier proxy "
                "should select fewer unsafe actions than an unconstrained heuristic "
                "proxy on fixed prompt-injection tasks."
            ),
            "## Reproduced mechanics\n\n" + workflow_reproduced,
            "## Not reproduced\n\n" + workflow_excluded,
            "## Synthetic test results\n\n"
            + _method_table(
                grouped[StudyId.WORKFLOW_AGENT],
                [
                    "task-accuracy",
                    "unsafe-action-rate",
                    "refusal-rate",
                    "mean-steps",
                    "api-cost-usd",
                ],
            ),
            (
                "## Scoped finding\n\nThe heuristic proxy unsafe-action rate was "
                f"`{unsafe_single:.6f}`; the constrained proxy rate was "
                f"`{unsafe_constrained:.6f}`. No tool was executed and API cost was "
                "zero by construction."
            ),
            (
                "## Rerun\n\n```text\npaper-repro run --study workflow-agent "
                "--output research/results/v0.1\n```"
            ),
        ],
    )

    retrieval_failures = [
        row
        for row in retrieval_rows
        if _as_float(row["reciprocal_rank"]) < 1.0
    ][:5]
    calibration_rows = predictions[StudyId.CALIBRATION]
    calibration_failures = [
        row
        for row in calibration_rows
        if (
            (_as_float(row["raw_probability"]) >= 0.8 and _as_int(row["label"]) == 0)
            or (_as_float(row["raw_probability"]) <= 0.2 and _as_int(row["label"]) == 1)
        )
    ][:5]
    workflow_failures = [
        row
        for row in predictions[StudyId.WORKFLOW_AGENT]
        if bool(row["unsafe"])
    ][:5]
    _write_markdown(
        output / "FAILURE_ANALYSIS.md",
        "Failure Analysis",
        [
            (
                "Failures below are synthetic record-level examples. They support "
                "debugging, not estimates of real-world prevalence."
            ),
            "## Retrieval misses or lower first-relevant ranks\n\n```json\n"
            + json.dumps(retrieval_failures, indent=2, sort_keys=True)
            + "\n```",
            "## High-confidence raw calibration errors\n\n```json\n"
            + json.dumps(calibration_failures, indent=2, sort_keys=True)
            + "\n```",
            "## Unsafe heuristic proposals\n\n```json\n"
            + json.dumps(workflow_failures, indent=2, sort_keys=True)
            + "\n```",
        ],
    )

    manifests = load_manifests(root)
    summary_table = "\n".join(
        [
            "| Study | Evidence labels | Methods |",
            "| --- | --- | ---: |",
            *[
                (
                    f"| `{manifest.study_id.value}` | "
                    f"{', '.join(label.value for label in manifest.evidence_labels)} | "
                    f"{len(manifest.methods)} |"
                )
                for manifest in manifests
            ],
        ]
    )
    _write_markdown(
        output / "REPRODUCTION_SUMMARY.md",
        "Paper Reproduction Lab v0.1 Summary",
        [
            (
                "This release contains scoped synthetic observations. It does not "
                "claim reproduction of original headline numbers, models, datasets, "
                "or statistical conclusions."
            ),
            "## Study registry\n\n" + summary_table,
            (
                "## Release counts\n\n"
                f"- Study manifests: {audit.manifests}\n"
                f"- Fixture records: {audit.records}\n"
                f"- Method results: {len(results)}\n"
                f"- Seed: {run_manifest.seed}"
            ),
            (
                "## Full rerun\n\n```text\n"
                "paper-repro report --results research/results/v0.1 "
                "--output reports/v0.1\n```"
            ),
        ],
    )

    limitations = root / "LIMITATIONS.md"
    (output / "LIMITATIONS.md").write_text(
        limitations.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    release_gate = (
        audit.passed
        and len(results) == 10
        and best_calibrated_ece < raw_ece
        and unsafe_constrained <= unsafe_single
        and accuracy_constrained >= accuracy_single
    )
    release = {
        "schema_version": "0.1.0",
        "synthetic": True,
        "studies": 3,
        "method_results": len(results),
        "fixture_records": audit.records,
        "local_findings": findings,
        "release_gate_passed": release_gate,
        "claim_boundary": "not-original-paper-results",
    }
    write_json(output / "release_summary.json", release)
    return release
