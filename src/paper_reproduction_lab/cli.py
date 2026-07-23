"""Offline CLI for validation, study execution, reporting, and audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paper_reproduction_lab.models import StudyId
from paper_reproduction_lab.reporting import generate_reports
from paper_reproduction_lab.runner import run_studies
from paper_reproduction_lab.validation import audit_release, audit_repository


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="paper-repro",
        description="Run honest, deterministic scoped reproduction studies.",
    )
    subparsers = root.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path("."))

    run = subparsers.add_parser("run")
    run.add_argument(
        "--study",
        choices=["retrieval", "calibration", "workflow-agent", "all"],
        default="all",
    )
    run.add_argument("--root", type=Path, default=Path("."))
    run.add_argument("--output", type=Path, default=Path("research/results/v0.1"))

    report = subparsers.add_parser("report")
    report.add_argument("--root", type=Path, default=Path("."))
    report.add_argument("--results", type=Path, default=Path("research/results/v0.1"))
    report.add_argument("--output", type=Path, default=Path("reports/v0.1"))

    audit = subparsers.add_parser("audit")
    audit.add_argument("--root", type=Path, default=Path("."))
    return root


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def run(arguments: list[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    try:
        if args.command == "validate":
            audit = audit_repository(args.root)
            _print(audit.model_dump(mode="json"))
            return 0 if audit.passed else 1
        if args.command == "audit":
            audit = audit_release(args.root)
            _print(audit.model_dump(mode="json"))
            return 0 if audit.passed else 1
        if args.command == "run":
            studies = None if args.study == "all" else [StudyId(args.study)]
            results, predictions, manifest = run_studies(
                args.root,
                args.output,
                studies,
            )
            _print(
                {
                    "studies": [study.value for study in manifest.studies],
                    "method_results": len(results),
                    "prediction_rows": sum(map(len, predictions.values())),
                    "output": args.output.as_posix(),
                    "synthetic": True,
                }
            )
            return 0
        if args.command == "report":
            release = generate_reports(args.root, args.results, args.output)
            _print(release)
            return 0 if bool(release["release_gate_passed"]) else 1
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
