"""Parse every required machine-readable v0.1 artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "results" / "v0.1"


def main() -> int:
    required = [
        RESULTS / "trial_results.json",
        RESULTS / "trial_metrics.csv",
        RESULTS / "retrieval_predictions.jsonl",
        RESULTS / "calibration_predictions.jsonl",
        RESULTS / "workflow-agent_predictions.jsonl",
        RESULTS / "run_manifest.json",
        RESULTS / "local_findings.json",
    ]
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty artifact: {path.relative_to(ROOT)}")
        if path.suffix == ".json":
            if not json.loads(path.read_text(encoding="utf-8")):
                raise ValueError(f"empty JSON value: {path.relative_to(ROOT)}")
        elif path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not rows:
                raise ValueError(f"empty JSONL: {path.relative_to(ROOT)}")
        else:
            with path.open(encoding="utf-8", newline="") as handle:
                if not list(csv.DictReader(handle)):
                    raise ValueError(f"empty CSV: {path.relative_to(ROOT)}")
    print(f"parsed {len(required)} non-empty machine-readable artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
