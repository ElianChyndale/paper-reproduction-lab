# Acceptance Criteria

## Claims and data

- [x] Exactly three study manifests validate.
- [x] Every source claim has a primary-source URL and an explicit local scope.
- [x] Every study lists original elements that were not reproduced.
- [x] All fixture records are synthetic, uniquely identified, and split-valid.
- [x] No report describes local synthetic observations as original paper
      results.

## Retrieval study

- [x] BM25, LSA, RRF hybrid, and reranking execute deterministically.
- [x] Recall@5, MRR, nDCG@5, and query-level comparisons are generated.
- [x] Duplicate retrievals and missing relevant-document IDs are rejected.
- [x] Metric implementations match manually calculable tests.
- [x] LSA is labelled as a proxy rather than DPR.

## Calibration and abstention study

- [x] Raw, temperature-scaled, and Platt-scaled probabilities are evaluated.
- [x] Calibration parameters fit only the calibration split.
- [x] Accuracy, NLL, Brier, ECE, risk-coverage points, and AURC are generated.
- [x] Empty bins, zero coverage, extreme logits, and invalid probabilities are
      tested.
- [x] Local findings do not generalize beyond the fixture.

## Workflow and agent study

- [x] Deterministic workflow, heuristic single-agent proxy, and constrained
      multi-agent proxy all execute.
- [x] Prompt-injection tasks are present.
- [x] Task accuracy, unsafe-action rate, refusal rate, steps, and zero API cost
      are recorded.
- [x] No policy executes external tools or mutates user data.
- [x] Agent proxies are not presented as real LLM experiments.

## Engineering and release

- [x] `paper-repro validate`, `run`, `report`, and `audit` pass.
- [x] All CSV, JSON, and JSONL artifacts parse and contain records.
- [x] Repeated fixture/result/report generation has identical hashes.
- [x] Public and packaged Draft 2020-12 schemas match.
- [x] Clean wheel installation loads all packaged schemas.
- [x] Pytest, Ruff, and strict mypy pass.
- [ ] CI passes on Ubuntu/Windows with Python 3.11/3.12.
- [ ] Public repository and `v0.1.0` tag point to the final CI-validated commit.
- [ ] The portfolio task log records commands, CI, URL, and remaining risks.

## Release commands

```text
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests scripts
python -m mypy src/paper_reproduction_lab
paper-repro validate
paper-repro run --study all --output research/results/v0.1
paper-repro report --results research/results/v0.1 --output reports/v0.1
paper-repro audit
python scripts/check_artifacts.py
```
