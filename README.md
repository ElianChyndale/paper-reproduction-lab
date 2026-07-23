# Paper Reproduction Lab

Paper Reproduction Lab is an offline framework for small, reviewable
reproduction studies. Version 0.1 runs three deterministic studies:

1. BM25, latent-semantic retrieval, reciprocal-rank fusion, and bounded
   reranking.
2. Raw confidence, temperature scaling, Platt scaling, and selective
   prediction.
3. A deterministic workflow, heuristic single-agent proxy, and constrained
   proposer/verifier proxy.

The release does **not** claim reproduction of original paper models,
benchmarks, tables, statistical significance, or headline conclusions. It
reproduces selected mechanics and evaluates pre-registered local hypotheses on
synthetic data.

## Quick start

```bash
python -m pip install -e ".[dev]"
paper-repro validate
paper-repro run --study all --output research/results/v0.1
paper-repro report --results research/results/v0.1 --output reports/v0.1
paper-repro audit
```

No API key, model download, private dataset, blockchain node, GPU, or network
service is required after installation.

## Evidence boundaries

Every manifest separates:

- source-paper claim context;
- local hypothesis;
- reproduced mechanics;
- non-reproduced components;
- exact rerun command.

The three allowed evidence labels are documented in
[SPEC.md](SPEC.md) and [REPRODUCTION_PROTOCOL.md](REPRODUCTION_PROTOCOL.md).
Primary sources and scoped interpretations are listed in
[PAPER_SOURCES.md](PAPER_SOURCES.md).

## CLI

```text
paper-repro validate
paper-repro run --study retrieval|calibration|workflow-agent|all --output DIRECTORY
paper-repro report --results DIRECTORY --output DIRECTORY
paper-repro audit
```

`report` reruns validation and all three studies before rendering outputs.
Subset runs are useful for development but do not satisfy the release gate.

## Repository layout

```text
paper-reproduction-lab/
??? templates/
??? papers/
?   ??? retrieval/
?   ??? calibration/
?   ??? workflow-agent/
??? datasets/
??? schemas/
??? src/paper_reproduction_lab/
??? tests/
??? research/results/v0.1/
??? reports/v0.1/
```

## Current synthetic observations

The generated release report records:

- retrieval rankings and Recall@5/MRR/nDCG@5 for four local methods;
- calibration accuracy, NLL, Brier, ECE, and risk-coverage metrics;
- workflow/agent-proxy accuracy, unsafe selections, refusals, steps, and zero
  API cost.

These observations are properties of the committed fixtures and code. They are
not external research findings.

## Reproducibility

```bash
python scripts/generate_schemas.py --check
python scripts/generate_fixtures.py --check
python -m pytest
python -m ruff check src tests scripts
python -m mypy src/paper_reproduction_lab
paper-repro report --results research/results/v0.1 --output reports/v0.1
python scripts/check_artifacts.py
git diff --exit-code
```

Read [LIMITATIONS.md](LIMITATIONS.md) and
[DATASET_CARD.md](DATASET_CARD.md) before using or citing results.

## Portfolio scope

This repository is a supporting research-engineering asset. It imports no code
from EcoQuant, Auralynq, Green Bond Lending, AI Research Engineering Lab, or
other supporting repositories, and it does not change the four-project
flagship structure.

## License

MIT. See [LICENSE](LICENSE).
