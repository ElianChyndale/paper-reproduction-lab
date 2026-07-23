# Paper Reproduction Lab v0.1 Specification

## Purpose

Paper Reproduction Lab is a standalone, offline framework for small, auditable
reproduction studies. It separates paper claims from local hypotheses and
prevents synthetic demonstrations from being reported as replications of
original headline results.

The project is a supporting research-engineering asset. It is not a fifth
flagship project and does not import or modify existing portfolio code.

## Reproduction labels

Every study uses one of three evidence labels:

- `method-behavior-reproduction`: the implementation exercises a method's
  documented mechanics on a bounded synthetic fixture.
- `scoped-claim-check`: a pre-registered local hypothesis is evaluated on the
  supplied fixture and configuration.
- `not-reproduced`: original datasets, models, training, scale, significance
  tests, or headline results that are explicitly outside scope.

No v0.1 result may be described simply as "the paper was reproduced."

## Studies

### Retrieval

Methods:

- BM25;
- deterministic TF-IDF + TruncatedSVD latent semantic retrieval;
- Reciprocal Rank Fusion of BM25 and LSA;
- a bounded lexical/semantic feature reranker.

Metrics:

- Recall@5;
- MRR;
- binary nDCG@5;
- query-level win/loss/tie counts.

The LSA and reranker methods are offline proxies. They do not reproduce DPR or
a neural cross-encoder.

### Calibration and abstention

Methods:

- raw overconfident binary logits;
- temperature scaling selected on a calibration split;
- Platt scaling fitted on the same calibration split;
- confidence-threshold abstention.

Metrics:

- accuracy;
- negative log likelihood;
- Brier score;
- 10-bin Expected Calibration Error;
- risk at 25%, 50%, 75%, and 100% coverage;
- discrete Area Under the Risk-Coverage curve.

The study uses synthetic logits rather than a trained neural network or the
original image/document datasets.

### Workflow and constrained agent

Methods:

- deterministic workflow;
- deterministic heuristic single-agent proxy;
- deterministic constrained proposer/verifier multi-agent proxy.

Metrics:

- task accuracy;
- unsafe-action rate;
- refusal rate;
- mean decision steps;
- recorded API cost, fixed to zero.

The agent methods are state-machine proxies. They do not call an LLM and do not
reproduce ReAct, AgentBench, or any commercial/open model.

## Engineering constraints

- Python 3.11 or newer.
- Pydantic 2 and JSON Schema Draft 2020-12.
- NumPy and scikit-learn only for local numerical/ML components.
- Setuptools packaging and seed `42`.
- No API key, private data, model download, network service, or GPU.
- Deterministic source data, configurations, results, and Markdown reports.
- Machine-readable artifacts under `research/results/v0.1/`.
- Public reports under `reports/v0.1/`.

## CLI

```text
paper-repro validate
paper-repro run --study retrieval|calibration|workflow-agent|all --output research/results/v0.1
paper-repro report --results research/results/v0.1 --output reports/v0.1
paper-repro audit
```

`report` reruns validation and all studies by default before rendering reports.

## Explicit exclusions

v0.1 does not include:

- original benchmark downloads;
- large model training or inference;
- a neural dense retriever, neural reranker, or LLM agent;
- claims of matching paper tables, significance, or state of the art;
- private application data;
- live financial decisions;
- frontend or integration with existing repositories.
