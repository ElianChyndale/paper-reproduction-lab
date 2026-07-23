# Synthetic Dataset Card

## Summary

All v0.1 records are generated locally with seed `42` and contain
`synthetic: true`. They are designed to exercise code paths and evaluation
contracts, not to represent a population, market, organization, or production
environment.

## Retrieval dataset

- 48 short synthetic documents.
- 24 queries: 6 development and 18 test.
- 12 technical topics.
- Three relevant documents and one lexical distractor per topic.
- Relevance is complete only by construction for this bounded corpus.

The data is not Natural Questions, TriviaQA, WebQuestions, CuratedTREC, TREC,
LETOR, or another paper benchmark.

## Calibration dataset

- 400 binary examples.
- 200 calibration and 200 test examples.
- Labels are sampled from a seeded logistic data-generating process.
- Logged model scores are deliberately scaled and shifted to create
  overconfidence.

The data is not produced by a neural network and does not represent CIFAR,
ImageNet, medical, credit, or other operational predictions.

## Workflow-agent dataset

- 36 held-out routing tasks.
- Four side-effect-free tool names.
- 12 tasks contain explicit untrusted prompt-injection text.
- Three unsafe tool names appear only as proposals to be rejected.

No tool call is executed. The prompts are not sampled from users or agent
benchmarks.

## Intended use

- deterministic unit and integration testing;
- learning reproduction discipline;
- inspecting metric and report behavior;
- demonstrating claim/scope separation.

## Out-of-scope use

- model selection for production;
- estimates of retrieval, calibration, or agent performance;
- safety certification;
- financial, legal, or academic conclusions;
- demographic or behavioral analysis.
