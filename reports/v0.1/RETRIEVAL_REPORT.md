# Retrieval Study

## Source context

- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) — The paper evaluates learned dual-encoder dense passage retrieval against strong sparse retrieval on open-domain QA datasets.
- [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — The paper reports effective rank combination with reciprocal rank fusion in its retrieval experiments.

## Local hypothesis

The declared methods should produce deterministic rankings, and RRF should combine sparse and latent-semantic rankings on this fixture.

## Reproduced mechanics

- BM25 scoring
- TF-IDF plus TruncatedSVD latent-semantic ranking
- RRF with constant 60
- bounded feature reranking
- Recall@5, MRR, and binary nDCG@5

## Not reproduced

- DPR model training or neural embeddings
- neural cross-encoder reranking
- Natural Questions, TriviaQA, WebQuestions, CuratedTREC, TREC, or LETOR
- original paper values and statistical significance

## Synthetic test results

| Method | recall-at-5 | mrr | ndcg-at-5 |
| --- | ---: | ---: | ---: |
| `bm25` | 0.796296 | 0.944444 | 0.784881 |
| `lsa` | 0.981481 | 0.972222 | 0.969172 |
| `rrf-hybrid` | 0.870370 | 0.972222 | 0.852128 |
| `feature-reranker` | 0.962963 | 0.972222 | 0.942080 |

## Query-level comparisons against BM25

```json
{
  "feature-reranker": {
    "losses": 0,
    "ties": 17,
    "wins": 1
  },
  "lsa": {
    "losses": 0,
    "ties": 17,
    "wins": 1
  },
  "rrf-hybrid": {
    "losses": 0,
    "ties": 17,
    "wins": 1
  }
}
```

## Interpretation

These are local method-behavior observations. LSA is not DPR, the feature reranker is not a neural cross-encoder, and the original benchmark values were not tested.

## Rerun

```text
paper-repro run --study retrieval --output research/results/v0.1
```
