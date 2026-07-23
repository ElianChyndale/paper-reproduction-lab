# Retrieval Claim Register

## Source context

DPR evaluates a trained dual-encoder retriever against sparse retrieval on
open-domain QA datasets. RRF evaluates a simple reciprocal-rank combination
method on retrieval rankings.

## Local hypothesis

All four offline methods produce deterministic rankings; RRF combines BM25 and
LSA rankings on the committed synthetic corpus.

## Decision rule

The study passes when all declared methods emit complete rankings and valid
Recall@5, MRR, and nDCG@5 values. Improvement over BM25 is recorded but is not
required and is not generalized.
