from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from paper_reproduction_lab.io import read_jsonl
from paper_reproduction_lab.models import RetrievalDocument, RetrievalQuery
from paper_reproduction_lab.retrieval import (
    bm25_scores,
    lsa_score_matrix,
    reciprocal_rank_fusion,
    run_retrieval,
    tokenize,
)


def _document(identifier: str, text: str) -> RetrievalDocument:
    return RetrievalDocument(
        document_id=identifier,
        title=identifier,
        text=text,
        topic="test-topic",
        synthetic=True,
    )


def test_tokenize_and_bm25_rank_exact_term_first() -> None:
    documents = [
        _document("alpha", "duration measures rate sensitivity"),
        _document("beta", "calibration measures confidence"),
    ]
    assert tokenize("Bond-DURATION, 2026!") == ["bond", "duration", "2026"]
    scores = bm25_scores(documents, "duration")
    assert scores[0] > scores[1]


@pytest.mark.parametrize(
    ("documents", "kwargs", "message"),
    [
        ([], {}, "must not be empty"),
        ([_document("empty", "!!!")], {}, "contain tokens"),
        ([_document("valid", "token")], {"k1": 0.0}, "invalid BM25"),
        ([_document("valid", "token")], {"b": 2.0}, "invalid BM25"),
    ],
)
def test_bm25_rejects_invalid_inputs(
    documents: list[RetrievalDocument],
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bm25_scores(documents, "query", **kwargs)


def test_rrf_has_deterministic_manual_order() -> None:
    ranking = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]], constant=1)
    assert ranking == ["a", "b", "c"]


def test_rrf_rejects_duplicate_and_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        reciprocal_rank_fusion([])
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([["a", "a"]])
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], constant=0)


def test_lsa_and_full_runner_are_deterministic(repository_root: Path) -> None:
    documents = read_jsonl(
        repository_root / "datasets/retrieval/documents.jsonl", RetrievalDocument
    )
    queries = read_jsonl(
        repository_root / "datasets/retrieval/queries.jsonl", RetrievalQuery
    )
    first_matrix = lsa_score_matrix(documents, queries[:2])
    second_matrix = lsa_score_matrix(documents, queries[:2])
    np.testing.assert_allclose(first_matrix, second_matrix, atol=0.0, rtol=0.0)
    first_results, first_predictions = run_retrieval(documents, queries)
    second_results, second_predictions = run_retrieval(documents, queries)
    assert [item.model_dump() for item in first_results] == [
        item.model_dump() for item in second_results
    ]
    assert first_predictions == second_predictions
    assert len(first_results) == 4
    assert len(first_predictions) == 72


def test_runner_rejects_missing_relevant_document(repository_root: Path) -> None:
    documents = read_jsonl(
        repository_root / "datasets/retrieval/documents.jsonl", RetrievalDocument
    )
    query = RetrievalQuery(
        query_id="missing-query",
        text="missing",
        relevant_document_ids=["missing-document"],
        split="test",
        synthetic=True,
    )
    with pytest.raises(ValueError, match="missing document"):
        run_retrieval(documents, [query])
