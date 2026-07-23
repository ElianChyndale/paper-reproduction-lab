"""Deterministic sparse, latent-semantic, hybrid, and reranking study."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from paper_reproduction_lab.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from paper_reproduction_lab.models import RetrievalDocument, RetrievalQuery, TrialResult

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def bm25_scores(
    documents: Sequence[RetrievalDocument],
    query: str,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> NDArray[np.float64]:
    if not documents:
        raise ValueError("documents must not be empty")
    if k1 <= 0.0 or not 0.0 <= b <= 1.0:
        raise ValueError("invalid BM25 parameters")
    tokenized = [tokenize(document.text) for document in documents]
    average_length = sum(map(len, tokenized)) / len(tokenized)
    if average_length == 0.0:
        raise ValueError("documents must contain tokens")
    frequencies = [Counter(tokens) for tokens in tokenized]
    query_terms = list(dict.fromkeys(tokenize(query)))
    scores = np.zeros(len(documents), dtype=np.float64)
    for term in query_terms:
        document_frequency = sum(term in counts for counts in frequencies)
        inverse_document_frequency = math.log(
            1.0 + (len(documents) - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for index, counts in enumerate(frequencies):
            frequency = counts[term]
            if frequency == 0:
                continue
            length_normalization = 1.0 - b + b * len(tokenized[index]) / average_length
            scores[index] += inverse_document_frequency * (
                frequency * (k1 + 1.0) / (frequency + k1 * length_normalization)
            )
    return scores


def _rank(
    documents: Sequence[RetrievalDocument],
    scores: NDArray[np.float64],
) -> list[str]:
    if len(documents) != len(scores):
        raise ValueError("document and score lengths differ")
    ordered = sorted(
        zip(documents, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].document_id),
    )
    return [document.document_id for document, _ in ordered]


def lsa_score_matrix(
    documents: Sequence[RetrievalDocument],
    queries: Sequence[RetrievalQuery],
    seed: int = 42,
) -> NDArray[np.float64]:
    if not documents or not queries:
        raise ValueError("documents and queries must not be empty")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    document_matrix = vectorizer.fit_transform(document.text for document in documents)
    maximum_components = min(
        16,
        document_matrix.shape[0] - 1,
        document_matrix.shape[1] - 1,
    )
    if maximum_components < 1:
        raise ValueError("corpus is too small for latent semantic retrieval")
    decomposition = TruncatedSVD(n_components=maximum_components, random_state=seed)
    document_vectors = normalize(decomposition.fit_transform(document_matrix))
    query_matrix = vectorizer.transform(query.text for query in queries)
    query_vectors = normalize(decomposition.transform(query_matrix))
    return np.asarray(query_vectors @ document_vectors.T, dtype=np.float64)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    constant: int = 60,
) -> list[str]:
    if not rankings:
        raise ValueError("at least one ranking is required")
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError("rankings must not contain duplicate IDs")
        for rank, identifier in enumerate(ranking, start=1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (constant + rank)
    return sorted(scores, key=lambda identifier: (-scores[identifier], identifier))


def _normalize_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def rerank_scores(
    documents: Sequence[RetrievalDocument],
    query: RetrievalQuery,
    bm25: NDArray[np.float64],
    lsa: NDArray[np.float64],
) -> NDArray[np.float64]:
    query_terms = set(tokenize(query.text))
    overlap = np.asarray(
        [
            len(query_terms & set(tokenize(document.title + " " + document.text)))
            / max(1, len(query_terms))
            for document in documents
        ],
        dtype=np.float64,
    )
    return 0.4 * _normalize_scores(bm25) + 0.5 * _normalize_scores(lsa) + 0.1 * overlap


def run_retrieval(
    documents: list[RetrievalDocument],
    queries: list[RetrievalQuery],
) -> tuple[list[TrialResult], list[dict[str, object]]]:
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document IDs must be unique")
    known = set(document_ids)
    test_queries = [query for query in queries if query.split == "test"]
    if not test_queries:
        raise ValueError("retrieval study requires test queries")
    for query in queries:
        if not set(query.relevant_document_ids) <= known:
            raise ValueError(f"{query.query_id} references a missing document")

    lsa_matrix = lsa_score_matrix(documents, test_queries)
    method_metrics: dict[str, list[tuple[float, float, float]]] = {
        method: [] for method in ["bm25", "lsa", "rrf-hybrid", "feature-reranker"]
    }
    predictions: list[dict[str, object]] = []
    for query_index, query in enumerate(test_queries):
        bm25 = bm25_scores(documents, query.text)
        lsa = lsa_matrix[query_index]
        rankings = {
            "bm25": _rank(documents, bm25),
            "lsa": _rank(documents, lsa),
        }
        rankings["rrf-hybrid"] = reciprocal_rank_fusion(
            [rankings["bm25"], rankings["lsa"]]
        )
        rankings["feature-reranker"] = _rank(
            documents,
            rerank_scores(documents, query, bm25, lsa),
        )
        relevant = set(query.relevant_document_ids)
        for method, ranking in rankings.items():
            metric_values = (
                recall_at_k(ranking, relevant, 5),
                reciprocal_rank(ranking, relevant),
                ndcg_at_k(ranking, relevant, 5),
            )
            method_metrics[method].append(metric_values)
            predictions.append(
                {
                    "study_id": "retrieval",
                    "query_id": query.query_id,
                    "method": method,
                    "ranking": ranking,
                    "relevant_document_ids": sorted(relevant),
                    "recall_at_5": metric_values[0],
                    "reciprocal_rank": metric_values[1],
                    "ndcg_at_5": metric_values[2],
                    "synthetic": True,
                }
            )

    results = []
    for method, samples in method_metrics.items():
        matrix = np.asarray(samples, dtype=np.float64)
        results.append(
            TrialResult(
                result_id=f"retrieval-{method}",
                study_id="retrieval",
                method=method,
                split="test",
                metrics={
                    "recall-at-5": float(np.mean(matrix[:, 0])),
                    "mrr": float(np.mean(matrix[:, 1])),
                    "ndcg-at-5": float(np.mean(matrix[:, 2])),
                    "queries": float(len(samples)),
                },
                synthetic=True,
            )
        )
    return results, predictions
