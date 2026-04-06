"""Hybrid retriever — BM25 + Vector fusion via Reciprocal Rank Fusion.

Retrieval strategy:
1. **BM25Retriever** – keyword search over the full corpus (term frequency,
   inverse document frequency, document-length normalisation).
   The BM25 index is cached and only rebuilt when the corpus changes.
2. **Vector retriever** – semantic similarity via LanceDB embeddings.
3. **Reciprocal Rank Fusion** – merges both ranked lists with configurable
   weights so documents that appear in both lists are boosted.
4. **Diversity filter** – limits chunks per document label so the LLM sees
   a broader slice of the knowledge base.
"""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from rag.config import (
    BM25_TOP_K,
    BM25_WEIGHT,
    FINAL_CONTEXT_K,
    MAX_CHUNKS_PER_LABEL,
    RERANKER_TOP_K,
    VECTOR_TOP_K,
    VECTOR_WEIGHT,
)
from rag.reranker import rerank
from rag.vectorstore import load_all_documents, similarity_search


# ---------------------------------------------------------------------------
# BM25 cache — avoids rebuilding the index on every single query
# ---------------------------------------------------------------------------

class _BM25Cache:
    """Lazily-built, invalidation-aware BM25 index cache."""

    def __init__(self) -> None:
        self._hash: str = ""
        self._retriever: BM25Retriever | None = None

    @staticmethod
    def _corpus_hash(corpus: list[Document]) -> str:
        h = hashlib.md5(usedforsecurity=False)
        for d in corpus:
            h.update(d.page_content.encode("utf-8", errors="replace"))
        return h.hexdigest()

    def get(self, corpus: list[Document], k: int = BM25_TOP_K) -> BM25Retriever:
        corpus_hash = self._corpus_hash(corpus)
        if self._retriever is None or corpus_hash != self._hash:
            self._retriever = BM25Retriever.from_documents(corpus, k=k)
            self._hash = corpus_hash
        else:
            self._retriever.k = k  # update k in case config changed
        return self._retriever

    def invalidate(self) -> None:
        self._hash = ""
        self._retriever = None


_bm25_cache = _BM25Cache()


# ---------------------------------------------------------------------------
# Vector retriever adapter
# ---------------------------------------------------------------------------

def _vector_retrieve(query: str, top_k: int = VECTOR_TOP_K) -> list[Document]:
    """Run vector similarity search and return LangChain Documents."""
    raw = similarity_search(query, top_k=top_k)
    docs: list[Document] = []
    for r in raw:
        text = r.get("text", "")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "keywords": r.get("keywords", ""),
                    "label": r.get("label", "N/A"),
                    "_distance": r.get("_distance"),
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    weights: list[float],
    source_names: list[str] | None = None,
    k: int = 60,
) -> list[Document]:
    """Merge multiple ranked lists using weighted RRF.

    For each document at position *rank* in a list with *weight*:
        score += weight / (k + rank + 1)
    Scores are summed across lists. Higher = more relevant.

    When *source_names* is provided, per-source contribution scores are
    stored in metadata as ``{name}_rrf`` (e.g. ``bm25_rrf``, ``vector_rrf``).
    """
    if source_names is None:
        source_names = [f"source_{i}" for i in range(len(result_lists))]

    scores: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    doc_map: dict[str, Document] = {}

    for idx, (result_list, weight) in enumerate(zip(result_lists, weights)):
        src = source_names[idx]
        for rank, doc in enumerate(result_list):
            key = doc.page_content
            if key not in doc_map:
                doc_map[key] = doc
                contributions[key] = {}
            contrib = weight / (k + rank + 1)
            scores[key] = scores.get(key, 0.0) + contrib
            contributions[key][src] = round(contrib, 6)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

    result: list[Document] = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = round(scores[key], 6)
        for src, val in contributions[key].items():
            doc.metadata[f"{src}_rrf"] = val
        result.append(doc)
    return result


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------

def _diversify(docs: list[Document], k: int = FINAL_CONTEXT_K) -> list[Document]:
    """Limit to *k* results with max ``MAX_CHUNKS_PER_LABEL`` per label."""
    selected: list[Document] = []
    label_counts: dict[str, int] = {}

    for doc in docs:
        lbl = (doc.metadata.get("label") or "N/A").lower()
        cnt = label_counts.get(lbl, 0)
        if cnt >= MAX_CHUNKS_PER_LABEL:
            continue
        selected.append(doc)
        label_counts[lbl] = cnt + 1
        if len(selected) >= k:
            break

    if len(selected) < k:
        for doc in docs:
            if doc not in selected:
                selected.append(doc)
            if len(selected) >= k:
                break

    return selected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str) -> tuple[list[Document], dict[str, Any]]:
    """Run BM25 + Vector ensemble retrieval and return final context chunks.

    Returns ``(documents, timing)`` where *timing* is a dict of
    step-level durations in milliseconds.
    """
    import time as _time

    timing: dict[str, Any] = {}

    # Load full corpus for BM25
    t0 = _time.time()
    corpus = load_all_documents()
    timing["corpusLoadMs"] = int((_time.time() - t0) * 1000)
    timing["corpusSize"] = len(corpus)
    if not corpus:
        return [], timing

    # BM25 keyword retrieval (cached — only rebuilds when corpus changes)
    t0 = _time.time()
    bm25 = _bm25_cache.get(corpus, k=BM25_TOP_K)
    bm25_results = bm25.invoke(query)
    timing["bm25Ms"] = int((_time.time() - t0) * 1000)
    timing["bm25Hits"] = len(bm25_results)

    # Vector semantic retrieval
    t0 = _time.time()
    vector_results = _vector_retrieve(query, top_k=VECTOR_TOP_K)
    timing["vectorMs"] = int((_time.time() - t0) * 1000)
    timing["vectorHits"] = len(vector_results)

    if not bm25_results and not vector_results:
        return [], timing

    # Fuse both lists via RRF
    t0 = _time.time()
    fused = _reciprocal_rank_fusion(
        [bm25_results, vector_results],
        weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        source_names=["bm25", "vector"],
    )
    timing["rrfMs"] = int((_time.time() - t0) * 1000)

    # Cross-encoder reranker — score (query, doc) pairs jointly for
    # far more accurate relevance than rank-position-based RRF alone.
    candidates = fused[:RERANKER_TOP_K]
    t0 = _time.time()
    reranked = rerank(query, candidates, top_k=RERANKER_TOP_K)
    timing["rerankMs"] = int((_time.time() - t0) * 1000)
    timing["rerankCandidates"] = len(candidates)

    # Diversity filter → final K
    result = _diversify(reranked, k=FINAL_CONTEXT_K)
    timing["finalChunks"] = len(result)
    return result, timing
