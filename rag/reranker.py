"""Cross-encoder reranker using sentence-transformers.

Scores each (query, document) pair jointly with a cross-encoder model,
producing far more accurate relevance scores than rank-based fusion alone.

The model is lazily loaded and cached as a module-level singleton so the
~80 MB model weights are only loaded once per process.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.config import RERANKER_MODEL, RERANKER_TOP_K


# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_cross_encoder = None


def _get_cross_encoder():
    """Return the cached CrossEncoder instance, loading it on first call."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    documents: list[Document],
    top_k: int = RERANKER_TOP_K,
) -> list[Document]:
    """Score *documents* against *query* and return the top-*k* by relevance.

    Each returned document gets a ``rerank_score`` entry in its metadata.
    """
    if not documents:
        return []

    model = _get_cross_encoder()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]
    scores = model.predict(pairs)

    # Attach scores and sort descending
    scored: list[tuple[float, Document]] = []
    for score, doc in zip(scores, documents):
        doc.metadata["rerank_score"] = round(float(score), 6)
        scored.append((float(score), doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored[:top_k]]
