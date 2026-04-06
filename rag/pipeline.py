"""RAG pipeline orchestrator.

Ties every module together into high-level functions consumed by the
FastAPI server (``app.py``). Each function returns a typed dict that
the API layer serialises straight to JSON.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.chain import ask_plain, ask_with_context
from rag.config import CHAT_MODEL, EMBEDDING_MODEL
from rag.file_loader import load_document, SUPPORTED_EXTENSIONS, SUPPORTED_MIMETYPES
from rag.retriever import retrieve
from rag.text_splitter import split_documents
from rag.vectorstore import get_rows, ingest_documents, truncate_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(chunks: list) -> str:
    """Build the context string the LLM will see.

    Accepts both LangChain ``Document`` objects and plain dicts.
    """
    parts: list[str] = []
    for item in chunks:
        # Support Document objects (new) and dicts (legacy)
        if hasattr(item, "page_content"):
            label = item.metadata.get("label", "N/A")
            keywords = item.metadata.get("keywords", "")
            text = item.page_content
        else:
            label = item.get("label", "N/A")
            keywords = item.get("keywords", "")
            text = item.get("text", "")
        parts.append(f"[Label: {label}]\n[Keywords: {keywords}]\n[Text: {text}]")
    return "\n---\n".join(parts)


def _chunk_debug_info(chunks: list) -> list[dict[str, Any]]:
    """Build a JSON-safe debug payload for each selected chunk."""
    info = []
    for rank, item in enumerate(chunks, start=1):
        if hasattr(item, "page_content"):
            meta = item.metadata
            dist = meta.get("_distance")
            info.append({
                "rank": rank,
                "label": meta.get("label", "N/A"),
                "keywords": meta.get("keywords", ""),
                "rrfScore": meta.get("rrf_score", 0),
                "rerankScore": meta.get("rerank_score"),
                "bm25Rrf": meta.get("bm25_rrf", 0),
                "vectorRrf": meta.get("vector_rrf", 0),
                "vectorDistance": round(dist, 6) if isinstance(dist, (int, float)) else None,
                "text": item.page_content,
            })
        else:
            info.append({
                "rank": rank,
                "label": item.get("label", "N/A"),
                "keywords": item.get("keywords", ""),
                "text": item.get("text", ""),
            })
    return info


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------

def ask_ai(message: str) -> dict[str, Any]:
    """Plain chat — no retrieval."""
    answer = ask_plain(message)
    return {
        "success": True,
        "model": CHAT_MODEL,
        "message": answer,
        "timestamp": datetime.now().isoformat(),
    }


def ask_ai_with_vector(message: str, context_history: list[str] | None = None) -> dict[str, Any]:
    """Full RAG pipeline: retrieve → re-rank → generate."""
    request_start = time.time()
    time_logs: dict[str, Any] = {}

    # Build retrieval query (optionally include conversation history)
    history = ""
    if context_history:
        recent = [h.strip() for h in context_history if h.strip()][-3:]
        history = "\n".join(recent)

    retrieval_query = (
        f"Conversation context:\n{history}\n\nCurrent user question:\n{message}"
        if history
        else message
    )

    # Retrieve & re-rank
    t0 = time.time()
    ranked, retrieval_timing = retrieve(retrieval_query)
    time_logs["retrievalMs"] = int((time.time() - t0) * 1000)
    time_logs.update(retrieval_timing)

    if not ranked:
        return {
            "success": False,
            "message": "No relevant knowledge base context found.",
        }

    # Build context & generate
    context_text = _format_context(ranked)
    time_logs["contextChars"] = len(context_text)

    # Pass recent conversation history to the LLM so it can reference
    # prior turns when answering follow-up questions.
    history_for_llm = ""
    if context_history:
        recent = [h.strip() for h in context_history if h.strip()][-3:]
        history_for_llm = "\n".join(recent)

    t0 = time.time()
    answer = ask_with_context(
        message, context_text, conversation_history=history_for_llm
    )
    time_logs["generationMs"] = int((time.time() - t0) * 1000)
    time_logs["totalMs"] = int((time.time() - request_start) * 1000)

    return {
        "success": True,
        "model": CHAT_MODEL,
        "response": answer,
        "retrievalCount": len(ranked),
        "selectedChunks": _chunk_debug_info(ranked),
        "timeLogs": time_logs,
        "timestamp": datetime.now().isoformat(),
    }


def upload_pdf(
    pdf_path: str | Path,
    label: str = "",
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Ingest a document (PDF, DOCX, XLSX, TXT) into the vector store.

    Steps: load → split → embed → store.

    Despite the name (kept for backward compatibility), this now accepts
    any format supported by ``rag.file_loader``.
    """
    docs = load_document(pdf_path, label=label)
    if not docs:
        return {"success": False, "message": "No extractable text found in the file."}

    chunks = split_documents(docs)
    if not chunks:
        return {"success": False, "message": "Splitting produced zero chunks."}

    rows_added = ingest_documents(
        chunks,
        label=label,
        embedding_model=embedding_model or EMBEDDING_MODEL,
    )

    return {
        "success": True,
        "message": f"Knowledge base updated. Added {rows_added} chunks.",
        "rowsAdded": rows_added,
    }


def admin_db_rows(limit: int = 20) -> dict[str, Any]:
    """Return a preview of knowledge base rows."""
    data = get_rows(limit=limit)
    return {"success": True, **data}


def admin_db_truncate() -> dict[str, Any]:
    """Drop the knowledge base table."""
    existed = truncate_table()
    return {
        "success": True,
        "message": "knowledge_base table truncated." if existed else "Table did not exist.",
    }
