"""Centralised configuration for the RAG pipeline.

All tunables live here so every module imports from one place.
Values are read from environment variables with sensible defaults.
"""

import os

# ---------------------------------------------------------------------------
# Ollama models
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1:8b")
EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------

# Tuned for DeepSeek R1 8B (larger context window, more reasoning capacity).
# Strategy: retrieve more, use larger/fewer chunks to maximize context usage.
VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "20"))
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))
FINAL_CONTEXT_K: int = int(os.getenv("FINAL_CONTEXT_K", "6"))
MAX_CHUNKS_PER_LABEL: int = int(os.getenv("MAX_CHUNKS_PER_LABEL", "3"))

# Weights for EnsembleRetriever (BM25 vs Vector). Must sum to 1.0.
# Larger models can balance BM25 and vector more evenly.
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.5"))
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Larger chunks = more efficient for bigger models with larger context.
CHUNK_TARGET_TOKENS: int = int(os.getenv("CHUNK_TARGET_TOKENS", "1200"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

# ---------------------------------------------------------------------------
# Vector DB
# ---------------------------------------------------------------------------
VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "data/vector-db")
VECTOR_TABLE_NAME: str = os.getenv("VECTOR_TABLE_NAME", "knowledge_base")

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_PORT: int = int(os.getenv("PORT", "8080"))
AUTH_KEY: str = os.getenv("AUTH_KEY", "SDP-AI-SERVER")

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "")
OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
OCR_LANG: str = os.getenv("OCR_LANG", "eng")
