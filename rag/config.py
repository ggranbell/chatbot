"""Centralised configuration for the RAG pipeline.

All tunables live here so every module imports from one place.
Values are read from environment variables with sensible defaults.
"""

import os

# ---------------------------------------------------------------------------
# Ollama models
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3.5:9b")
EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# LLM generation speed
# ---------------------------------------------------------------------------
# Smaller num_ctx reduces KV-cache overhead → faster token generation.
# Keep it large enough for the RAG context + answer but well under the 32K max.
NUM_CTX: int = int(os.getenv("NUM_CTX", "8192"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
NUM_PREDICT: int = int(os.getenv("NUM_PREDICT", "1024"))
REPEAT_PENALTY: float = float(os.getenv("REPEAT_PENALTY", "1.1"))

# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------

# Tuned for Qwen 3.5 9B (32K context window, strong semantic understanding).
# Strategy: retrieve more candidates, use larger chunks, bias toward vector search.
VECTOR_TOP_K: int = int(os.getenv("VECTOR_TOP_K", "15"))
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "15"))
FINAL_CONTEXT_K: int = int(os.getenv("FINAL_CONTEXT_K", "5"))
MAX_CHUNKS_PER_LABEL: int = int(os.getenv("MAX_CHUNKS_PER_LABEL", "2"))

# Cross-encoder reranker (sentence-transformers)
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "10"))

# Weights for EnsembleRetriever (BM25 vs Vector). Must sum to 1.0.
# Qwen has strong semantic embeddings; favour vector retrieval slightly.
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.35"))
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.65"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Larger chunks exploit Qwen 3.5 9B's 32K context window efficiently.
CHUNK_TARGET_TOKENS: int = int(os.getenv("CHUNK_TARGET_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

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
