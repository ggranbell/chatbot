"""Mola Chatbot — FastAPI server.

Drop-in replacement for the Node.js ``server.js``.  Exposes the same
endpoints so the frontend JS works without changes:

    GET  /                         → health check
    POST /ask-ai                   → plain chat
    POST /ask-ai-with-vector       → RAG chat
    POST /upload-knowledge-base    → PDF ingestion
    GET  /admin/db-rows            → preview stored rows
    POST /admin/db-truncate        → drop the table

Static files are served from ``src/`` (same as the Express setup).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.config import AUTH_KEY, SERVER_PORT
from rag.pipeline import (
    admin_db_rows,
    admin_db_truncate,
    ask_ai,
    ask_ai_with_vector,
    upload_pdf,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Mola Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS, JS, images) from src/
SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir():
    app.mount("/src", StaticFiles(directory=str(SRC_DIR)), name="static-src")
    # Also mount at root level so paths like ../css/output/output.css resolve
    app.mount("/css", StaticFiles(directory=str(SRC_DIR / "css")), name="static-css")
    app.mount("/js", StaticFiles(directory=str(SRC_DIR / "js")), name="static-js")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    message: str
    context: list[str] | None = None


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _check_auth(auth_key: str | None) -> None:
    if auth_key != AUTH_KEY:
        raise HTTPException(status_code=401, detail="Authentication key is invalid.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = SRC_DIR / "pages" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Server is online."})


@app.post("/ask-ai")
async def route_ask_ai(body: AskRequest, x_auth_key: str | None = Header(None)):
    _check_auth(x_auth_key)

    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")

    result = ask_ai(body.message)
    return result


@app.post("/ask-ai-with-vector")
async def route_ask_ai_with_vector(body: AskRequest, x_auth_key: str | None = Header(None)):
    _check_auth(x_auth_key)

    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")

    result = ask_ai_with_vector(body.message, context_history=body.context)

    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("message", "No context found."))

    return result


@app.post("/upload-knowledge-base")
async def route_upload_knowledge_base(
    file: UploadFile = File(...),
    label: str = Form(""),
):
    from rag.file_loader import SUPPORTED_EXTENSIONS, SUPPORTED_MIMETYPES

    # Validate by extension (more reliable than content-type for Office files)
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # Write upload to a temp file
    tmp_dir = Path(tempfile.gettempdir()) / "mola-chatbot-ingest"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{int(__import__('time').time())}-{filename}"
    tmp_path = tmp_dir / safe_name

    try:
        contents = await file.read()
        tmp_path.write_bytes(contents)
        result = upload_pdf(tmp_path, label=label)
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("message", "Ingestion failed."))
        return result
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@app.get("/admin/db-rows")
async def route_db_rows(
    limit: int = 20,
    x_auth_key: str | None = Header(None),
):
    _check_auth(x_auth_key)
    return admin_db_rows(limit=min(max(limit, 1), 100))


@app.post("/admin/db-truncate")
async def route_db_truncate(x_auth_key: str | None = Header(None)):
    _check_auth(x_auth_key)
    return admin_db_truncate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=SERVER_PORT, reload=True)
