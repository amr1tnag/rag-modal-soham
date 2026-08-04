"""FastAPI app: upload documents, ask questions, serve the web UI."""

import json
from collections.abc import Iterator
from pathlib import Path

import ollama
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import store
from app.chunking import chunk_text, extract_text
from app.config import CHAT_MODEL, EMBED_MODEL, OLLAMA_HOST, TOP_K
from app.rag import stream_answer

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="Local RAG")


class Question(BaseModel):
    question: str
    top_k: int = TOP_K


@app.get("/api/health")
def health() -> dict:
    """Report whether Ollama is reachable and the needed models are pulled."""
    try:
        tags = ollama.Client(host=OLLAMA_HOST).list()
        installed = {m.get("model", "").split(":")[0] for m in tags["models"]}
        return {
            "ollama": "up",
            "chat_model": CHAT_MODEL,
            "embed_model": EMBED_MODEL,
            "chat_model_ready": CHAT_MODEL.split(":")[0] in installed,
            "embed_model_ready": EMBED_MODEL.split(":")[0] in installed,
            "chunks": store.count(),
        }
    except Exception as exc:
        return {"ollama": "down", "error": str(exc), "chunks": store.count()}


@app.post("/api/upload")
async def upload(file: UploadFile) -> dict:
    name = file.filename or "untitled"
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400, f"Unsupported file type. Allowed: {sorted(ALLOWED_SUFFIXES)}"
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 20 MB.")

    try:
        text = extract_text(name, data)
    except Exception as exc:
        raise HTTPException(400, f"Could not read {name}: {exc}") from exc

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, f"No readable text found in {name}.")

    try:
        stored = store.add_document(name, chunks)
    except Exception as exc:
        raise HTTPException(
            503, f"Embedding failed — is Ollama running? ({exc})"
        ) from exc

    return {"source": name, "chunks": stored}


@app.get("/api/sources")
def sources() -> dict:
    return {"sources": store.list_sources(), "chunks": store.count()}


@app.delete("/api/sources/{filename}")
def delete_source(filename: str) -> dict:
    store.delete_source(filename)
    return {"deleted": filename}


@app.post("/api/reset")
def reset() -> dict:
    store.reset()
    return {"status": "cleared"}


@app.post("/api/ask")
def ask(payload: Question) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    try:
        hits = store.search(question, top_k=payload.top_k)
    except Exception as exc:
        raise HTTPException(
            503, f"Retrieval failed — is Ollama running? ({exc})"
        ) from exc

    def events() -> Iterator[str]:
        # Sources go first so the UI can show citations while tokens stream in.
        yield json.dumps({"type": "sources", "sources": hits}) + "\n"
        try:
            for token in stream_answer(question, hits):
                yield json.dumps({"type": "token", "text": token}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
