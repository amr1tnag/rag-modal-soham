"""Ollama embeddings over a small on-disk vector store.

Deliberately dependency-light: vectors live in one .npz file and search is a
NumPy dot product. For the tens of thousands of chunks a personal document
set produces, an exhaustive scan takes milliseconds — an ANN index like FAISS
or hnswlib only earns its keep (and its C++ toolchain) past ~1M vectors.
"""

import json
import threading
from pathlib import Path

import numpy as np
import ollama

from app.config import EMBED_MODEL, OLLAMA_HOST, STORE_DIR, TOP_K

_ollama = ollama.Client(host=OLLAMA_HOST)

_DIR = Path(STORE_DIR)
_VECTORS = _DIR / "vectors.npz"
_META = _DIR / "meta.json"

# Uvicorn serves requests from a thread pool; guard the shared arrays.
_lock = threading.Lock()

_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)
_records: list[dict] = []


def _load() -> None:
    """Read the store from disk into memory. Called once at import."""
    global _matrix, _records
    if _VECTORS.exists() and _META.exists():
        _matrix = np.load(_VECTORS)["vectors"].astype(np.float32)
        _records = json.loads(_META.read_text(encoding="utf-8"))
    else:
        _matrix = np.zeros((0, 0), dtype=np.float32)
        _records = []


def _save() -> None:
    _DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_VECTORS, vectors=_matrix)
    _META.write_text(json.dumps(_records), encoding="utf-8")


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product *is* cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of strings with the local Ollama embedding model."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    response = _ollama.embed(model=EMBED_MODEL, input=texts)
    return np.asarray(response["embeddings"], dtype=np.float32)


def add_document(filename: str, chunks: list[str]) -> int:
    """Store every chunk of one document. Returns the number stored."""
    global _matrix, _records
    if not chunks:
        return 0

    # Embed outside the lock — this is the slow part and needs no shared state.
    vectors = _normalise(embed(chunks))

    with _lock:
        if _matrix.size and _matrix.shape[1] != vectors.shape[1]:
            raise ValueError(
                f"Embedding dimension changed ({_matrix.shape[1]} -> "
                f"{vectors.shape[1]}). The embedding model was switched; "
                "clear the store and re-upload your documents."
            )
        _matrix = vectors if not _matrix.size else np.vstack([_matrix, vectors])
        _records.extend(
            {"source": filename, "chunk": i, "text": text}
            for i, text in enumerate(chunks)
        )
        _save()

    return len(chunks)


def search(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the chunks most similar to `question`, best match first."""
    with _lock:
        if not _records:
            return []
        matrix, records = _matrix, list(_records)

    query = _normalise(embed([question]))[0]
    scores = matrix @ query

    # argpartition finds the top k without sorting all of them.
    k = min(top_k, len(records))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]

    return [
        {
            "text": records[i]["text"],
            "source": records[i]["source"],
            "score": round(float(scores[i]), 3),
        }
        for i in top
    ]


def list_sources() -> list[dict]:
    """Every distinct filename in the store, with its chunk count."""
    with _lock:
        counts: dict[str, int] = {}
        for record in _records:
            counts[record["source"]] = counts.get(record["source"], 0) + 1
    return [{"source": name, "chunks": n} for name, n in sorted(counts.items())]


def delete_source(filename: str) -> None:
    global _matrix, _records
    with _lock:
        keep = [i for i, r in enumerate(_records) if r["source"] != filename]
        if len(keep) == len(_records):
            return
        _records = [_records[i] for i in keep]
        _matrix = (
            _matrix[keep]
            if keep
            else np.zeros((0, 0), dtype=np.float32)
        )
        _save()


def reset() -> None:
    """Drop every document."""
    global _matrix, _records
    with _lock:
        _matrix = np.zeros((0, 0), dtype=np.float32)
        _records = []
        _save()


def count() -> int:
    with _lock:
        return len(_records)


_load()
