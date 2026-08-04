"""Ollama embeddings + a persistent Chroma collection."""

import logging
import uuid

import chromadb
import ollama

from app.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    OLLAMA_HOST,
    TOP_K,
)

# Chroma 0.5.23 ships a telemetry call that mismatches newer posthog and logs a
# noisy error on every operation. It is harmless, and disabled anyway — mute it.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

_ollama = ollama.Client(host=OLLAMA_HOST)
_chroma = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=chromadb.config.Settings(anonymized_telemetry=False),
)

# embedding_function=None because we always supply vectors ourselves; this also
# stops Chroma from downloading its bundled ONNX model on first use.
_collection = _chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings with the local Ollama embedding model."""
    if not texts:
        return []
    return _ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]


def add_document(filename: str, chunks: list[str]) -> int:
    """Store every chunk of one document. Returns the number stored."""
    if not chunks:
        return 0

    vectors = embed(chunks)
    _collection.add(
        ids=[str(uuid.uuid4()) for _ in chunks],
        embeddings=vectors,
        documents=chunks,
        metadatas=[
            {"source": filename, "chunk": i} for i in range(len(chunks))
        ],
    )
    return len(chunks)


def search(question: str, top_k: int = TOP_K) -> list[dict]:
    """Return the chunks most similar to `question`, best match first."""
    if _collection.count() == 0:
        return []

    result = _collection.query(
        query_embeddings=embed([question]),
        n_results=min(top_k, _collection.count()),
    )

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                # Cosine distance -> a 0..1 similarity that reads naturally.
                "score": round(1 - distance, 3),
            }
        )
    return hits


def list_sources() -> list[dict]:
    """Every distinct filename in the store, with its chunk count."""
    if _collection.count() == 0:
        return []
    metadatas = _collection.get(include=["metadatas"])["metadatas"]
    counts: dict[str, int] = {}
    for meta in metadatas:
        name = meta.get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return [
        {"source": name, "chunks": n} for name, n in sorted(counts.items())
    ]


def delete_source(filename: str) -> None:
    _collection.delete(where={"source": filename})


def reset() -> None:
    """Drop every document and recreate an empty collection."""
    global _collection
    _chroma.delete_collection(COLLECTION_NAME)
    _collection = _chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def count() -> int:
    return _collection.count()
