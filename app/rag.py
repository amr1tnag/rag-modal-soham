"""Retrieve, build a grounded prompt, stream the answer from Ollama."""

from collections.abc import Iterator

import ollama

from app.config import CHAT_MODEL, OLLAMA_HOST, SYSTEM_PROMPT, TOP_K
from app.store import search

_ollama = ollama.Client(host=OLLAMA_HOST)

NO_CONTEXT = (
    "I don't have any documents to work from yet. Upload one first and I'll "
    "answer from it."
)


def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = [
        f"[Source {i}: {hit['source']}]\n{hit['text']}"
        for i, hit in enumerate(hits, start=1)
    ]
    context = "\n\n---\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def answer(question: str, top_k: int = TOP_K) -> Iterator[str]:
    """Yield the answer token by token, so the page can render as it arrives."""
    yield from stream_answer(question, search(question, top_k=top_k))


def stream_answer(question: str, hits: list[dict]) -> Iterator[str]:
    """Same as `answer`, but for callers that already ran retrieval."""
    if not hits:
        yield NO_CONTEXT
        return

    stream = _ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        stream=True,
    )
    for part in stream:
        token = part.get("message", {}).get("content", "")
        if token:
            yield token
