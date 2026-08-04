"""Retrieve, build a grounded prompt, stream the answer from Ollama."""

from collections.abc import Iterator

import ollama

from app.config import (
    CHAT_MODEL,
    CITE_SOURCES,
    HISTORY_TURNS,
    OLLAMA_HOST,
    SYSTEM_PROMPT,
    TOP_K,
)
from app.store import search

_ollama = ollama.Client(host=OLLAMA_HOST)

NO_CONTEXT = (
    "I don't have any documents to work from yet. Upload one first and I'll "
    "answer from it."
)


def condense(question: str, history: list[dict]) -> str:
    """Build the string used for retrieval, not the one shown to the model.

    A follow-up like "explain that more simply" carries almost no meaning on
    its own, so embedding it retrieves noise. Prepending the previous question
    restores the missing subject. Doing it by concatenation rather than with a
    second LLM call keeps follow-ups as fast as opening questions.
    """
    previous = [m["content"] for m in history if m.get("role") == "user"]
    if not previous:
        return question
    return f"{previous[-1]}\n{question}"


def build_prompt(question: str, hits: list[dict]) -> str:
    # Filenames are withheld unless CITE_SOURCES is on — a model that never
    # sees them cannot repeat them, which is more reliable than instructing
    # it not to.
    blocks = [
        f"[Source {i}: {hit['source']}]\n{hit['text']}"
        if CITE_SOURCES
        else f"[Source {i}]\n{hit['text']}"
        for i, hit in enumerate(hits, start=1)
    ]
    context = "\n\n---\n\n".join(blocks)
    return f"Context:\n\n{context}\n\nQuestion: {question}"


def answer(
    question: str, top_k: int = TOP_K, history: list[dict] | None = None
) -> Iterator[str]:
    """Yield the answer token by token, so the page can render as it arrives."""
    history = history or []
    hits = search(condense(question, history), top_k=top_k)
    yield from stream_answer(question, hits, history)


def stream_answer(
    question: str, hits: list[dict], history: list[dict] | None = None
) -> Iterator[str]:
    """Same as `answer`, but for callers that already ran retrieval."""
    if not hits:
        yield NO_CONTEXT
        return

    # Earlier turns go in verbatim; only the newest one carries retrieved
    # context, so the prompt doesn't grow by a full context block per turn.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_recent(history))
    messages.append({"role": "user", "content": build_prompt(question, hits)})

    stream = _ollama.chat(model=CHAT_MODEL, messages=messages, stream=True)
    for part in stream:
        token = part.get("message", {}).get("content", "")
        if token:
            yield token


def _recent(history: list[dict] | None) -> list[dict]:
    """The last few well-formed turns, oldest first."""
    if not history:
        return []
    clean = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]
    return clean[-HISTORY_TURNS * 2 :]
