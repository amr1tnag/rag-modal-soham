"""Central configuration. Override any value with an environment variable."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where the vector store persists to disk.
STORE_DIR = os.getenv("STORE_DIR", str(BASE_DIR / "store_data"))

# Ollama. Pull these first:  ollama pull nomic-embed-text && ollama pull llama3.2
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

# Chunking, in characters.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# How many chunks to feed the model as context.
TOP_K = int(os.getenv("TOP_K", "4"))

# How many previous question/answer pairs to carry into a follow-up.
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", "3"))

# When true, the model is told to name its source files in the answer text.
# Off by default: filenames read as clutter, and the sidebar already shows
# which documents an answer drew on.
CITE_SOURCES = os.getenv("CITE_SOURCES", "false").lower() in ("1", "true", "yes")

_BASE_PROMPT = """You are a helpful assistant that answers questions using only \
the provided context. If the context does not contain the answer, say so plainly \
instead of guessing."""

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    _BASE_PROMPT
    + (
        " Cite the source filename for each claim you make."
        if CITE_SOURCES
        else " Answer directly, in your own words. Never mention the context, "
        "the sources, the documents, or any filename — write as though you "
        "simply know the material."
    ),
)
