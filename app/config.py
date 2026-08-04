"""Central configuration. Override any value with an environment variable."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where Chroma persists its vectors on disk.
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documents")

# Ollama. Pull these first:  ollama pull nomic-embed-text && ollama pull llama3.2
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2")

# Chunking, in characters.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# How many chunks to feed the model as context.
TOP_K = int(os.getenv("TOP_K", "4"))

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only \
the provided context. If the context does not contain the answer, say so plainly \
instead of guessing. Cite the source filename for each claim you make."""
