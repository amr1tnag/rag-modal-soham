"""Turn raw uploaded bytes into overlapping text chunks."""

import io
import re

from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def extract_text(filename: str, data: bytes) -> str:
    """Pull plain text out of a PDF or a text-like file."""
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    return data.decode("utf-8", errors="ignore")


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split on paragraph boundaries, packing paragraphs up to `size` characters.

    Consecutive chunks share `overlap` characters of tail text so a sentence
    straddling a boundary still retrieves cleanly.
    """
    text = _normalise(text)
    if not text:
        return []

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # A single oversized paragraph gets hard-split on its own.
        if len(para) > size:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(para), size - overlap):
                chunks.append(para[i : i + size])
            continue

        if len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}" if current else para
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]
