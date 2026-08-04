# Local RAG with Ollama

A retrieval-augmented-generation web app that runs entirely on your machine.
Upload PDFs, text, or Markdown; ask questions; get answers grounded in those
documents with citations. No API keys, no cloud, nothing leaves your computer.

- **Ollama** — embeddings and text generation, both local
- **ChromaDB** — persistent vector store on disk
- **FastAPI** — backend and streaming API
- **Plain HTML/JS** — no build step, no npm

## 1. Install Ollama

Download from <https://ollama.com/download> (macOS, Windows, Linux). On Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then pull the two models this app uses:

```bash
ollama pull nomic-embed-text   # embeddings, ~275 MB
ollama pull llama3.2           # generation, ~2 GB
```

Leave Ollama running. The desktop app starts automatically; on a server run
`ollama serve`. Verify it responds:

```bash
curl http://localhost:11434/api/tags
```

## 2. Install the app

```bash
git clone https://github.com/amr1tnag/rag-modal-soham.git
cd rag-modal-soham

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## 3. Run it

```bash
uvicorn app.main:app --reload
```

Open <http://localhost:8000>. The dot in the sidebar turns green once Ollama is
reachable and both models are pulled.

Drop a PDF into the sidebar, wait for "Indexed", then ask a question. Answers
stream in token by token, with the source documents shown underneath.

## How it works

```
upload ──> extract text ──> chunk (1000 chars, 150 overlap)
                                  │
                                  ├──> Ollama embeds each chunk
                                  └──> stored in ChromaDB

ask ──> Ollama embeds the question
            │
            ├──> Chroma returns the 4 nearest chunks (cosine)
            └──> chunks + question ──> llama3.2 ──> streamed answer
```

The model is instructed to answer only from the retrieved context and to say so
when the answer isn't there — that's what keeps it from inventing facts.

## Configuration

Every setting is an environment variable with a sensible default
(see `app/config.py`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama listens |
| `CHAT_MODEL` | `llama3.2` | Generation model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `1000` | Chunk length in characters |
| `CHUNK_OVERLAP` | `150` | Shared characters between chunks |
| `TOP_K` | `4` | Chunks retrieved per question |
| `CHROMA_DIR` | `./chroma_db` | Vector store location |

Example — use a bigger model:

```bash
ollama pull mistral
CHAT_MODEL=mistral uvicorn app.main:app --reload
```

Changing `EMBED_MODEL` invalidates existing vectors; clear the store
(sidebar → "Clear all documents") and re-upload.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Ollama status, model readiness, chunk count |
| `POST` | `/api/upload` | Multipart file upload; indexes it |
| `GET` | `/api/sources` | Indexed filenames and chunk counts |
| `DELETE` | `/api/sources/{filename}` | Remove one document |
| `POST` | `/api/reset` | Drop everything |
| `POST` | `/api/ask` | `{"question": "..."}` → NDJSON token stream |

Interactive docs at <http://localhost:8000/docs>.

## Troubleshooting

**Sidebar says "Ollama not reachable"** — Ollama isn't running. Start the
desktop app, or `ollama serve`.

**"Pull the models"** — run the two `ollama pull` commands above.

**Answers are slow** — generation is CPU-bound without a GPU. Try a smaller
model: `ollama pull llama3.2:1b` then `CHAT_MODEL=llama3.2:1b`.

**A PDF indexes as 0 chunks** — it's a scanned image with no text layer. This
app doesn't do OCR; run the PDF through OCR first.

**Answers ignore the document** — raise `TOP_K` to pull in more context, or
lower `CHUNK_SIZE` so retrieval is more precise.
