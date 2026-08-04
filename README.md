# Local RAG with Ollama

A retrieval-augmented-generation web app that runs entirely on your machine.
Upload PDFs, text, or Markdown; ask questions; get answers grounded in those
documents with citations. No API keys, no cloud, nothing leaves your computer.

- **Ollama** — embeddings and text generation, both local
- **NumPy** — vectors persisted to disk, cosine search by dot product
- **FastAPI** — backend and streaming API
- **Plain HTML/JS** — no build step, no npm

No compiler needed: every dependency ships prebuilt wheels on Windows, macOS,
and Linux.

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
source .venv/bin/activate
pip install -r requirements.txt
```

On **Windows PowerShell**, `source` doesn't exist — use the venv's Python
directly, which avoids activation entirely:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Python 3.10 or newer. Python 3.13 is fine.

## 3. Run it

```bash
uvicorn app.main:app --reload
```

Windows PowerShell:

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open <http://localhost:8000>. The dot in the sidebar turns green once Ollama is
reachable and both models are pulled.

Drop a PDF into the sidebar, wait for "Indexed", then ask a question. Answers
stream in token by token, with the matched sources shown underneath — click a
source chip to read the exact text the answer was drawn from.

Follow-up questions work: "explain that more simply" keeps the previous turn in
view. "Start a new conversation" in the sidebar clears that memory without
touching your indexed documents.

By default answers never mention filenames — the model isn't shown them. Set
`CITE_SOURCES=true` if you would rather it cite documents by name in the text.

## How it works

```
upload ──> extract text ──> chunk (1000 chars, 150 overlap)
                                  │
                                  ├──> Ollama embeds each chunk
                                  └──> stored in store_data/vectors.npz

ask ──> Ollama embeds the question
            │
            ├──> dot product against every chunk -> 4 nearest (cosine)
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
| `HISTORY_TURNS` | `3` | Previous Q&A pairs carried into a follow-up |
| `CITE_SOURCES` | `false` | Let the model name source files in its answer |
| `SYSTEM_PROMPT` | *(see config.py)* | Overrides the assistant's instructions wholesale |
| `STORE_DIR` | `./store_data` | Vector store location |

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

**`pip install` fails building a wheel** — shouldn't happen any more, since
nothing here needs compiling. If it does, make sure you pulled the latest of
this branch; earlier revisions depended on ChromaDB, which requires a C++
toolchain on Windows.

**A PDF indexes as 0 chunks** — it's a scanned image with no text layer. This
app doesn't do OCR; run the PDF through OCR first.

**Answers mention the PDF filename** — set `CITE_SOURCES=false` (the default).
If it still happens, the model is echoing a name that appears inside the
document text itself, which no prompt setting can remove.

**A follow-up question gets the wrong context** — retrieval blends the previous
question with the new one. Ask a fully-worded question, or click "Start a new
conversation" when changing topic.

**Answers ignore the document** — raise `TOP_K` to pull in more context, or
lower `CHUNK_SIZE` so retrieval is more precise.
