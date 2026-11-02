# RAG Q&A Assistant

Upload documents (.txt, .md or .pdf) and ask questions about them. Answers come only from your documents and quote the passages they rely on.

**Stack:** Python, FastAPI, Claude API (`claude-opus-5-5`) with Citations, local embeddings (fastembed, BAAI/bge-small-en-v1.5), PostgreSQL + pgvector (HNSW index) or an in-memory store, React + TypeScript (Vite), pytest, GitHub Actions.

## How it works

```
upload ──> extract text (pypdf) ──> overlapping chunks ──> embeddings ──> vector store
                                                                              │
question ──> embedding ──> top-k cosine search ──> chunks above a score threshold
                                                        │
                                   Claude, one citable document per chunk ──> answer + quoted sources
```

- **Chunking** (`app/chunking.py`): ~180-word windows with a 40-word overlap that keep paragraphs together, so an answer split across a boundary is still retrievable.
- **Embeddings** (`app/embeddings.py`): bge-small runs locally through ONNX, so indexing costs nothing and needs no key. A hashing embedder keeps the tests offline.
- **Vector store** (`app/store.py`): in memory with numpy for development; PostgreSQL + pgvector with an HNSW cosine index when `DATABASE_URL` is set.
- **Grounded answers** (`app/answer.py`): each retrieved chunk is sent to Claude as a document with citations enabled. The response carries the exact quoted text and which chunk it came from, and the UI shows both.
- **No wasted calls:** if nothing scores above the threshold, the API answers "not found" without calling the model.

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on macOS/Linux
pip install -r requirements-dev.txt
cp .env.example .env            # then set ANTHROPIC_API_KEY
uvicorn app.main:app --reload   # API on http://localhost:8000/docs

cd web
npm install
npm run dev                     # UI on http://localhost:5173
```

For persistent storage:

```bash
docker compose up -d            # PostgreSQL 16 with pgvector
# then in .env: DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5432/rag
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | Upload and index a file |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{id}` | Remove a document and its chunks |
| POST | `/ask` | `{"question": "...", "top_k": 5}` returns `answer`, `citations`, `sources` |

## Tests

```bash
pytest -q
```

Covers chunk sizes and overlap, upload and listing, retrieval picking the right document, skipping the model when nothing matches, error handling, deletion, cosine ranking, and the citation request and parsing.
