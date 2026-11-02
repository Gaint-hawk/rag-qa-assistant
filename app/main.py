import io
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.answer import Answerer, AnswerError, ClaudeAnswerer, source_label
from app.chunking import split_into_chunks
from app.config import settings
from app.embeddings import Embedder, make_embedder
from app.store import StoredChunk, VectorStore, make_store

app = FastAPI(title="RAG Q&A Assistant", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
)

ALLOWED = {".txt", ".md", ".pdf"}
MAX_BYTES = 10 * 1024 * 1024


@lru_cache
def get_embedder() -> Embedder:
    return make_embedder(settings.embedder)


@lru_cache
def get_store() -> VectorStore:
    return make_store(settings.database_url, get_embedder().dim)


@lru_cache
def get_answerer() -> Answerer:
    return ClaudeAnswerer()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=settings.top_k, ge=1, le=20)


class SourceOut(BaseModel):
    source: str
    score: float


class CitationOut(BaseModel):
    source: str
    cited_text: str


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    sources: list[SourceOut]


def extract_text(filename: str, data: bytes) -> str:
    if Path(filename).suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="replace")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents", status_code=201)
def upload_document(
    file: UploadFile = File(...),
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_store),
):
    name = file.filename or "document.txt"
    if Path(name).suffix.lower() not in ALLOWED:
        raise HTTPException(415, "Upload a .txt, .md or .pdf file")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 10 MB")

    chunks = split_into_chunks(extract_text(name, data), settings.chunk_words, settings.chunk_overlap_words)
    if not chunks:
        raise HTTPException(422, "No text found in the file")

    doc_id = uuid.uuid4().hex[:12]
    stored = [StoredChunk(doc_id, name, c.index, c.text) for c in chunks]
    store.add(stored, embedder.embed([c.text for c in chunks]))
    return {"id": doc_id, "title": name, "chunks": len(chunks)}


@app.get("/documents")
def list_documents(store: VectorStore = Depends(get_store)):
    return store.documents()


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: str, store: VectorStore = Depends(get_store)):
    if not store.delete_document(document_id):
        raise HTTPException(404, "Document not found")


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_store),
    answerer: Answerer = Depends(get_answerer),
):
    query = embedder.embed([body.question])[0]
    hits = [h for h in store.search(query, body.top_k) if h.score >= settings.min_score]
    sources = [SourceOut(source=source_label(h), score=round(h.score, 3)) for h in hits]
    if not hits:
        # Nothing relevant was retrieved: answer without spending an API call.
        return AskResponse(answer="I could not find anything about that in the uploaded documents.",
                           citations=[], sources=[])
    try:
        result = answerer.answer(body.question, hits)
    except AnswerError as e:
        raise HTTPException(502, str(e)) from e
    return AskResponse(
        answer=result.text,
        citations=[CitationOut(source=c.source, cited_text=c.cited_text) for c in result.citations],
        sources=sources,
    )
