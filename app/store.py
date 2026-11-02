"""Vector stores: in-memory (numpy) for development, PostgreSQL + pgvector for production."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StoredChunk:
    document_id: str
    document_title: str
    chunk_index: int
    text: str


@dataclass
class SearchHit:
    chunk: StoredChunk
    score: float  # cosine similarity, 1.0 = identical


class VectorStore(Protocol):
    def add(self, chunks: list[StoredChunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, top_k: int) -> list[SearchHit]: ...
    def documents(self) -> list[dict]: ...
    def delete_document(self, document_id: str) -> bool: ...


class InMemoryStore:
    def __init__(self, dim: int):
        self.chunks: list[StoredChunk] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, chunks, vectors):
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, vector, top_k):
        if not self.chunks:
            return []
        scores = self.vectors @ vector  # vectors are normalized, so this is cosine similarity
        order = np.argsort(-scores)[:top_k]
        return [SearchHit(self.chunks[i], float(scores[i])) for i in order]

    def documents(self):
        seen: dict[str, dict] = {}
        for c in self.chunks:
            doc = seen.setdefault(c.document_id, {"id": c.document_id, "title": c.document_title, "chunks": 0})
            doc["chunks"] += 1
        return list(seen.values())

    def delete_document(self, document_id):
        keep = [i for i, c in enumerate(self.chunks) if c.document_id != document_id]
        if len(keep) == len(self.chunks):
            return False
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep]
        return True


class PgVectorStore:
    """Stores chunks in PostgreSQL and ranks them with pgvector's cosine distance operator."""

    def __init__(self, url: str, dim: int):
        from pgvector.sqlalchemy import Vector
        from sqlalchemy import Integer, String, Text, create_engine, text
        from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

        class Base(DeclarativeBase):
            pass

        class ChunkRow(Base):
            __tablename__ = "chunks"
            id: Mapped[int] = mapped_column(Integer, primary_key=True)
            document_id: Mapped[str] = mapped_column(String(64), index=True)
            document_title: Mapped[str] = mapped_column(String(255))
            chunk_index: Mapped[int] = mapped_column(Integer)
            text: Mapped[str] = mapped_column(Text)
            embedding = mapped_column(Vector(dim))

        self.Row = ChunkRow
        self.engine = create_engine(url)
        with self.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as conn:
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            ))

    def add(self, chunks, vectors):
        from sqlalchemy.orm import Session

        with Session(self.engine) as s, s.begin():
            s.add_all(
                self.Row(document_id=c.document_id, document_title=c.document_title,
                         chunk_index=c.chunk_index, text=c.text, embedding=v.tolist())
                for c, v in zip(chunks, vectors)
            )

    def search(self, vector, top_k):
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        distance = self.Row.embedding.cosine_distance(vector.tolist())
        with Session(self.engine) as s:
            rows = s.execute(select(self.Row, distance.label("d")).order_by(distance).limit(top_k)).all()
        return [
            SearchHit(StoredChunk(r.document_id, r.document_title, r.chunk_index, r.text), 1.0 - float(d))
            for r, d in rows
        ]

    def documents(self):
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session

        stmt = select(self.Row.document_id, self.Row.document_title, func.count()).group_by(
            self.Row.document_id, self.Row.document_title
        )
        with Session(self.engine) as s:
            return [{"id": i, "title": t, "chunks": n} for i, t, n in s.execute(stmt)]

    def delete_document(self, document_id):
        from sqlalchemy import delete
        from sqlalchemy.orm import Session

        with Session(self.engine) as s, s.begin():
            return s.execute(delete(self.Row).where(self.Row.document_id == document_id)).rowcount > 0


def make_store(url: str, dim: int) -> VectorStore:
    return PgVectorStore(url, dim) if url.startswith("postgresql") else InMemoryStore(dim)
