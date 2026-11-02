from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.answer import Answer, AnswerError, Citation, ClaudeAnswerer
from app.chunking import split_into_chunks
from app.embeddings import HashEmbedder
from app.store import InMemoryStore, SearchHit, StoredChunk

PUMP_MANUAL = """Infusion Pump IP-200 Maintenance Guide

Replace the pump battery every 24 months. Use only the approved lithium-ion battery pack.

Clean the housing with a damp cloth and mild detergent. Never immerse the pump in water.

Calibrate the flow sensor every 12 months using the calibration kit CK-9."""

VENT_MANUAL = """Ventilator V-50 Quick Reference

The ventilator alarm sounds when airway pressure exceeds the set limit.
Check the patient circuit for kinks before resetting the alarm."""


class StubAnswerer:
    def __init__(self, fail=False):
        self.fail = fail
        self.hits = None

    def answer(self, question, hits):
        self.hits = hits
        if self.fail:
            raise AnswerError("The model declined to answer this question")
        top = hits[0]
        return Answer("Replace it every 24 months.", [Citation(f"{top.chunk.document_title} (part 1)", top.chunk.text[:40])])


@pytest.fixture
def client_for():
    def _make(answerer):
        embedder = HashEmbedder()
        store = InMemoryStore(embedder.dim)
        main.app.dependency_overrides = {
            main.get_embedder: lambda: embedder,
            main.get_store: lambda: store,
            main.get_answerer: lambda: answerer,
        }
        return TestClient(main.app)

    yield _make
    main.app.dependency_overrides = {}


def upload(client, name, text):
    return client.post("/documents", files={"file": (name, text.encode(), "text/plain")})


def test_chunks_respect_size_and_overlap():
    text = "\n\n".join(" ".join(f"w{p}_{i}" for i in range(100)) for p in range(5))
    chunks = split_into_chunks(text, max_words=180, overlap_words=40)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 180 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-40:] == second[:40]


def test_chunking_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_into_chunks("a b c", max_words=10, overlap_words=10)


def test_upload_and_list_documents(client_for):
    client = client_for(StubAnswerer())
    doc = upload(client, "pump.md", PUMP_MANUAL).json()
    upload(client, "vent.txt", VENT_MANUAL)
    titles = {d["title"] for d in client.get("/documents").json()}
    assert doc["chunks"] >= 1
    assert titles == {"pump.md", "vent.txt"}


def test_ask_retrieves_the_relevant_document(client_for):
    answerer = StubAnswerer()
    client = client_for(answerer)
    upload(client, "pump.md", PUMP_MANUAL)
    upload(client, "vent.txt", VENT_MANUAL)

    r = client.post("/ask", json={"question": "How often should the pump battery be replaced?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Replace it every 24 months."
    assert body["sources"][0]["source"].startswith("pump.md")
    assert answerer.hits[0].chunk.document_title == "pump.md"
    assert body["citations"][0]["source"].startswith("pump.md")


def test_ask_with_no_documents_skips_the_model(client_for):
    answerer = StubAnswerer()
    client = client_for(answerer)
    body = client.post("/ask", json={"question": "What is the battery life?"}).json()
    assert body["sources"] == []
    assert answerer.hits is None


def test_model_failure_returns_502(client_for):
    client = client_for(StubAnswerer(fail=True))
    upload(client, "pump.md", PUMP_MANUAL)
    r = client.post("/ask", json={"question": "How often should the pump battery be replaced?"})
    assert r.status_code == 502


def test_delete_document(client_for):
    client = client_for(StubAnswerer())
    doc = upload(client, "pump.md", PUMP_MANUAL).json()
    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_rejects_unsupported_file(client_for):
    client = client_for(StubAnswerer())
    r = client.post("/documents", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 415


def test_claude_answerer_sends_citable_documents_and_parses_citations():
    hit = SearchHit(StoredChunk("d1", "pump.md", 0, "Replace the pump battery every 24 months."), 0.9)
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="text", text="Every 24 months.", citations=[
                SimpleNamespace(document_title="pump.md (part 1)", cited_text="Replace the pump battery every 24 months."),
            ]),
        ],
    )
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return response

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    result = ClaudeAnswerer(client=client, model="claude-opus-5-5").answer("How often?", [hit])

    doc = sent["messages"][0]["content"][0]
    assert doc["type"] == "document" and doc["citations"] == {"enabled": True}
    assert doc["title"] == "pump.md (part 1)"
    assert sent["fallbacks"] == "default"
    assert result.text == "Every 24 months."
    assert result.citations[0].source == "pump.md (part 1)"


def test_in_memory_store_ranks_by_cosine():
    store = InMemoryStore(2)
    store.add([StoredChunk("a", "a", 0, "a"), StoredChunk("b", "b", 0, "b")], np.array([[1, 0], [0, 1]], dtype=np.float32))
    hits = store.search(np.array([0.9, 0.1], dtype=np.float32), top_k=2)
    assert [h.chunk.document_id for h in hits] == ["a", "b"]
