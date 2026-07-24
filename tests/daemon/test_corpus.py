import httpx
import pytest
from httpx import ASGITransport

FIXED = [0.3] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("efficient.modules.hybrid_retrieval.embed_documents",
                        lambda ts: [FIXED] * len(ts))
    monkeypatch.setattr("efficient.modules.hybrid_retrieval.embed_query",
                        lambda t: FIXED)


@pytest.fixture
async def client(efficient_db):
    from efficient.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_add_chunks_stores_and_counts(client, efficient_db):
    from efficient.db.collections import CORPUS_CHUNKS
    r = await client.post("/corpus/add-chunks", json={
        "corpus_id": "docs",
        "chunks": [
            {"text": "MongoDB is a document database.", "source_file": "a.md"},
            {"text": "Redis is an in-memory store.", "source_file": "a.md"},
        ],
    })
    assert r.status_code == 200
    assert r.json() == {"corpus_id": "docs", "added": 2}
    n = await efficient_db[CORPUS_CHUNKS].count_documents({"corpus_id": "docs"})
    assert n == 2
    # chunk_index auto-assigned when omitted
    idxs = sorted([d["chunk_index"] async for d in
                   efficient_db[CORPUS_CHUNKS].find({"corpus_id": "docs"})])
    assert idxs == [0, 1]


async def test_add_chunks_metrics_store(client):
    await client.post("/corpus/add-chunks", json={
        "corpus_id": "docs", "chunks": [{"text": "hello", "source_file": "x"}]})
    r = await client.get("/metrics")
    assert r.json()["store"]["corpus_chunks"] >= 1


async def test_remove_file(client, efficient_db):
    from efficient.db.collections import CORPUS_CHUNKS
    await client.post("/corpus/add-chunks", json={
        "corpus_id": "docs", "chunks": [
            {"text": "a", "source_file": "keep.md"},
            {"text": "b", "source_file": "drop.md"},
        ]})
    r = await client.post("/corpus/remove-file",
                          json={"corpus_id": "docs", "source_file": "drop.md"})
    assert r.json()["removed"] == 1
    remaining = [d["source_file"] async for d in
                 efficient_db[CORPUS_CHUNKS].find({"corpus_id": "docs"})]
    assert remaining == ["keep.md"]
