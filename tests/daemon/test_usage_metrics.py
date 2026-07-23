import httpx
import pytest
from httpx import ASGITransport

from efficient.db.collections import REQUEST_LOG

SOURCE = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n" + ("# pad\n" * 40)


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    fixed = [0.1] * 1024
    monkeypatch.setattr("efficient.modules.codebase_graph.embed_query", lambda t: fixed)
    monkeypatch.setattr("efficient.modules.codebase_graph.embed_documents", lambda ts: [fixed] * len(ts))


@pytest.fixture
async def client(efficient_db):
    from efficient.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_query_records_honest_savings(client, efficient_db, monkeypatch):
    # fixed vectors make any stored symbol a top match for any query
    async def fake_vs(collection, pipeline):
        cursor = collection.find({})
        return [{**d, "_score": 0.99} async for d in cursor]
    monkeypatch.setattr("efficient.modules.codebase_graph.vector_search", fake_vs)

    await client.post("/codebase/index-file", json={
        "repo_id": "proj", "file_path": "m.py", "source": SOURCE,
    })
    r = await client.post("/codebase/query", json={"repo_id": "proj", "query": "add numbers", "k": 5})
    assert r.status_code == 200
    assert r.json()["results"]

    events = [d async for d in efficient_db[REQUEST_LOG].find({"module": "codebase_graph"})]
    assert len(events) == 1
    # snippets are a subset of the padded file: savings must be positive and
    # bounded by the whole-file token count
    assert 0 < events[0]["tokens_saved"] <= events[0]["baseline_tokens"]


async def test_query_no_results_records_nothing(client, efficient_db):
    r = await client.post("/codebase/query", json={"repo_id": "empty", "query": "x", "k": 5})
    assert r.status_code == 200
    events = [d async for d in efficient_db[REQUEST_LOG].find({"module": "codebase_graph"})]
    assert events == []


async def test_metrics_includes_store_section(client, efficient_db):
    await client.post("/codebase/index-file", json={
        "repo_id": "proj", "file_path": "m.py", "source": SOURCE,
    })
    r = await client.get("/metrics")
    body = r.json()
    assert "store" in body
    store = body["store"]
    assert store["codebase"]["symbols"] >= 2
    assert store["codebase"]["files"] == 1
    assert store["codebase"]["repos"] == 1
    for key in ("cache_entries", "memory", "corpus_chunks"):
        assert key in store
    assert set(store["memory"]) == {"working_sessions", "episodic", "semantic_facts"}
