import httpx
import pytest
from httpx import ASGITransport

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("efficient.modules.codebase_graph.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("efficient.modules.codebase_graph.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
async def client(efficient_db):
    from efficient.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_config_rejects_operator_keys(client):
    r = await client.put("/config", json={"$set": {"strategy": "x"}})
    assert r.status_code == 422


async def test_config_rejects_dotted_keys(client):
    r = await client.put("/config", json={"modules.semantic_cache.enabled": False})
    assert r.status_code == 422


async def test_config_rejects_unknown_top_level_key(client):
    r = await client.put("/config", json={"evil": 1})
    assert r.status_code == 422


async def test_config_accepts_valid_patch(client):
    r = await client.put("/config", json={"strategy": "compose_then_compress"})
    assert r.status_code == 200


async def test_codebase_query_bounds_k(client):
    r = await client.post("/codebase/query", json={"repo_id": "r", "query": "q", "k": 10**9})
    assert r.status_code == 422
    r = await client.post("/codebase/query", json={"repo_id": "r", "query": "q", "k": "x"})
    assert r.status_code == 422


async def test_codebase_index_rejects_unlisted_path(client, tmp_path):
    r = await client.post("/codebase/index", json={"repo_id": "r", "path": str(tmp_path)})
    assert r.status_code == 403


async def test_codebase_index_allows_configured_root(client, tmp_path, monkeypatch):
    monkeypatch.setenv("EFFICIENT_ALLOWED_INDEX_ROOTS", str(tmp_path))
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    r = await client.post("/codebase/index", json={"repo_id": "r", "path": str(tmp_path)})
    assert r.status_code == 200
    assert r.json()["indexed_files"] == 1


async def test_optimize_rejects_non_string_prompt(client):
    r = await client.post("/optimize", json={"prompt": ["not", "a", "string"]})
    assert r.status_code == 422
