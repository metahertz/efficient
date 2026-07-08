import pytest
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.codebase_graph.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.codebase_graph.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_codebase_index_indexes_fixture_dir(client):
    resp = await client.post("/codebase/index", json={"repo_id": "r1", "path": "tests/fixtures"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "r1"
    assert data["indexed_files"] >= 1
    assert data["indexed_symbols"] > 0


async def test_codebase_query_returns_results_shape(client):
    await client.post("/codebase/index", json={"repo_id": "r2", "path": "tests/fixtures"})
    resp = await client.post("/codebase/query", json={"repo_id": "r2", "query": "add two numbers", "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "r2"
    assert isinstance(data["results"], list)
    for r in data["results"]:
        assert set(r.keys()) == {"symbol", "type", "file_path", "line_start", "line_end", "source_snippet"}


async def test_codebase_query_defaults_are_safe(client):
    resp = await client.post("/codebase/query", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "default"
    assert isinstance(data["results"], list)


@pytest.mark.parametrize("body,expected_repo", [
    ({"repo_id": "r3", "path": "/nope/does/not/exist"}, "r3"),
    ({"repo_id": "r4", "path": ""}, "r4"),
    ({"repo_id": "r5"}, "r5"),
])
async def test_codebase_index_missing_path_returns_zero(client, body, expected_repo):
    resp = await client.post("/codebase/index", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"repo_id": expected_repo, "indexed_files": 0, "indexed_symbols": 0}


async def test_codebase_index_file_endpoint(client, finops_db):
    from finops.db.collections import CODEBASE_NODES
    resp = await client.post("/codebase/index-file", json={
        "repo_id": "rif", "file_path": "x.py", "source": "def foo():\n    return 1\n"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "rif"
    assert data["file_path"] == "x.py"
    assert data["indexed_symbols"] >= 1
    resp2 = await client.post("/codebase/index-file", json={
        "repo_id": "rif", "file_path": "x.py", "source": "# nothing here\n"})
    assert resp2.status_code == 200
    assert resp2.json()["indexed_symbols"] == 0
    gone = await finops_db[CODEBASE_NODES].find_one(
        {"repo_id": "rif", "file_path": "x.py", "symbol": "foo"})
    assert gone is None


async def test_codebase_references_endpoint(client):
    await client.post("/codebase/index", json={"repo_id": "rref", "path": "tests/fixtures"})
    resp = await client.post("/codebase/references", json={"repo_id": "rref", "symbol": "helper"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "rref"
    assert data["symbol"] == "helper"
    caller_symbols = {c["symbol"] for c in data["callers"]}
    assert "main" in caller_symbols
    assert "run" in caller_symbols
    assert isinstance(data["callees"], list)
