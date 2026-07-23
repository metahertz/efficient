import httpx
import pytest
from httpx import ASGITransport

FIXED = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("efficient.daemon.memory_files.embed_documents",
                        lambda ts: [FIXED] * len(ts))


@pytest.fixture
async def client(efficient_db):
    from efficient.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _tool(client, **kwargs):
    r = await client.post("/memory/tool", json=kwargs)
    assert r.status_code == 200
    return r.json()


async def test_create_and_view_file(client):
    out = await _tool(client, command="create", path="/memories/notes.md",
                      file_text="alpha\nbeta\n")
    assert out["ok"] is True
    out = await _tool(client, command="view", path="/memories/notes.md")
    assert out["ok"] is True
    assert "1: alpha" in out["result"]
    assert "2: beta" in out["result"]


async def test_view_directory_lists_files(client):
    await _tool(client, command="create", path="/memories/a.md", file_text="x")
    await _tool(client, command="create", path="/memories/sub/b.md", file_text="y")
    out = await _tool(client, command="view", path="/memories")
    assert "/memories/a.md" in out["result"]
    assert "/memories/sub/b.md" in out["result"]


async def test_view_range(client):
    await _tool(client, command="create", path="/memories/n.md",
                file_text="l1\nl2\nl3\nl4\n")
    out = await _tool(client, command="view", path="/memories/n.md",
                      view_range=[2, 3])
    assert "2: l2" in out["result"] and "3: l3" in out["result"]
    assert "l1" not in out["result"] and "l4" not in out["result"]


async def test_create_overwrites(client):
    await _tool(client, command="create", path="/memories/o.md", file_text="old")
    await _tool(client, command="create", path="/memories/o.md", file_text="new")
    out = await _tool(client, command="view", path="/memories/o.md")
    assert "new" in out["result"] and "old" not in out["result"]


async def test_str_replace(client):
    await _tool(client, command="create", path="/memories/s.md",
                file_text="keep CHANGE keep")
    out = await _tool(client, command="str_replace", path="/memories/s.md",
                      old_str="CHANGE", new_str="changed")
    assert out["ok"] is True
    out = await _tool(client, command="view", path="/memories/s.md")
    assert "changed" in out["result"]


async def test_str_replace_requires_unique(client):
    await _tool(client, command="create", path="/memories/u.md", file_text="x x")
    out = await _tool(client, command="str_replace", path="/memories/u.md",
                      old_str="x", new_str="y")
    assert out["ok"] is False
    assert "once" in out["error"] or "unique" in out["error"].lower()


async def test_insert(client):
    await _tool(client, command="create", path="/memories/i.md", file_text="a\nc")
    out = await _tool(client, command="insert", path="/memories/i.md",
                      insert_line=1, insert_text="b")
    assert out["ok"] is True
    out = await _tool(client, command="view", path="/memories/i.md")
    assert "1: a" in out["result"] and "2: b" in out["result"] and "3: c" in out["result"]


async def test_delete_and_rename(client):
    await _tool(client, command="create", path="/memories/r1.md", file_text="z")
    out = await _tool(client, command="rename", old_path="/memories/r1.md",
                      new_path="/memories/r2.md")
    assert out["ok"] is True
    out = await _tool(client, command="view", path="/memories/r1.md")
    assert out["ok"] is False
    out = await _tool(client, command="delete", path="/memories/r2.md")
    assert out["ok"] is True
    out = await _tool(client, command="view", path="/memories/r2.md")
    assert out["ok"] is False


async def test_clear_all_scoped_to_agent(client):
    await _tool(client, command="create", path="/memories/m.md", file_text="a",
                agent_id="one")
    await _tool(client, command="create", path="/memories/m.md", file_text="b",
                agent_id="two")
    out = await _tool(client, command="clear_all", agent_id="one")
    assert out["ok"] is True
    out = await _tool(client, command="view", path="/memories/m.md", agent_id="one")
    assert out["ok"] is False
    out = await _tool(client, command="view", path="/memories/m.md", agent_id="two")
    assert out["ok"] is True


async def test_path_traversal_rejected(client):
    for bad in ("/etc/passwd", "/memories/../etc/passwd", "memories/x"):
        out = await _tool(client, command="create", path=bad, file_text="x")
        assert out["ok"] is False, bad


async def test_retrieve_includes_files(client, monkeypatch):
    async def fake_vs(collection, pipeline):
        cursor = collection.find({})
        return [{**d, "_score": 0.9} async for d in cursor]
    monkeypatch.setattr("efficient.daemon.app.vector_search", fake_vs)
    monkeypatch.setattr("efficient.modules.agent_memory.embed_query", lambda t: FIXED)
    monkeypatch.setattr("efficient.modules.embeddings.embed_query", lambda t: FIXED)

    await _tool(client, command="create", path="/memories/facts.md",
                file_text="the deploy password lives in vault")
    r = await client.post("/memory/retrieve",
                          json={"agent_id": "default", "query": "deploy"})
    body = r.json()
    assert "files" in body
    assert any(f["path"] == "/memories/facts.md" for f in body["files"])
