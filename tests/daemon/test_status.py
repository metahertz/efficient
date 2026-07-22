import httpx
import pytest
from httpx import ASGITransport

from efficient import activity


@pytest.fixture(autouse=True)
def fresh_feed():
    activity.reset()
    yield
    activity.reset()


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


def test_emit_and_snapshot_since():
    s1 = activity.emit("one")
    s2 = activity.emit("two", notify=True)
    snap = activity.snapshot()
    assert [e["message"] for e in snap["events"]] == ["one", "two"]
    assert snap["last_seq"] == s2
    later = activity.snapshot(since=s1)
    assert [e["message"] for e in later["events"]] == ["two"]
    assert later["events"][0]["notify"] is True


def test_activity_context_manager_tracks_inflight():
    with activity.activity("loading model"):
        snap = activity.snapshot()
        assert any(a["message"] == "loading model" for a in snap["active"])
        assert "elapsed_s" in snap["active"][0]
    snap = activity.snapshot()
    assert snap["active"] == []
    messages = [e["message"] for e in snap["events"]]
    assert messages[0].startswith("loading model")
    assert any("done" in m for m in messages)


def test_activity_failure_emits_error():
    with pytest.raises(ValueError):
        with activity.activity("doomed"):
            raise ValueError("boom")
    snap = activity.snapshot()
    assert any(e["level"] == "error" and "failed" in e["message"] for e in snap["events"])
    assert snap["active"] == []


def test_index_batch_coalesces_and_closes():
    activity.note_indexed("a.py")
    activity.note_indexed("b.py")
    snap = activity.snapshot()
    # one start event, not one per file; batch visible as in-flight
    starts = [e for e in snap["events"] if "indexing codebase" in e["message"]]
    assert len(starts) == 1
    assert any("2 files" in a["message"] for a in snap["active"])
    # force the batch idle window to elapse
    activity._batch["last_ts"] -= 999
    snap = activity.snapshot()
    assert any("indexed 2 files" in e["message"] and e["notify"] for e in snap["events"])
    assert not any("indexing codebase" in a["message"] for a in snap["active"])


async def test_status_endpoint_shape(client):
    activity.emit("hello", notify=True)
    r = await client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"active", "events", "last_seq"}
    assert any(e["message"] == "hello" for e in body["events"])
    since = body["last_seq"]
    r2 = await client.get(f"/status?since={since}")
    assert r2.json()["events"] == []


async def test_status_exempt_from_auth(client, monkeypatch):
    monkeypatch.setenv("EFFICIENT_API_TOKEN", "sekret")
    r = await client.get("/status")
    assert r.status_code == 200


async def test_index_file_endpoint_feeds_batch(client):
    await client.post("/codebase/index-file", json={
        "repo_id": "r", "file_path": "m.py", "source": "def f():\n    return 1\n",
    })
    snap = activity.snapshot()
    assert any("indexing codebase" in e["message"] for e in snap["events"])
