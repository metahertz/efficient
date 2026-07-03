import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.agent_memory.embed_query", lambda t: [0.1] * 1024)
    monkeypatch.setattr("finops.modules.agent_memory.embed_documents", lambda ts: [[0.1] * 1024] * len(ts))


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content="")
    monkeypatch.setattr("finops.modules.agent_memory.ChatAnthropic", lambda **kw: fake)


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_memory_retrieve_empty(client):
    resp = await client.post("/memory/retrieve", json={"agent_id": "x", "query": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["working"] == []
    assert data["episodic"] == []
    assert data["semantic"] == []


async def test_memory_store_and_retrieve(client):
    store_resp = await client.post("/memory/store", json={
        "agent_id": "u1", "session_id": "s1", "turn": "Hello", "response": "Hi there"
    })
    assert store_resp.status_code == 200
    assert store_resp.json()["stored"] is True
    ret_resp = await client.post("/memory/retrieve", json={"agent_id": "u1", "query": "Hello"})
    data = ret_resp.json()
    assert len(data["working"]) == 2
    assert data["working"][0]["role"] == "user"
