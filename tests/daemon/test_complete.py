import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app
from finops.db.collections import CACHE_ENTRIES, WORKING_MEMORY


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.semantic_cache.embed_query", lambda t: [0.1] * 1024)
    monkeypatch.setattr("finops.modules.semantic_cache.embed_documents", lambda ts: [[0.1] * 1024] * len(ts))
    monkeypatch.setattr("finops.modules.agent_memory.embed_query", lambda t: [0.1] * 1024)
    monkeypatch.setattr("finops.modules.agent_memory.embed_documents", lambda ts: [[0.1] * 1024] * len(ts))


@pytest.fixture(autouse=True)
def mock_llm_and_facts(monkeypatch):
    monkeypatch.setattr(
        "finops.daemon.providers.call_llm",
        AsyncMock(return_value=("LLM answer", 100, 50)),
    )
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content="")
    monkeypatch.setattr("finops.modules.agent_memory.ChatAnthropic", lambda **kw: fake)


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _enable_only_cache_and_memory(finops_db):
    from finops.daemon.config import save_config
    await save_config(finops_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": True, "cache_key": "prompt+scope"},
        "agent_memory": {"enabled": True}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})


async def test_complete_miss_calls_llm_and_populates(client, finops_db):
    await _enable_only_cache_and_memory(finops_db)
    resp = await client.post("/complete", json={
        "prompt": "unique complete prompt", "agent_id": "u1", "session_id": "s1",
        "framework": "test", "provider": "anthropic", "model": "claude-x",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["cache_hit"] is False
    assert data["response"] == "LLM answer"
    assert data["input_tokens"] == 100
    assert data["output_tokens"] == 50
    assert await finops_db[CACHE_ENTRIES].count_documents({}) == 1
    assert await finops_db[WORKING_MEMORY].count_documents({"agent_id": "u1"}) == 1


async def test_second_identical_complete_hits_cache_and_skips_llm(client, finops_db, monkeypatch):
    await _enable_only_cache_and_memory(finops_db)
    body = {
        "prompt": "cache-me complete prompt", "agent_id": "u2", "session_id": "s2",
        "framework": "test", "provider": "anthropic", "model": "claude-x",
    }
    first = await client.post("/complete", json=body)
    assert first.json()["cache_hit"] is False

    spy = AsyncMock(return_value=("SHOULD NOT BE CALLED", 1, 1))
    monkeypatch.setattr("finops.daemon.providers.call_llm", spy)

    second = await client.post("/complete", json=body)
    data = second.json()
    assert data["cache_hit"] is True
    assert data["response"] == "LLM answer"
    spy.assert_not_called()
