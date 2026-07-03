import pytest
from unittest.mock import MagicMock
from finops.modules.agent_memory import AgentMemory
from finops.modules._base import OptimizeRequest
from finops.db.collections import WORKING_MEMORY, SEMANTIC_MEMORY

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.agent_memory.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.agent_memory.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="user prefers Python\nproject uses FastAPI")
    monkeypatch.setattr("finops.modules.agent_memory.ChatAnthropic", lambda **kw: fake_llm)


@pytest.fixture
def config():
    return {"working_memory_turns": 3, "episodic_ttl_days": 30, "semantic_ttl_days": 90}


@pytest.fixture
async def memory(finops_db, config):
    return AgentMemory(finops_db, config)


@pytest.fixture
def req():
    return OptimizeRequest(prompt="help with Python", context="original ctx", agent_id="a1", framework="test")


async def test_process_with_no_memory_returns_original_context(memory, req):
    new_req, result = await memory.process(req)
    assert new_req.context == "original ctx"
    assert result.module == "agent_memory"
    assert result.short_circuit is False


async def test_store_turn_writes_working_memory(memory, finops_db):
    await memory.store_turn("a1", "s1", "hello", "world")
    doc = await finops_db[WORKING_MEMORY].find_one({"agent_id": "a1", "session_id": "s1"})
    assert doc is not None
    assert len(doc["messages"]) == 2
    assert doc["messages"][0]["role"] == "user"
    assert doc["messages"][1]["role"] == "assistant"


async def test_store_turn_extracts_facts_to_semantic_memory(memory, finops_db):
    await memory.store_turn("a1", "s1", "I use Python", "Great choice")
    count = await finops_db[SEMANTIC_MEMORY].count_documents({"agent_id": "a1"})
    assert count >= 1


async def test_process_appends_working_memory_section(memory, finops_db, req):
    await memory.store_turn("a1", "s1", "first turn", "first response")
    new_req, result = await memory.process(req)
    assert new_req.context.startswith("original ctx")
    assert "## Memory" in new_req.context
    assert "first turn" in new_req.context or "first response" in new_req.context
    assert result.tokens_added > 0


async def test_working_memory_respects_turn_limit(memory, finops_db):
    for i in range(5):
        await memory.store_turn("a2", "s2", f"turn {i}", f"resp {i}")
    working = await memory._get_working_memory("a2")
    assert len(working) <= 6
