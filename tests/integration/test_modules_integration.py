import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from finops.modules.semantic_cache import SemanticCache
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules.hybrid_retrieval import HybridRetrieval
from finops.modules.agent_memory import AgentMemory
from finops.modules._base import OptimizeRequest
from finops.db.collections import (
    CACHE_ENTRIES, CODEBASE_NODES, CORPUS_CHUNKS,
    EPISODIC_MEMORY, SEMANTIC_MEMORY,
)
from finops.daemon.app import app
from tests.conftest import wait_for_queryable

pytestmark = pytest.mark.integration

_SAMPLE_SOURCE = Path(__file__).parent.parent / "fixtures" / "sample.py"


async def test_semantic_cache_paraphrase_hit(finops_db, sync_db):
    cache = SemanticCache(finops_db, {"similarity_threshold": 0.55, "cache_key": "prompt"})
    await cache.store(
        prompt="How do I reverse a list in Python?",
        response="Use lst[::-1] or reversed(lst).",
        framework="test",
        model="claude",
        tokens_saved=300,
    )
    wait_for_queryable(sync_db[CACHE_ENTRIES], "cache_vector_index")
    req = OptimizeRequest(
        prompt="What is the way to reverse a Python list?",
        context="orig",
        agent_id="a1",
        framework="test",
    )
    new_req, result = await cache.process(req)
    print(result.detail)
    assert result.short_circuit is True
    assert "reversed" in new_req.context or "[::-1]" in new_req.context


async def test_codebase_graph_symbol_recall(finops_db, sync_db):
    graph = CodebaseGraph(finops_db, {"repo_paths": []})
    source = _SAMPLE_SOURCE.read_text(encoding="utf-8")
    n = await graph.index_file("repoI", "sample.py", source)
    assert n >= 2
    wait_for_queryable(sync_db[CODEBASE_NODES], "codebase_vector_index")
    results = await graph.query("repoI", "function that adds two numbers", k=5)
    print([r["symbol"] for r in results])
    symbols = [r["symbol"] for r in results]
    assert "add" in symbols


async def test_hybrid_retrieval_top_ranked(finops_db, sync_db):
    retrieval = HybridRetrieval(finops_db, {"top_k": 2, "rrf_k": 60})
    await retrieval.add_chunks("corpI", [
        {"text": "MongoDB is a document-oriented NoSQL database.",
         "source_file": "d.txt", "chunk_index": 0, "metadata": {}},
        {"text": "The Eiffel Tower is located in Paris, France.",
         "source_file": "d.txt", "chunk_index": 1, "metadata": {}},
        {"text": "Photosynthesis converts sunlight into chemical energy.",
         "source_file": "d.txt", "chunk_index": 2, "metadata": {}},
    ])
    wait_for_queryable(sync_db[CORPUS_CHUNKS], "corpus_vector_index")
    req = OptimizeRequest(
        prompt="what kind of database is MongoDB?",
        context="",
        agent_id="a1",
        framework="test",
        corpus_id="corpI",
    )
    new_req, result = await retrieval.process(req)
    print(result.detail)
    assert "## Retrieved Docs" in new_req.context
    assert "document-oriented" in new_req.context


async def test_agent_memory_recall(finops_db, sync_db, monkeypatch):
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content="")
    monkeypatch.setattr("finops.modules.agent_memory.ChatAnthropic", lambda **kw: fake)
    memory = AgentMemory(finops_db, {"working_memory_turns": 20})
    await memory.store_turn("agentI", "sess", "My favorite language is Rust.", "Noted, Rust it is.")
    await memory.store_turn("agentI", "sess", "I also enjoy hiking on weekends.", "Great hobby.")

    working = await memory._get_working_memory("agentI")
    assert any("Rust" in m["content"] for m in working)

    wait_for_queryable(sync_db[EPISODIC_MEMORY], "episodic_vector_index")
    wait_for_queryable(sync_db[SEMANTIC_MEMORY], "semantic_vector_index")

    req = OptimizeRequest(
        prompt="what language do I like?",
        context="ctx",
        agent_id="agentI",
        framework="test",
    )
    new_req, result = await memory.process(req)
    print(result.detail)
    assert "## Memory" in new_req.context
    assert "Rust" in new_req.context


async def test_end_to_end_complete_real_embeddings(finops_db, sync_db, monkeypatch):
    from finops.daemon.config import save_config
    await save_config(finops_db, {"modules": {
        "codebase_graph": {"enabled": False},
        "semantic_cache": {"enabled": True, "similarity_threshold": 0.55},
        "agent_memory": {"enabled": False},
        "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    monkeypatch.setattr(
        "finops.daemon.providers.call_llm",
        AsyncMock(return_value=("real answer", 80, 40)),
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        body = {
            "prompt": "Explain what a hash map is.",
            "agent_id": "e2e", "session_id": "s",
            "framework": "test", "provider": "anthropic", "model": "claude-x",
        }
        first = await c.post("/complete", json=body)
        assert first.json()["cache_hit"] is False
        assert first.json()["response"] == "real answer"

        wait_for_queryable(sync_db[CACHE_ENTRIES], "cache_vector_index")

        spy = AsyncMock(return_value=("SHOULD NOT CALL", 1, 1))
        monkeypatch.setattr("finops.daemon.providers.call_llm", spy)
        second = await c.post("/complete", json=body)
        data = second.json()
        assert data["cache_hit"] is True
        assert data["response"] == "real answer"
        spy.assert_not_called()
