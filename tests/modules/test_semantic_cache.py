import hashlib
import pytest
from datetime import datetime, timezone
from efficient.modules.semantic_cache import SemanticCache
from efficient.modules._base import OptimizeRequest
from efficient.db.collections import CACHE_ENTRIES

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("efficient.modules.semantic_cache.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("efficient.modules.semantic_cache.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
def config():
    return {"similarity_threshold": 0.92, "ttl_hours": 168, "cache_key": "prompt+scope"}


@pytest.fixture
async def cache(efficient_db, config):
    return SemanticCache(efficient_db, config)


@pytest.fixture
def req():
    return OptimizeRequest(prompt="what is Python?", context="ctx", agent_id="a1",
                           framework="test", corpus_id="c1")


async def test_cache_miss_returns_unchanged_request(cache, req):
    new_req, result = await cache.process(req)
    assert new_req.context == "ctx"
    assert result.tokens_saved == 0
    assert result.short_circuit is False
    assert "miss" in result.detail


async def test_exact_hit_replaces_context_and_short_circuits(cache, efficient_db, req):
    key = cache._key_material(req)
    prompt_hash = hashlib.sha256(key.encode()).hexdigest()
    await efficient_db[CACHE_ENTRIES].insert_one({
        "prompt_hash": prompt_hash,
        "embedding": FIXED_EMBEDDING,
        "prompt_preview": req.prompt[:200],
        "response": "Python is a language",
        "framework": "test",
        "model": "claude",
        "tokens_saved": 500,
        "hit_count": 0,
        "created_at": datetime.now(timezone.utc),
        "last_hit_at": None,
        "expires_at": None,
    })
    new_req, result = await cache.process(req)
    assert new_req.context == "Python is a language"
    assert result.tokens_saved == 500
    assert result.short_circuit is True
    assert "exact" in result.detail
    doc = await efficient_db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    assert doc["hit_count"] == 1


async def test_store_writes_entry(cache, efficient_db, req):
    await cache.store(prompt=req.prompt, response="Python is great", framework="test",
                      model="claude", tokens_saved=200, agent_id=req.agent_id, corpus_id=req.corpus_id)
    key = cache._key_material(req)
    prompt_hash = hashlib.sha256(key.encode()).hexdigest()
    doc = await efficient_db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    assert doc is not None
    assert doc["response"] == "Python is great"
    assert doc["tokens_saved"] == 200
    assert doc["hit_count"] == 0


async def test_store_is_idempotent(cache, efficient_db, req):
    await cache.store(req.prompt, "resp", "test", "m", 100, agent_id=req.agent_id, corpus_id=req.corpus_id)
    await cache.store(req.prompt, "resp", "test", "m", 100, agent_id=req.agent_id, corpus_id=req.corpus_id)
    key = cache._key_material(req)
    prompt_hash = hashlib.sha256(key.encode()).hexdigest()
    count = await efficient_db[CACHE_ENTRIES].count_documents({"prompt_hash": prompt_hash})
    assert count == 1


async def test_prompt_plus_scope_differs_by_corpus(efficient_db):
    cache = SemanticCache(efficient_db, {"cache_key": "prompt+scope"})
    r1 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c1")
    r2 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c2")
    assert cache._key_material(r1) != cache._key_material(r2)


async def test_prompt_only_ignores_corpus(efficient_db):
    cache = SemanticCache(efficient_db, {"cache_key": "prompt"})
    r1 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c1")
    r2 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c2")
    assert cache._key_material(r1) == cache._key_material(r2)
