import pytest
import hashlib
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app
from finops.db.collections import CACHE_ENTRIES


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.semantic_cache.embed_documents", lambda ts: [[0.1] * 1024] * len(ts))
    monkeypatch.setattr("finops.modules.semantic_cache.embed_query", lambda t: [0.1] * 1024)


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_cache_lookup_miss(client):
    resp = await client.get("/cache/lookup", params={"prompt_hash": "abc123"})
    assert resp.status_code == 200
    assert resp.json()["hit"] is False


async def test_cache_lookup_exact_hit(client, finops_db):
    prompt = "hello world"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    await finops_db[CACHE_ENTRIES].insert_one({
        "prompt_hash": prompt_hash, "embedding": [0.1] * 1024, "prompt_preview": prompt,
        "response": "hi there", "framework": "test", "model": "claude", "tokens_saved": 100,
        "hit_count": 0, "created_at": datetime.now(timezone.utc), "last_hit_at": None, "expires_at": None,
    })
    resp = await client.get("/cache/lookup", params={"prompt_hash": prompt_hash})
    data = resp.json()
    assert data["hit"] is True
    assert data["response"] == "hi there"


async def test_cache_lookup_exact_hit_increments_hit_count(client, finops_db):
    prompt = "increment me"
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    await finops_db[CACHE_ENTRIES].insert_one({
        "prompt_hash": prompt_hash, "embedding": [0.1] * 1024, "prompt_preview": prompt,
        "response": "incremented", "framework": "test", "model": "claude", "tokens_saved": 50,
        "hit_count": 0, "created_at": datetime.now(timezone.utc), "last_hit_at": None, "expires_at": None,
    })
    resp = await client.get("/cache/lookup", params={"prompt_hash": prompt_hash})
    assert resp.json()["hit"] is True
    doc = await finops_db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    assert doc["hit_count"] == 1
    assert doc["last_hit_at"] is not None


async def test_cache_store_endpoint_writes(client, finops_db):
    resp = await client.post("/cache/store", json={
        "prompt": "cache me", "response": "cached answer", "framework": "test",
        "model": "claude", "tokens_saved": 321, "agent_id": "a1", "corpus_id": "c1",
    })
    assert resp.status_code == 200
    assert resp.json()["stored"] is True
    count = await finops_db[CACHE_ENTRIES].count_documents({})
    assert count == 1
    doc = await finops_db[CACHE_ENTRIES].find_one({})
    assert doc["response"] == "cached answer"
    assert doc["tokens_saved"] == 321
