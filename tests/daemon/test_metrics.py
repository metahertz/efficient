import pytest
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app
from finops.db.collections import CACHE_ENTRIES, COMPRESSION_STATS, REQUEST_LOG
from finops.daemon.metrics import record_module_events


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_metrics_empty_collections(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_tokens_saved"] == 0
    assert data["cache_hit_rate"] == 0.0
    assert data["compression_ratio"] == 0.0
    assert isinstance(data["per_module"], list)
    assert data["per_module"] == []


async def test_metrics_with_cache_entries(client, finops_db):
    now = datetime.now(timezone.utc)
    await finops_db[CACHE_ENTRIES].insert_many([
        {"prompt_hash": "h1", "embedding": [], "prompt_preview": "", "response": "",
         "framework": "test", "model": "m", "tokens_saved": 400, "hit_count": 3,
         "created_at": now, "last_hit_at": now, "expires_at": None},
        {"prompt_hash": "h2", "embedding": [], "prompt_preview": "", "response": "",
         "framework": "test", "model": "m", "tokens_saved": 200, "hit_count": 0,
         "created_at": now, "last_hit_at": None, "expires_at": None},
    ])
    resp = await client.get("/metrics")
    data = resp.json()
    assert data["total_tokens_saved"] == 1200
    assert abs(data["cache_hit_rate"] - 0.5) < 0.01
    cache_entry = next((m for m in data["per_module"] if m["module"] == "semantic_cache"), None)
    assert cache_entry is not None
    assert cache_entry["tokens_saved"] == 1200


async def test_metrics_with_compression_stats(client, finops_db):
    now = datetime.now(timezone.utc)
    await finops_db[COMPRESSION_STATS].insert_many([
        {"request_id": "r1", "framework": "test", "model": "",
         "original_tokens": 1000, "compressed_tokens": 250, "ratio": 4.0,
         "latency_ms": 120.0, "created_at": now},
        {"request_id": "r2", "framework": "test", "model": "",
         "original_tokens": 800, "compressed_tokens": 200, "ratio": 4.0,
         "latency_ms": 90.0, "created_at": now},
    ])
    resp = await client.get("/metrics")
    data = resp.json()
    assert data["compression_ratio"] == pytest.approx(4.0, rel=0.01)
    comp_entry = next((m for m in data["per_module"] if m["module"] == "context_compressor"), None)
    assert comp_entry is not None
    assert comp_entry["tokens_saved"] == 1350


async def test_metrics_combined_cache_and_compression(client, finops_db):
    now = datetime.now(timezone.utc)
    # Cache: one entry with tokens_saved=400, hit_count=3 → 1200 saved; hit_rate = 1/1 = 1.0
    await finops_db[CACHE_ENTRIES].insert_many([
        {"prompt_hash": "h1", "embedding": [], "prompt_preview": "", "response": "",
         "framework": "test", "model": "m", "tokens_saved": 400, "hit_count": 3,
         "created_at": now, "last_hit_at": now, "expires_at": None},
    ])
    # Compression: two entries → (1000-250) + (800-200) = 750 + 600 = 1350; avg_ratio = 4.0
    await finops_db[COMPRESSION_STATS].insert_many([
        {"request_id": "r1", "framework": "test", "model": "",
         "original_tokens": 1000, "compressed_tokens": 250, "ratio": 4.0,
         "latency_ms": 120.0, "created_at": now},
        {"request_id": "r2", "framework": "test", "model": "",
         "original_tokens": 800, "compressed_tokens": 200, "ratio": 4.0,
         "latency_ms": 90.0, "created_at": now},
    ])
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()

    # Total must be the exact sum of both modules
    assert data["total_tokens_saved"] == 2550  # 1200 + 1350

    # Compression ratio is the average of both entries (both 4.0)
    assert data["compression_ratio"] == pytest.approx(4.0, rel=0.01)

    # One hit entry out of one total cache entry
    assert data["cache_hit_rate"] == pytest.approx(1.0, rel=0.01)

    # per_module must contain exactly both modules
    modules = {m["module"]: m for m in data["per_module"]}
    assert "semantic_cache" in modules
    assert modules["semantic_cache"]["tokens_saved"] == 1200
    assert "context_compressor" in modules
    assert modules["context_compressor"]["tokens_saved"] == 1350


async def test_record_module_events_inserts(finops_db):
    await record_module_events(finops_db, [
        {"module": "codebase_graph", "tokens_saved": 100, "tokens_added": 20,
         "baseline_tokens": 120, "latency_ms": 1.0},
        {"module": "agent_memory", "tokens_saved": 0, "tokens_added": 50,
         "baseline_tokens": 50, "latency_ms": 1.0},
    ])
    count = await finops_db[REQUEST_LOG].count_documents({})
    assert count == 2

    # Empty list inserts nothing.
    await record_module_events(finops_db, [])
    count = await finops_db[REQUEST_LOG].count_documents({})
    assert count == 2


async def test_metrics_includes_augmenter_modules(client, finops_db):
    now = datetime.now(timezone.utc)
    await finops_db[REQUEST_LOG].insert_many([
        {"module": "codebase_graph", "tokens_saved": 100, "tokens_added": 0,
         "baseline_tokens": 100, "latency_ms": 1.0, "created_at": now},
        {"module": "codebase_graph", "tokens_saved": 100, "tokens_added": 0,
         "baseline_tokens": 100, "latency_ms": 1.0, "created_at": now},
        {"module": "hybrid_retrieval", "tokens_saved": 30, "tokens_added": 0,
         "baseline_tokens": 30, "latency_ms": 1.0, "created_at": now},
    ])
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()

    modules = {m["module"]: m for m in data["per_module"]}
    assert "codebase_graph" in modules
    assert modules["codebase_graph"]["tokens_saved"] == 200
    assert modules["codebase_graph"]["events"] == 2
    assert "hybrid_retrieval" in modules
    assert modules["hybrid_retrieval"]["tokens_saved"] == 30
    assert modules["hybrid_retrieval"]["events"] == 1

    # total includes the augmenter savings (no cache/compression seeded)
    assert data["total_tokens_saved"] == 230


async def test_metrics_augmenters_and_cache_combined(client, finops_db):
    now = datetime.now(timezone.utc)
    # Cache: tokens_saved=400, hit_count=3 → 1200 saved
    await finops_db[CACHE_ENTRIES].insert_many([
        {"prompt_hash": "h1", "embedding": [], "prompt_preview": "", "response": "",
         "framework": "test", "model": "m", "tokens_saved": 400, "hit_count": 3,
         "created_at": now, "last_hit_at": now, "expires_at": None},
    ])
    # Compression: (1000-250) + (800-200) = 1350
    await finops_db[COMPRESSION_STATS].insert_many([
        {"request_id": "r1", "framework": "test", "model": "",
         "original_tokens": 1000, "compressed_tokens": 250, "ratio": 4.0,
         "latency_ms": 120.0, "created_at": now},
        {"request_id": "r2", "framework": "test", "model": "",
         "original_tokens": 800, "compressed_tokens": 200, "ratio": 4.0,
         "latency_ms": 90.0, "created_at": now},
    ])
    # Augmenters via request_log: codebase_graph 100+100=200, agent_memory 50
    await finops_db[REQUEST_LOG].insert_many([
        {"module": "codebase_graph", "tokens_saved": 100, "tokens_added": 0,
         "baseline_tokens": 100, "latency_ms": 1.0, "created_at": now},
        {"module": "codebase_graph", "tokens_saved": 100, "tokens_added": 0,
         "baseline_tokens": 100, "latency_ms": 1.0, "created_at": now},
        {"module": "agent_memory", "tokens_saved": 50, "tokens_added": 0,
         "baseline_tokens": 50, "latency_ms": 1.0, "created_at": now},
    ])
    resp = await client.get("/metrics")
    data = resp.json()

    modules = {m["module"]: m for m in data["per_module"]}
    assert "semantic_cache" in modules
    assert modules["semantic_cache"]["tokens_saved"] == 1200
    assert "context_compressor" in modules
    assert modules["context_compressor"]["tokens_saved"] == 1350
    assert "codebase_graph" in modules
    assert modules["codebase_graph"]["tokens_saved"] == 200
    assert modules["codebase_graph"]["events"] == 2
    assert "agent_memory" in modules
    assert modules["agent_memory"]["tokens_saved"] == 50
    assert modules["agent_memory"]["events"] == 1

    # total = cache(1200) + compression(1350) + augmenters(250), no double count
    assert data["total_tokens_saved"] == 2800
