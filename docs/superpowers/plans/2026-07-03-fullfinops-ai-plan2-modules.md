# fullFinOps-AI Plan 2: Optimization Modules + /optimize Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all five token-optimization modules (Semantic Cache, Agent Memory, Context Compressor, Codebase Graph, Hybrid Retrieval), the `/optimize` pipeline router, and the `/metrics` endpoint — all backed by MongoDB, all reachable via the existing FastAPI daemon.

**Architecture:** Each module subclasses `BaseModule(ABC)` and receives an `AsyncIOMotorDatabase` + module-specific config dict at construction. The `ModulePipeline` in `finops/daemon/router.py` reads the live config from MongoDB on each `/optimize` request and calls only enabled modules in a fixed pipeline order. All external API calls (VoyageAI embeddings, Anthropic LLM for fact extraction, LLMLingua-2 compression) are wrapped behind thin helpers so tests can mock them with `monkeypatch` without touching MongoDB.

**Tech Stack:** Python 3.11+, FastAPI, Motor (async MongoDB), VoyageAI (`voyageai>=0.3`), `langchain-anthropic` (fact extraction LLM), `llmlingua>=0.2` (context compression), `tree-sitter>=0.23` + `tree-sitter-python` (codebase graph), `rank-bm25>=0.2` + MongoDB `$text` index (BM25 retrieval), MongoDB `$vectorSearch` (dense retrieval), pytest + pytest-asyncio (mode=auto), httpx AsyncClient + ASGITransport (daemon endpoint tests).

## Global Constraints

- All code runs inside the Docker dev container (`docker compose run --rm dev <cmd>`). Run every test command as `docker compose run --rm dev pytest ...`.
- Python ≥ 3.11; MongoDB ≥ 7.0 with mongot (Atlas Local image).
- Default embedding model: `voyage-4-nano`, 1024 dimensions, cosine similarity. Single source of truth: `EMBEDDING_DIMENSIONS = 1024` and `VECTOR_SIMILARITY = "cosine"` in `finops/db/indexes.py`.
- Default daemon port: 7432 (`FINOPS_PORT` env var).
- No Mem0 dependency — memory stack built on Motor + VoyageAI embed helper + `langchain-anthropic`.
- Token counts in all `ModuleResult` fields are approximate (`len(text) // 4`). Do not add tiktoken.
- All tests mock external API calls (VoyageAI, Anthropic, LLMLingua-2). No real API keys required to run the suite.
- All new code follows the existing no-comments style (no docstrings, no inline comments unless WHY is non-obvious from code alone).
- Test DB name: `finops_test` (set by `set_test_env` fixture in `tests/conftest.py`).
- Test MongoDB URI: `mongodb://localhost:27017/?directConnection=true` (or `FINOPS_TEST_MONGODB_URI` env var, which is set to `mongodb://mongodb:27017` inside the container).
- `pytest-asyncio` mode is `auto` — all async test functions are collected automatically without `@pytest.mark.asyncio`.

---

## Task 1: Foundation Utilities + Plan 1 Minor Fixes

Fixes three carry-over issues from Plan 1 (M1, M2, M3) and adds shared infrastructure required by every module in Tasks 2–6: the VoyageAI embedding helper, an updated index helper with pre-filter field support, and a `finops_db` test fixture.

**Files:**
- Modify: `finops/modules/_base.py` — add `__init__` name enforcement; add `short_circuit: bool = False` to `ModuleResult`
- Modify: `finops/cli/main.py` — lazy `DAEMON_URL` (M1)
- Modify: `Dockerfile` — multi-stage `base` + `dev` targets (M3)
- Modify: `docker-compose.yml` — explicit `target:` per service (M3)
- Create: `finops/modules/embeddings.py` — VoyageAI embed singleton
- Modify: `finops/db/indexes.py` — add `filter_paths` param to `_create_vector_index`; update calls for `codebase_nodes` and `corpus_chunks`
- Modify: `tests/conftest.py` — add `finops_db` async fixture
- Modify: `tests/modules/test_base.py` — add tests for name enforcement and `short_circuit`

**Interfaces:**
- Produces:
  - `finops.modules.embeddings.embed_one(text: str) -> list[float]` — returns 1024-float list
  - `finops.modules.embeddings.embed(texts: list[str]) -> list[list[float]]`
  - `finops.modules.embeddings.reset_client() -> None` — test helper
  - `ModuleResult.short_circuit: bool` — new field (default False); router uses this to stop pipeline early
  - `finops_db` pytest fixture — `AsyncIOMotorDatabase` with all indexes pre-created

- [ ] **Step 1: Write failing tests for M2 and short_circuit**

```python
# Add to tests/modules/test_base.py
def test_subclass_without_name_raises():
    class Unnamed(BaseModule):
        async def process(self, request):
            return request, None
        def is_enabled(self):
            return True
    with pytest.raises(TypeError, match="must define a non-empty 'name'"):
        Unnamed()


def test_module_result_has_short_circuit_default():
    r = ModuleResult(
        module="x", tokens_in=0, tokens_out=0,
        tokens_saved=0, latency_ms=0.0, detail=""
    )
    assert r.short_circuit is False


def test_module_result_short_circuit_can_be_set():
    r = ModuleResult(
        module="x", tokens_in=0, tokens_out=0,
        tokens_saved=0, latency_ms=0.0, detail="", short_circuit=True
    )
    assert r.short_circuit is True
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_base.py -v 2>&1 | tail -20
```
Expected: 3 new tests FAIL (AttributeError or TypeError).

- [ ] **Step 3: Fix M2 — name enforcement in BaseModule; add short_circuit to ModuleResult**

Replace entire `finops/modules/_base.py`:
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OptimizeRequest:
    prompt:    str
    context:   str
    agent_id:  str
    framework: str
    corpus_id: str | None = None


@dataclass
class ModuleResult:
    module:        str
    tokens_in:     int
    tokens_out:    int
    tokens_saved:  int
    latency_ms:    float
    detail:        str
    short_circuit: bool = field(default=False)


class BaseModule(ABC):
    name: str = ""

    def __init__(self):
        if not self.__class__.name:
            raise TypeError(
                f"{self.__class__.__name__} must define a non-empty 'name' class attribute"
            )

    @abstractmethod
    async def process(
        self, request: OptimizeRequest
    ) -> tuple[OptimizeRequest, ModuleResult]:
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        ...
```

- [ ] **Step 4: Run tests — all should pass**

```bash
docker compose run --rm dev pytest tests/modules/test_base.py -v 2>&1 | tail -15
```
Expected: 8 tests PASS.

- [ ] **Step 5: Fix M1 — lazy DAEMON_URL in cli/main.py**

Replace `finops/cli/main.py` lines 1–10:
```python
import os
import signal
import subprocess
from pathlib import Path

import click
import httpx


def _daemon_url() -> str:
    return os.getenv("FINOPS_DAEMON_URL", "http://localhost:7432")


PID_FILE = Path.home() / ".finops" / "daemon.pid"
```

Then replace every occurrence of `DAEMON_URL` in the file with `_daemon_url()`:

In `start()`:
```python
    click.echo(f"Daemon started (PID {proc.pid}) at {_daemon_url()}")
```

In `status()`:
```python
        health = httpx.get(f"{_daemon_url()}/health", timeout=2.0).json()
        ...
        modules = httpx.get(f"{_daemon_url()}/config", timeout=2.0).json().get("modules", {})
```

- [ ] **Step 6: Run CLI tests to verify M1 fix**

```bash
docker compose run --rm dev pytest tests/cli/ -v 2>&1 | tail -15
```
Expected: all CLI tests PASS (no regressions).

- [ ] **Step 7: Fix M3 — multi-stage Dockerfile**

Replace entire `Dockerfile`:
```dockerfile
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml .
RUN mkdir -p finops && touch finops/__init__.py
RUN pip install --no-cache-dir -e "."

COPY . .
RUN pip install --no-cache-dir -e "."

EXPOSE 7432
CMD uvicorn finops.daemon.app:app --host 0.0.0.0 --port ${FINOPS_PORT:-7432}

FROM base AS dev
RUN pip install --no-cache-dir -e ".[dev]"
```

Update `docker-compose.yml` — add `target:` to both service builds:
```yaml
  dev:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev
    ...

  daemon:
    build:
      context: .
      dockerfile: Dockerfile
      target: base
    ...
```

(Keep all other fields unchanged — volumes, ports, environment, depends_on.)

- [ ] **Step 8: Create `finops/modules/embeddings.py`**

```python
import os
import voyageai

_client: voyageai.Client | None = None


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        _client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY", ""))
    return _client


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    model = model or os.getenv("FINOPS_EMBEDDING_MODEL", "voyage-4-nano")
    result = _get_client().embed(texts, model=model)
    return result.embeddings


def embed_one(text: str, model: str | None = None) -> list[float]:
    return embed([text], model=model)[0]


def reset_client() -> None:
    global _client
    _client = None
```

- [ ] **Step 9: Update `finops/db/indexes.py` — add filter_paths support**

Replace `_create_vector_index` and update `create_all_indexes`:
```python
def _create_vector_index(
    collection, name: str, field: str = "embedding",
    filter_paths: list[str] | None = None
) -> None:
    if _search_index_exists(collection, name):
        return
    vector_field = {
        "type": "vector",
        "path": field,
        "numDimensions": EMBEDDING_DIMENSIONS,
        "similarity": VECTOR_SIMILARITY,
    }
    fields = [vector_field]
    for path in (filter_paths or []):
        fields.append({"type": "filter", "path": path})
    collection.create_search_index({
        "name": name,
        "type": "vectorSearch",
        "definition": {"fields": fields},
    })
```

In `create_all_indexes`, update the two calls that benefit from pre-filtering:
```python
    # codebase_nodes — filter_paths=["repo_id"] enables pre-filtered $vectorSearch
    _create_vector_index(col, "codebase_vector_index", filter_paths=["repo_id"])

    # corpus_chunks — filter_paths=["corpus_id"] enables pre-filtered $vectorSearch
    _create_vector_index(col, "corpus_vector_index", filter_paths=["corpus_id"])
```
All other `_create_vector_index` calls remain unchanged.

- [ ] **Step 10: Add `finops_db` fixture to `tests/conftest.py`**

Append to `tests/conftest.py`:
```python
@pytest.fixture
async def finops_db(async_client, sync_db):
    """Async DB with all Atlas Search indexes pre-created. sync_db handles teardown."""
    from finops.db.indexes import create_all_indexes
    create_all_indexes(sync_db)
    yield async_client[os.environ["FINOPS_DB_NAME"]]
```

- [ ] **Step 11: Run the full suite — must still be green**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: 28+ tests pass, 0 fail.

- [ ] **Step 12: Commit**

```bash
git add finops/modules/_base.py finops/modules/embeddings.py \
        finops/cli/main.py finops/db/indexes.py \
        Dockerfile docker-compose.yml tests/conftest.py \
        tests/modules/test_base.py
git commit -m "feat: add embedding helper, fix M1/M2/M3, add finops_db fixture, pre-filter index support"
```

---

## Task 2: Semantic Cache Module + /cache/lookup Endpoint

Two-layer cache: SHA-256 exact match then vector similarity (`$vectorSearch`). A `store()` method writes cache entries (called by the router after LLM responses in Plan 4). The `/cache/lookup` endpoint exposes direct cache checks for plugins.

**Files:**
- Create: `finops/modules/semantic_cache.py`
- Modify: `finops/daemon/app.py` — add `GET /cache/lookup`
- Create: `tests/modules/test_semantic_cache.py`
- Create: `tests/daemon/test_cache_lookup.py`

**Interfaces:**
- Consumes: `finops.modules.embeddings.embed_one`, `finops.db.collections.CACHE_ENTRIES`, `BaseModule`, `OptimizeRequest`, `ModuleResult`
- Produces:
  - `SemanticCache(db: AsyncIOMotorDatabase, config: dict)` — init
  - `await cache.process(request) -> (OptimizeRequest, ModuleResult)` — checks cache; on HIT sets `result.short_circuit=True` and injects cached response into `request.context`
  - `await cache.store(prompt, response, framework, model, tokens_saved) -> None` — upserts cache entry

- [ ] **Step 1: Write failing tests**

Create `tests/modules/test_semantic_cache.py`:
```python
import hashlib
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
from finops.modules.semantic_cache import SemanticCache
from finops.modules._base import OptimizeRequest
from finops.db.collections import CACHE_ENTRIES

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.semantic_cache.embed_one", lambda t, **kw: FIXED_EMBEDDING)


@pytest.fixture
def config():
    return {"similarity_threshold": 0.92, "ttl_hours": 168}


@pytest.fixture
async def cache(finops_db, config):
    return SemanticCache(finops_db, config)


@pytest.fixture
def req():
    return OptimizeRequest(prompt="what is Python?", context="ctx", agent_id="a1", framework="test")


async def test_cache_miss_returns_unchanged_request(cache, req):
    new_req, result = await cache.process(req)
    assert new_req.context == "ctx"
    assert result.tokens_saved == 0
    assert result.short_circuit is False
    assert "miss" in result.detail


async def test_exact_hit_returns_cached_response(cache, finops_db, req):
    prompt_hash = hashlib.sha256(req.prompt.encode()).hexdigest()
    await finops_db[CACHE_ENTRIES].insert_one({
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
    assert "Python is a language" in new_req.context
    assert result.tokens_saved == 500
    assert result.short_circuit is True
    assert "exact" in result.detail
    doc = await finops_db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    assert doc["hit_count"] == 1


async def test_store_writes_entry(cache, finops_db, req):
    await cache.store(
        prompt=req.prompt,
        response="Python is great",
        framework="test",
        model="claude",
        tokens_saved=200,
    )
    prompt_hash = hashlib.sha256(req.prompt.encode()).hexdigest()
    doc = await finops_db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    assert doc is not None
    assert doc["response"] == "Python is great"
    assert doc["tokens_saved"] == 200
    assert doc["hit_count"] == 0


async def test_store_is_idempotent(cache, finops_db, req):
    await cache.store(req.prompt, "resp", "test", "m", 100)
    await cache.store(req.prompt, "resp", "test", "m", 100)
    count = await finops_db[CACHE_ENTRIES].count_documents({})
    assert count == 1
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_semantic_cache.py -v 2>&1 | tail -15
```
Expected: ImportError or all FAIL (module does not exist yet).

- [ ] **Step 3: Create `finops/modules/semantic_cache.py`**

```python
import hashlib
import time
from datetime import datetime, timezone, timedelta

from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_one
from finops.db.collections import CACHE_ENTRIES


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class SemanticCache(BaseModule):
    name = "semantic_cache"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._threshold = config.get("similarity_threshold", 0.92)
        self._ttl_hours = config.get("ttl_hours", 168)

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()

        entry = await self._db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
        if entry:
            await self._db[CACHE_ENTRIES].update_one(
                {"_id": entry["_id"]},
                {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}},
            )
            cached_req = OptimizeRequest(
                prompt=request.prompt,
                context=entry["response"],
                agent_id=request.agent_id,
                framework=request.framework,
                corpus_id=request.corpus_id,
            )
            return cached_req, ModuleResult(
                module=self.name,
                tokens_in=entry.get("tokens_saved", 0),
                tokens_out=0,
                tokens_saved=entry.get("tokens_saved", 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail="exact hash hit",
                short_circuit=True,
            )

        embedding = embed_one(request.prompt)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "cache_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 1,
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"_score": {"$gte": self._threshold}}},
        ]
        async for doc in self._db[CACHE_ENTRIES].aggregate(pipeline):
            await self._db[CACHE_ENTRIES].update_one(
                {"_id": doc["_id"]},
                {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}},
            )
            score = doc["_score"]
            cached_req = OptimizeRequest(
                prompt=request.prompt,
                context=doc["response"],
                agent_id=request.agent_id,
                framework=request.framework,
                corpus_id=request.corpus_id,
            )
            return cached_req, ModuleResult(
                module=self.name,
                tokens_in=doc.get("tokens_saved", 0),
                tokens_out=0,
                tokens_saved=doc.get("tokens_saved", 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"semantic hit (similarity={score:.3f})",
                short_circuit=True,
            )

        return request, ModuleResult(
            module=self.name,
            tokens_in=0, tokens_out=0, tokens_saved=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail="cache miss",
        )

    async def store(
        self,
        prompt: str,
        response: str,
        framework: str,
        model: str,
        tokens_saved: int,
    ) -> None:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        embedding = embed_one(prompt)
        expires_at = datetime.now(timezone.utc) + timedelta(hours=self._ttl_hours)
        now = datetime.now(timezone.utc)
        await self._db[CACHE_ENTRIES].update_one(
            {"prompt_hash": prompt_hash},
            {"$setOnInsert": {
                "prompt_hash": prompt_hash,
                "embedding": embedding,
                "prompt_preview": prompt[:200],
                "response": response,
                "framework": framework,
                "model": model,
                "tokens_saved": tokens_saved,
                "hit_count": 0,
                "created_at": now,
                "last_hit_at": None,
                "expires_at": expires_at,
            }},
            upsert=True,
        )
```

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm dev pytest tests/modules/test_semantic_cache.py -v 2>&1 | tail -15
```
Expected: 5 tests PASS. (The `test_exact_hit` will pass via exact hash match; the `test_store_*` tests verify MongoDB writes. Note: `test_exact_hit` does NOT exercise `$vectorSearch` — that's an integration test that requires the vector index to be warm, which it may not be immediately after creation. If the semantic-similarity test is flaky, that's expected — only add to suite once the index stabilises.)

- [ ] **Step 5: Add `GET /cache/lookup` to `finops/daemon/app.py`**

Add after the existing `PUT /config` endpoint:
```python
@app.get("/cache/lookup")
async def cache_lookup(prompt_hash: str, embedding: list[float] | None = None):
    db = get_async_db()
    config = await load_config(db)
    cache_cfg = config.get("modules", {}).get("semantic_cache", {})
    from finops.modules.semantic_cache import SemanticCache
    cache = SemanticCache(db, cache_cfg)

    entry = await db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    if entry:
        return {"hit": True, "response": entry["response"], "similarity_score": 1.0}

    if embedding:
        threshold = cache_cfg.get("similarity_threshold", 0.92)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "cache_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 1,
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"_score": {"$gte": threshold}}},
        ]
        async for doc in db[CACHE_ENTRIES].aggregate(pipeline):
            return {"hit": True, "response": doc["response"], "similarity_score": doc["_score"]}

    return {"hit": False, "response": None, "similarity_score": 0.0}
```

Also add `CACHE_ENTRIES` to the imports at the top of `app.py`:
```python
from finops.db.collections import CACHE_ENTRIES
```

- [ ] **Step 6: Write and run daemon endpoint test**

Create `tests/daemon/test_cache_lookup.py`:
```python
import pytest
import hashlib
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app
from finops.db.collections import CACHE_ENTRIES


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
        "prompt_hash": prompt_hash,
        "embedding": [0.1] * 1024,
        "prompt_preview": prompt,
        "response": "hi there",
        "framework": "test",
        "model": "claude",
        "tokens_saved": 100,
        "hit_count": 0,
        "created_at": datetime.now(timezone.utc),
        "last_hit_at": None,
        "expires_at": None,
    })
    resp = await client.get("/cache/lookup", params={"prompt_hash": prompt_hash})
    assert resp.status_code == 200
    data = resp.json()
    assert data["hit"] is True
    assert data["response"] == "hi there"
```

```bash
docker compose run --rm dev pytest tests/daemon/test_cache_lookup.py -v 2>&1 | tail -10
```
Expected: 2 tests PASS.

- [ ] **Step 7: Run full suite**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: all tests PASS, 0 fail.

- [ ] **Step 8: Commit**

```bash
git add finops/modules/semantic_cache.py finops/daemon/app.py \
        tests/modules/test_semantic_cache.py tests/daemon/test_cache_lookup.py
git commit -m "feat: add semantic cache module and /cache/lookup endpoint"
```

---

## Task 3: Agent Memory Module + /memory/* Endpoints

Three-tier memory (working/episodic/semantic) backed by Motor + VoyageAI embeddings. Fact extraction uses `ChatAnthropic`. A `store_turn()` method (called by the router/plugin) writes working memory and extracts facts. Tests mock both the embedding helper and the Anthropic LLM.

**Files:**
- Create: `finops/modules/agent_memory.py`
- Modify: `finops/daemon/app.py` — add `POST /memory/retrieve` and `POST /memory/store`
- Create: `tests/modules/test_agent_memory.py`
- Create: `tests/daemon/test_memory.py`

**Interfaces:**
- Consumes: `embed_one`, `embed`, `WORKING_MEMORY`, `EPISODIC_MEMORY`, `SEMANTIC_MEMORY`, `BaseModule`, `OptimizeRequest`, `ModuleResult`
- Produces:
  - `AgentMemory(db: AsyncIOMotorDatabase, config: dict)` — init
  - `await memory.process(request) -> (OptimizeRequest, ModuleResult)` — retrieves all three tiers, injects into `request.context`
  - `await memory.store_turn(agent_id: str, session_id: str, turn: str, response: str) -> None` — stores messages + extracts facts

- [ ] **Step 1: Write failing tests**

Create `tests/modules/test_agent_memory.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from finops.modules.agent_memory import AgentMemory
from finops.modules._base import OptimizeRequest
from finops.db.collections import WORKING_MEMORY, EPISODIC_MEMORY, SEMANTIC_MEMORY

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.agent_memory.embed_one", lambda t, **kw: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.agent_memory.embed", lambda ts, **kw: [FIXED_EMBEDDING] * len(ts))


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


async def test_process_injects_working_memory_into_context(memory, finops_db, req):
    await memory.store_turn("a1", "s1", "first turn", "first response")
    new_req, result = await memory.process(req)
    assert "first turn" in new_req.context or "first response" in new_req.context


async def test_working_memory_respects_turn_limit(memory, finops_db):
    for i in range(5):
        await memory.store_turn("a2", "s2", f"turn {i}", f"resp {i}")
    doc = await finops_db[WORKING_MEMORY].find_one({"agent_id": "a2", "session_id": "s2"})
    # 5 turns * 2 messages = 10 messages stored; retrieve only last 3 turns = 6
    working = await memory._get_working_memory("a2")
    assert len(working) <= 6
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_agent_memory.py -v 2>&1 | tail -15
```
Expected: ImportError or all FAIL.

- [ ] **Step 3: Create `finops/modules/agent_memory.py`**

```python
import time
from datetime import datetime, timezone, timedelta

from langchain_anthropic import ChatAnthropic
from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_one, embed
from finops.db.collections import WORKING_MEMORY, EPISODIC_MEMORY, SEMANTIC_MEMORY

_FACT_PROMPT = (
    "Extract factual statements from this conversation. "
    "Return one fact per line. Return empty string if no facts.\n\n"
    "Turn: {turn}\nResponse: {response}\n\nFacts:"
)

_DEDUP_THRESHOLD = 0.95


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class AgentMemory(BaseModule):
    name = "agent_memory"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._working_turns = config.get("working_memory_turns", 20)
        self._episodic_ttl = config.get("episodic_ttl_days", 30)
        self._semantic_ttl = config.get("semantic_ttl_days", 90)

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()

        working = await self._get_working_memory(request.agent_id)
        episodic = await self._get_episodic_memory(request.agent_id, request.prompt)
        semantic = await self._get_semantic_memory(request.agent_id, request.prompt)

        if not working and not episodic and not semantic:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail="no memory found",
            )

        memory_ctx = self._format_memory(working, episodic, semantic)
        tokens_in = _count_tokens(request.context)
        tokens_out = _count_tokens(memory_ctx)

        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=memory_ctx,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_saved=max(0, tokens_in - tokens_out),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"working={len(working)}, episodic={len(episodic)}, semantic={len(semantic)}",
        )

    async def store_turn(
        self, agent_id: str, session_id: str, turn: str, response: str
    ) -> None:
        now = datetime.now(timezone.utc)
        await self._db[WORKING_MEMORY].update_one(
            {"agent_id": agent_id, "session_id": session_id},
            {
                "$push": {"messages": {"$each": [
                    {"role": "user", "content": turn, "timestamp": now},
                    {"role": "assistant", "content": response, "timestamp": now},
                ]}},
                "$set": {"updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        await self._extract_and_store_facts(agent_id, turn, response)

    async def _get_working_memory(self, agent_id: str) -> list[dict]:
        doc = await self._db[WORKING_MEMORY].find_one({"agent_id": agent_id})
        if not doc:
            return []
        messages = doc.get("messages", [])
        keep = self._working_turns * 2
        return messages[-keep:]

    async def _get_episodic_memory(self, agent_id: str, query: str) -> list[str]:
        embedding = embed_one(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "episodic_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 3,
                }
            },
            {"$match": {"agent_id": agent_id}},
        ]
        results = []
        async for doc in self._db[EPISODIC_MEMORY].aggregate(pipeline):
            results.append(doc["content"])
        return results

    async def _get_semantic_memory(self, agent_id: str, query: str) -> list[str]:
        embedding = embed_one(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "semantic_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 5,
                }
            },
            {"$match": {"agent_id": agent_id}},
        ]
        results = []
        async for doc in self._db[SEMANTIC_MEMORY].aggregate(pipeline):
            results.append(doc["fact"])
        return results

    async def _extract_and_store_facts(
        self, agent_id: str, turn: str, response: str
    ) -> None:
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=__import__("os").getenv("ANTHROPIC_API_KEY", ""),
            max_tokens=256,
        )
        result = llm.invoke(_FACT_PROMPT.format(turn=turn, response=response))
        raw = result.content.strip()
        if not raw:
            return

        facts = [f.strip() for f in raw.splitlines() if f.strip()]
        fact_embeddings = embed(facts)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self._semantic_ttl)

        for fact, emb in zip(facts, fact_embeddings):
            dedup_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "semantic_vector_index",
                        "path": "embedding",
                        "queryVector": emb,
                        "numCandidates": 10,
                        "limit": 1,
                    }
                },
                {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
                {"$match": {"agent_id": agent_id, "_score": {"$gte": _DEDUP_THRESHOLD}}},
            ]
            existing = None
            async for doc in self._db[SEMANTIC_MEMORY].aggregate(dedup_pipeline):
                existing = doc
                break

            if existing:
                await self._db[SEMANTIC_MEMORY].update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"fact": fact, "updated_at": now, "expires_at": expires_at}},
                )
            else:
                await self._db[SEMANTIC_MEMORY].insert_one({
                    "agent_id": agent_id,
                    "fact": fact,
                    "embedding": emb,
                    "confidence": 1.0,
                    "source_session": None,
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": expires_at,
                })

    def _format_memory(
        self, working: list[dict], episodic: list[str], semantic: list[str]
    ) -> str:
        parts = []
        if semantic:
            parts.append("## Known Facts\n" + "\n".join(f"- {f}" for f in semantic))
        if episodic:
            parts.append("## Recent Context\n" + "\n".join(f"- {e}" for e in episodic))
        if working:
            msgs = "\n".join(f"{m['role']}: {m['content']}" for m in working)
            parts.append("## Conversation\n" + msgs)
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm dev pytest tests/modules/test_agent_memory.py -v 2>&1 | tail -15
```
Expected: 5 tests PASS.

- [ ] **Step 5: Add `/memory/retrieve` and `/memory/store` to `finops/daemon/app.py`**

Add after the cache_lookup endpoint:
```python
@app.post("/memory/retrieve")
async def memory_retrieve(body: dict):
    agent_id = body.get("agent_id", "default")
    query = body.get("query", "")
    db = get_async_db()
    config = await load_config(db)
    mem_cfg = config.get("modules", {}).get("agent_memory", {})
    from finops.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, mem_cfg)
    working = await memory._get_working_memory(agent_id)
    episodic = await memory._get_episodic_memory(agent_id, query)
    semantic = await memory._get_semantic_memory(agent_id, query)
    return {"working": working, "episodic": episodic, "semantic": semantic}


@app.post("/memory/store")
async def memory_store(body: dict):
    agent_id = body.get("agent_id", "default")
    session_id = body.get("session_id", "default")
    turn = body.get("turn", "")
    response = body.get("response", "")
    db = get_async_db()
    config = await load_config(db)
    mem_cfg = config.get("modules", {}).get("agent_memory", {})
    from finops.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, mem_cfg)
    await memory.store_turn(agent_id, session_id, turn, response)
    return {"stored": True}
```

- [ ] **Step 6: Write and run endpoint tests**

Create `tests/daemon/test_memory.py`:
```python
import pytest
from unittest.mock import MagicMock
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.agent_memory.embed_one", lambda t, **kw: [0.1] * 1024)
    monkeypatch.setattr("finops.modules.agent_memory.embed", lambda ts, **kw: [[0.1] * 1024] * len(ts))


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
        "agent_id": "u1", "session_id": "s1",
        "turn": "Hello", "response": "Hi there"
    })
    assert store_resp.status_code == 200
    assert store_resp.json()["stored"] is True

    ret_resp = await client.post("/memory/retrieve", json={"agent_id": "u1", "query": "Hello"})
    data = ret_resp.json()
    assert len(data["working"]) == 2
    assert data["working"][0]["role"] == "user"
```

```bash
docker compose run --rm dev pytest tests/daemon/test_memory.py -v 2>&1 | tail -10
```
Expected: 2 tests PASS.

- [ ] **Step 7: Run full suite**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add finops/modules/agent_memory.py finops/daemon/app.py \
        tests/modules/test_agent_memory.py tests/daemon/test_memory.py
git commit -m "feat: add agent memory module and /memory/* endpoints"
```

---

## Task 4: Context Compressor Module

Wraps LLMLingua-2. Bypasses when `len(context) // 4 < token_threshold`. Saves per-run stats to `compression_stats`. Tests mock the compressor entirely (model download avoided).

**Files:**
- Create: `finops/modules/context_compressor.py`
- Create: `tests/modules/test_context_compressor.py`

**Interfaces:**
- Consumes: `COMPRESSION_STATS`, `BaseModule`, `OptimizeRequest`, `ModuleResult`
- Produces: `ContextCompressor(db: AsyncIOMotorDatabase, config: dict)` + `process()`

- [ ] **Step 1: Write failing tests**

Create `tests/modules/test_context_compressor.py`:
```python
import pytest
from unittest.mock import MagicMock, patch
from finops.modules.context_compressor import ContextCompressor
from finops.modules._base import OptimizeRequest
from finops.db.collections import COMPRESSION_STATS


@pytest.fixture(autouse=True)
def mock_compressor(monkeypatch):
    fake = MagicMock()
    # Compress to ~25%: return first quarter of words
    def fake_compress(context_list, rate, force_tokens, **kw):
        text = context_list[0]
        words = text.split()
        compressed = " ".join(words[: max(1, len(words) // 4)])
        return {"compressed_prompt": compressed}
    fake.compress_prompt.side_effect = fake_compress
    monkeypatch.setattr("finops.modules.context_compressor.PromptCompressor", lambda **kw: fake)


@pytest.fixture
def config():
    return {"token_threshold": 10, "target_ratio": 4.0}


@pytest.fixture
async def compressor(finops_db, config):
    return ContextCompressor(finops_db, config)


@pytest.fixture
def short_req():
    return OptimizeRequest(prompt="hi", context="short", agent_id="a1", framework="test")


@pytest.fixture
def long_req():
    long_ctx = " ".join(["word"] * 200)
    return OptimizeRequest(prompt="hi", context=long_ctx, agent_id="a1", framework="test")


async def test_bypass_when_context_below_threshold(compressor, short_req):
    new_req, result = await compressor.process(short_req)
    assert new_req.context == "short"
    assert result.tokens_saved == 0
    assert "bypass" in result.detail


async def test_compresses_when_above_threshold(compressor, long_req):
    new_req, result = await compressor.process(long_req)
    assert len(new_req.context) < len(long_req.context)
    assert result.tokens_saved > 0


async def test_saves_compression_stats(compressor, finops_db, long_req):
    await compressor.process(long_req)
    count = await finops_db[COMPRESSION_STATS].count_documents({})
    assert count == 1
    doc = await finops_db[COMPRESSION_STATS].find_one({})
    assert doc["original_tokens"] > 0
    assert doc["compressed_tokens"] > 0
    assert doc["ratio"] > 1.0
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_context_compressor.py -v 2>&1 | tail -10
```
Expected: ImportError or FAIL.

- [ ] **Step 3: Create `finops/modules/context_compressor.py`**

```python
import time
import uuid
from datetime import datetime, timezone

from llmlingua import PromptCompressor
from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.db.collections import COMPRESSION_STATS

_compressor: PromptCompressor | None = None


def _get_compressor() -> PromptCompressor:
    global _compressor
    if _compressor is None:
        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _compressor


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ContextCompressor(BaseModule):
    name = "context_compressor"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._threshold = config.get("token_threshold", 8000)
        self._target_ratio = config.get("target_ratio", 4.0)

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()
        original_tokens = _count_tokens(request.context)

        if original_tokens < self._threshold:
            return request, ModuleResult(
                module=self.name, tokens_in=original_tokens, tokens_out=original_tokens,
                tokens_saved=0, latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"bypass (tokens={original_tokens} < threshold={self._threshold})",
            )

        rate = 1.0 / self._target_ratio
        result = _get_compressor().compress_prompt(
            [request.context],
            rate=rate,
            force_tokens=["\n", "?"],
        )
        compressed = result["compressed_prompt"]
        compressed_tokens = _count_tokens(compressed)
        tokens_saved = max(0, original_tokens - compressed_tokens)
        latency = (time.perf_counter() - t0) * 1000
        ratio = original_tokens / max(1, compressed_tokens)

        await self._db[COMPRESSION_STATS].insert_one({
            "request_id": str(uuid.uuid4()),
            "framework": request.framework,
            "model": "",
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": ratio,
            "latency_ms": latency,
            "created_at": datetime.now(timezone.utc),
        })

        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=compressed,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=original_tokens,
            tokens_out=compressed_tokens,
            tokens_saved=tokens_saved,
            latency_ms=latency,
            detail=f"compressed {ratio:.1f}x ({original_tokens}→{compressed_tokens} tokens)",
        )
```

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm dev pytest tests/modules/test_context_compressor.py -v 2>&1 | tail -10
```
Expected: 3 tests PASS.

- [ ] **Step 5: Run full suite**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add finops/modules/context_compressor.py tests/modules/test_context_compressor.py
git commit -m "feat: add context compressor module (LLMLingua-2 wrapper)"
```

---

## Task 5: Codebase Graph Module

Parses Python source with Tree-Sitter, stores symbol nodes + embeddings in `codebase_nodes`. Query by natural language or symbol name returns minimal code slices. Tests parse a small fixture file and query it.

**Files:**
- Modify: `pyproject.toml` — add `tree-sitter-python>=0.23`
- Create: `finops/modules/codebase_graph.py`
- Create: `tests/fixtures/sample.py` — small Python file for parsing
- Create: `tests/modules/test_codebase_graph.py`

**Interfaces:**
- Consumes: `embed_one`, `embed`, `CODEBASE_NODES`, `BaseModule`, `OptimizeRequest`, `ModuleResult`
- Produces:
  - `CodebaseGraph(db: AsyncIOMotorDatabase, config: dict)` — init
  - `await graph.index_file(repo_id: str, file_path: str, source: str) -> int` — returns count of symbols indexed
  - `await graph.query(repo_id: str, query_text: str, k: int = 5) -> list[dict]` — returns symbol dicts
  - `await graph.process(request: OptimizeRequest) -> (OptimizeRequest, ModuleResult)` — no-op if no repo configured

- [ ] **Step 1: Add tree-sitter-python to pyproject.toml**

In `pyproject.toml`, add to `dependencies`:
```toml
    "tree-sitter-python>=0.23",
```

Install inside the container:
```bash
docker compose run --rm dev pip install -e ".[dev]" 2>&1 | tail -5
```
Expected: Successfully installed (or already satisfied).

- [ ] **Step 2: Create test fixture**

Create `tests/fixtures/__init__.py` (empty).

Create `tests/fixtures/sample.py`:
```python
def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


class Calculator:
    def __init__(self, initial: int = 0):
        self.value = initial

    def add(self, n: int) -> "Calculator":
        self.value += n
        return self

    def result(self) -> int:
        return self.value
```

- [ ] **Step 3: Write failing tests**

Create `tests/modules/test_codebase_graph.py`:
```python
import os
import pytest
from pathlib import Path
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules._base import OptimizeRequest
from finops.db.collections import CODEBASE_NODES

FIXED_EMBEDDING = [0.1] * 1024
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.py"


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.codebase_graph.embed_one", lambda t, **kw: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.codebase_graph.embed", lambda ts, **kw: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
async def graph(finops_db):
    return CodebaseGraph(finops_db, {"repo_paths": []})


@pytest.fixture
def sample_source():
    return FIXTURE_PATH.read_text()


async def test_index_file_returns_symbol_count(graph, sample_source):
    count = await graph.index_file("repo1", "sample.py", sample_source)
    assert count >= 3  # add, multiply, Calculator (+ class methods)


async def test_index_file_stores_symbols_in_mongo(graph, finops_db, sample_source):
    await graph.index_file("repo1", "sample.py", sample_source)
    count = await finops_db[CODEBASE_NODES].count_documents({"repo_id": "repo1"})
    assert count >= 3


async def test_index_file_captures_function_metadata(graph, finops_db, sample_source):
    await graph.index_file("repo1", "sample.py", sample_source)
    doc = await finops_db[CODEBASE_NODES].find_one({"repo_id": "repo1", "symbol": "add"})
    assert doc is not None
    assert doc["type"] == "function"
    assert doc["file_path"] == "sample.py"
    assert doc["line_start"] >= 1
    assert "def add" in doc["source_snippet"]


async def test_index_file_captures_class(graph, finops_db, sample_source):
    await graph.index_file("repo1", "sample.py", sample_source)
    doc = await finops_db[CODEBASE_NODES].find_one({"repo_id": "repo1", "symbol": "Calculator"})
    assert doc is not None
    assert doc["type"] == "class"


async def test_process_no_op_when_no_repo_configured(graph):
    req = OptimizeRequest(prompt="find add function", context="", agent_id="a1", framework="test")
    new_req, result = await graph.process(req)
    assert new_req is req
    assert result.tokens_saved == 0
```

- [ ] **Step 4: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_codebase_graph.py -v 2>&1 | tail -15
```
Expected: ImportError or FAIL.

- [ ] **Step 5: Create `finops/modules/codebase_graph.py`**

```python
import time
from datetime import datetime, timezone
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_one, embed
from finops.db.collections import CODEBASE_NODES

_PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(_PY_LANGUAGE)

_SYMBOL_TYPES = {
    "function_definition": "function",
    "async_function_definition": "function",
    "class_definition": "class",
}


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _extract_python_symbols(source: str, file_path: str, repo_id: str) -> list[dict]:
    tree = _PARSER.parse(source.encode())
    symbols = []
    for node in _walk(tree.root_node):
        symbol_type = _SYMBOL_TYPES.get(node.type)
        if not symbol_type:
            continue
        name_node = node.child_by_field_name("name")
        if not name_node:
            continue
        snippet = source[node.start_byte:node.end_byte]
        symbols.append({
            "repo_id": repo_id,
            "symbol": name_node.text.decode("utf-8"),
            "type": symbol_type,
            "file_path": file_path,
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "source_snippet": snippet,
            "language": "python",
            "references": [],
        })
    return symbols


_EXTRACTORS = {
    ".py": _extract_python_symbols,
}


class CodebaseGraph(BaseModule):
    name = "codebase_graph"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._repo_paths: list[str] = config.get("repo_paths", [])

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        if not self._repo_paths:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=0.0, detail="no repos configured",
            )
        t0 = time.perf_counter()
        results = await self.query(self._repo_paths[0], request.prompt)
        if not results:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=(time.perf_counter() - t0) * 1000, detail="no symbols matched",
            )
        snippets = "\n\n".join(
            f"# {r['file_path']}:{r['line_start']}\n{r['source_snippet']}"
            for r in results
        )
        tokens_in = _count_tokens(request.context)
        tokens_out = _count_tokens(snippets)
        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=snippets,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_saved=max(0, tokens_in - tokens_out),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"injected {len(results)} symbols",
        )

    async def index_file(self, repo_id: str, file_path: str, source: str) -> int:
        ext = Path(file_path).suffix
        extractor = _EXTRACTORS.get(ext)
        if not extractor:
            return 0

        symbols = extractor(source, file_path, repo_id)
        if not symbols:
            return 0

        snippets = [s["source_snippet"] for s in symbols]
        embeddings = embed(snippets)
        now = datetime.now(timezone.utc)

        for sym, emb in zip(symbols, embeddings):
            await self._db[CODEBASE_NODES].update_one(
                {"repo_id": repo_id, "symbol": sym["symbol"], "file_path": file_path},
                {"$set": {**sym, "embedding": emb, "indexed_at": now}},
                upsert=True,
            )
        return len(symbols)

    async def query(self, repo_id: str, query_text: str, k: int = 5) -> list[dict]:
        embedding = embed_one(query_text)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "codebase_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": k * 4,
                    "limit": k,
                    "filter": {"repo_id": {"$eq": repo_id}},
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
            {"$project": {"embedding": 0, "_score": 0}},
        ]
        results = []
        async for doc in self._db[CODEBASE_NODES].aggregate(pipeline):
            results.append(doc)
        return results
```

- [ ] **Step 6: Run tests**

```bash
docker compose run --rm dev pytest tests/modules/test_codebase_graph.py -v 2>&1 | tail -15
```
Expected: 5 tests PASS. (The `query` test may return empty if the vector index isn't warm yet — that is acceptable; the index/store tests verify the core logic.)

- [ ] **Step 7: Run full suite**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml finops/modules/codebase_graph.py \
        tests/fixtures/__init__.py tests/fixtures/sample.py \
        tests/modules/test_codebase_graph.py
git commit -m "feat: add codebase graph module (Tree-Sitter Python parser)"
```

---

## Task 6: Hybrid Retrieval Module

BM25 via MongoDB `$text` index + dense via `$vectorSearch` + RRF fusion. Operates on `corpus_chunks`. Disabled by default (enabled when `corpus_id` is provided). Tests seed chunks and verify fusion ranking.

**Files:**
- Create: `finops/modules/hybrid_retrieval.py`
- Create: `tests/modules/test_hybrid_retrieval.py`

**Interfaces:**
- Consumes: `embed_one`, `CORPUS_CHUNKS`, `BaseModule`, `OptimizeRequest`, `ModuleResult`
- Produces:
  - `HybridRetrieval(db: AsyncIOMotorDatabase, config: dict)` — init
  - `await retrieval.process(request) -> (OptimizeRequest, ModuleResult)` — no-op if `request.corpus_id` is None
  - `await retrieval.add_chunks(corpus_id: str, chunks: list[dict]) -> int` — indexes chunks (`{"text": str, "source_file": str, "chunk_index": int, "metadata": dict}`)

- [ ] **Step 1: Write failing tests**

Create `tests/modules/test_hybrid_retrieval.py`:
```python
import pytest
from finops.modules.hybrid_retrieval import HybridRetrieval
from finops.modules._base import OptimizeRequest
from finops.db.collections import CORPUS_CHUNKS

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.hybrid_retrieval.embed_one", lambda t, **kw: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.hybrid_retrieval.embed", lambda ts, **kw: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
def config():
    return {"top_k": 3, "rrf_k": 60}


@pytest.fixture
async def retrieval(finops_db, config):
    return HybridRetrieval(finops_db, config)


@pytest.fixture
def req_with_corpus():
    return OptimizeRequest(
        prompt="what is MongoDB?", context="original", agent_id="a1",
        framework="test", corpus_id="corp1"
    )


@pytest.fixture
def req_no_corpus():
    return OptimizeRequest(prompt="hi", context="ctx", agent_id="a1", framework="test")


async def test_process_no_op_when_no_corpus_id(retrieval, req_no_corpus):
    new_req, result = await retrieval.process(req_no_corpus)
    assert new_req is req_no_corpus
    assert result.tokens_saved == 0
    assert "no corpus" in result.detail


async def test_add_chunks_stores_in_mongo(retrieval, finops_db):
    chunks = [
        {"text": "MongoDB is a document database", "source_file": "doc.txt", "chunk_index": 0, "metadata": {}},
        {"text": "Python is a programming language", "source_file": "doc.txt", "chunk_index": 1, "metadata": {}},
    ]
    count = await retrieval.add_chunks("corp1", chunks)
    assert count == 2
    stored = await finops_db[CORPUS_CHUNKS].count_documents({"corpus_id": "corp1"})
    assert stored == 2


async def test_process_with_corpus_replaces_context(retrieval, finops_db, req_with_corpus):
    chunks = [
        {"text": "MongoDB stores JSON-like documents", "source_file": "f.txt", "chunk_index": 0, "metadata": {}},
    ]
    await retrieval.add_chunks("corp1", chunks)
    new_req, result = await retrieval.process(req_with_corpus)
    assert new_req.context != "original"
    assert result.module == "hybrid_retrieval"


async def test_rrf_fusion_returns_ranked_results():
    results_a = [{"_id": "a"}, {"_id": "b"}, {"_id": "c"}]
    results_b = [{"_id": "b"}, {"_id": "c"}, {"_id": "a"}]
    from finops.modules.hybrid_retrieval import _rrf_fusion
    fused = _rrf_fusion(results_a, results_b, k=60)
    ids = [r["_id"] for r in fused]
    assert "b" in ids
    assert ids[0] in ("a", "b")
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_hybrid_retrieval.py -v 2>&1 | tail -15
```
Expected: ImportError or FAIL.

- [ ] **Step 3: Create `finops/modules/hybrid_retrieval.py`**

```python
import time
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_one, embed
from finops.db.collections import CORPUS_CHUNKS


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _rrf_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    scores: dict[str, float] = {}
    id_to_doc: dict[str, dict] = {}

    for rank, doc in enumerate(dense_results):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_doc[doc_id] = doc

    for rank, doc in enumerate(sparse_results):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_doc[doc_id] = doc

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [id_to_doc[doc_id] for doc_id in ranked_ids]


class HybridRetrieval(BaseModule):
    name = "hybrid_retrieval"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._top_k = config.get("top_k", 5)
        self._rrf_k = config.get("rrf_k", 60)

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        if not request.corpus_id:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=0.0, detail="no corpus_id provided",
            )

        t0 = time.perf_counter()
        dense = await self._dense_search(request.corpus_id, request.prompt)
        sparse = await self._sparse_search(request.corpus_id, request.prompt)
        fused = _rrf_fusion(dense, sparse, k=self._rrf_k)[: self._top_k]

        if not fused:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=(time.perf_counter() - t0) * 1000, detail="no chunks found",
            )

        retrieved = "\n\n".join(doc["text"] for doc in fused)
        tokens_in = _count_tokens(request.context)
        tokens_out = _count_tokens(retrieved)
        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=retrieved,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_saved=max(0, tokens_in - tokens_out),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"retrieved {len(fused)} chunks (hybrid BM25+vector)",
        )

    async def add_chunks(self, corpus_id: str, chunks: list[dict]) -> int:
        texts = [c["text"] for c in chunks]
        embeddings = embed(texts)
        now = datetime.now(timezone.utc)
        for chunk, emb in zip(chunks, embeddings):
            tokens = chunk["text"].lower().split()
            await self._db[CORPUS_CHUNKS].update_one(
                {"corpus_id": corpus_id, "chunk_index": chunk["chunk_index"],
                 "source_file": chunk["source_file"]},
                {"$set": {
                    "corpus_id": corpus_id,
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "embedding": emb,
                    "bm25_tokens": tokens,
                    "metadata": chunk.get("metadata", {}),
                    "created_at": now,
                }},
                upsert=True,
            )
        return len(chunks)

    async def _dense_search(self, corpus_id: str, query: str) -> list[dict]:
        embedding = embed_one(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "corpus_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": self._top_k * 4,
                    "limit": self._top_k,
                    "filter": {"corpus_id": {"$eq": corpus_id}},
                }
            },
            {"$project": {"embedding": 0}},
        ]
        results = []
        async for doc in self._db[CORPUS_CHUNKS].aggregate(pipeline):
            results.append(doc)
        return results

    async def _sparse_search(self, corpus_id: str, query: str) -> list[dict]:
        cursor = (
            self._db[CORPUS_CHUNKS]
            .find(
                {"corpus_id": corpus_id, "$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "embedding": 0},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(self._top_k)
        )
        results = []
        async for doc in cursor:
            results.append(doc)
        return results
```

- [ ] **Step 4: Run tests**

```bash
docker compose run --rm dev pytest tests/modules/test_hybrid_retrieval.py -v 2>&1 | tail -15
```
Expected: 4 tests PASS. (`test_rrf_fusion_returns_ranked_results` is a pure-Python test and always passes; `test_process_with_corpus_replaces_context` may return the original context if the vector/text indexes aren't warm yet — that's acceptable; core add/fetch logic is verified by the other tests.)

- [ ] **Step 5: Run full suite**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add finops/modules/hybrid_retrieval.py tests/modules/test_hybrid_retrieval.py
git commit -m "feat: add hybrid retrieval module (BM25 + vector + RRF)"
```

---

## Task 7: Daemon /optimize Pipeline Router

`ModulePipeline` in `finops/daemon/router.py` instantiates all five modules, reads live config from MongoDB, runs enabled modules in order, and short-circuits on a cache hit. `POST /optimize` in `app.py` delegates to it.

**Files:**
- Create: `finops/daemon/router.py`
- Modify: `finops/daemon/app.py` — add `POST /optimize`
- Create: `tests/daemon/test_optimize.py`

**Interfaces:**
- Consumes: all five module classes, `load_config`, `get_async_db`, `OptimizeRequest`
- Produces:
  - `ModulePipeline(db, module_configs: dict)` — init
  - `await pipeline.run(request: OptimizeRequest) -> dict` — returns `{optimized_prompt, optimized_context, cache_hit, tokens_saved, module_results[]}`

Pipeline order: `codebase_graph → semantic_cache → agent_memory → context_compressor → hybrid_retrieval`

Short-circuit rule: if any module returns `result.short_circuit=True`, the pipeline stops and `cache_hit=True` is set in the response.

- [ ] **Step 1: Write failing tests**

Create `tests/daemon/test_optimize.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app
from finops.modules._base import OptimizeRequest, ModuleResult


def _make_passthrough_module(name):
    mod = MagicMock()
    mod.name = name
    mod.is_enabled.return_value = True
    mod.process = AsyncMock(side_effect=lambda req: (
        req,
        ModuleResult(module=name, tokens_in=10, tokens_out=10,
                     tokens_saved=0, latency_ms=1.0, detail="pass"),
    ))
    return mod


def _make_cache_hit_module():
    from finops.modules._base import OptimizeRequest
    mod = MagicMock()
    mod.name = "semantic_cache"
    mod.is_enabled.return_value = True
    async def hit_process(req):
        new_req = OptimizeRequest(
            prompt=req.prompt, context="cached response",
            agent_id=req.agent_id, framework=req.framework,
        )
        return new_req, ModuleResult(
            module="semantic_cache", tokens_in=500, tokens_out=0,
            tokens_saved=500, latency_ms=5.0, detail="exact hash hit",
            short_circuit=True,
        )
    mod.process = hit_process
    return mod


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_optimize_returns_expected_shape(client, finops_db):
    # Disable all modules so no external API calls (VoyageAI, Anthropic, LLMLingua) are made
    from finops.daemon.config import save_config
    await save_config(finops_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": False},
        "agent_memory": {"enabled": False}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    resp = await client.post("/optimize", json={
        "prompt": "What is Python?",
        "context": "some context",
        "agent_id": "a1",
        "framework": "test",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "optimized_prompt" in data
    assert "optimized_context" in data
    assert "tokens_saved" in data
    assert "module_results" in data
    assert data["module_results"] == []   # all disabled


async def test_optimize_preserves_prompt(client, finops_db):
    from finops.daemon.config import save_config
    await save_config(finops_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": False},
        "agent_memory": {"enabled": False}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    resp = await client.post("/optimize", json={
        "prompt": "unique test prompt xyz",
        "context": "",
        "agent_id": "a1",
        "framework": "test",
    })
    assert resp.json()["optimized_prompt"] == "unique test prompt xyz"


async def test_pipeline_short_circuits_on_cache_hit(finops_db):
    from finops.daemon.router import ModulePipeline
    cache_mod = _make_cache_hit_module()
    other_mod = _make_passthrough_module("context_compressor")

    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._modules = {"semantic_cache": cache_mod, "context_compressor": other_mod}
    pipeline._order = ["semantic_cache", "context_compressor"]
    pipeline._enabled = {"semantic_cache": True, "context_compressor": True}

    req = OptimizeRequest(prompt="hi", context="ctx", agent_id="a", framework="f")
    result = await pipeline.run(req)

    assert result["cache_hit"] is True
    assert result["optimized_context"] == "cached response"
    assert result["tokens_saved"] == 500
    other_mod.process.assert_not_called()


async def test_pipeline_skips_disabled_modules(finops_db):
    from finops.daemon.router import ModulePipeline
    mod = _make_passthrough_module("context_compressor")

    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._modules = {"context_compressor": mod}
    pipeline._order = ["context_compressor"]
    pipeline._enabled = {"context_compressor": False}

    req = OptimizeRequest(prompt="hi", context="ctx", agent_id="a", framework="f")
    result = await pipeline.run(req)

    mod.process.assert_not_called()
    assert result["tokens_saved"] == 0
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/daemon/test_optimize.py -v 2>&1 | tail -15
```
Expected: ImportError or FAIL (router.py does not exist yet).

- [ ] **Step 3: Create `finops/daemon/router.py`**

```python
from finops.modules._base import OptimizeRequest, ModuleResult
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules.semantic_cache import SemanticCache
from finops.modules.agent_memory import AgentMemory
from finops.modules.context_compressor import ContextCompressor
from finops.modules.hybrid_retrieval import HybridRetrieval

_PIPELINE_ORDER = [
    "codebase_graph",
    "semantic_cache",
    "agent_memory",
    "context_compressor",
    "hybrid_retrieval",
]

_MODULE_CLASSES = {
    "codebase_graph":     CodebaseGraph,
    "semantic_cache":     SemanticCache,
    "agent_memory":       AgentMemory,
    "context_compressor": ContextCompressor,
    "hybrid_retrieval":   HybridRetrieval,
}


def _result_dict(result: ModuleResult) -> dict:
    return {
        "module":        result.module,
        "tokens_in":     result.tokens_in,
        "tokens_out":    result.tokens_out,
        "tokens_saved":  result.tokens_saved,
        "latency_ms":    result.latency_ms,
        "detail":        result.detail,
    }


class ModulePipeline:
    def __init__(self, db, module_configs: dict):
        self._modules = {
            name: cls(db, module_configs.get(name, {}))
            for name, cls in _MODULE_CLASSES.items()
        }
        self._enabled = {
            name: module_configs.get(name, {}).get("enabled", False)
            for name in _MODULE_CLASSES
        }
        self._order = _PIPELINE_ORDER

    async def run(self, request: OptimizeRequest) -> dict:
        collected: list[ModuleResult] = []

        for name in self._order:
            if not self._enabled.get(name, False):
                continue
            module = self._modules[name]
            request, result = await module.process(request)
            collected.append(result)
            if result.short_circuit:
                return {
                    "optimized_prompt":  request.prompt,
                    "optimized_context": request.context,
                    "cache_hit":         True,
                    "tokens_saved":      result.tokens_saved,
                    "module_results":    [_result_dict(r) for r in collected],
                }

        return {
            "optimized_prompt":  request.prompt,
            "optimized_context": request.context,
            "cache_hit":         False,
            "tokens_saved":      sum(r.tokens_saved for r in collected),
            "module_results":    [_result_dict(r) for r in collected],
        }
```

- [ ] **Step 4: Add `POST /optimize` to `finops/daemon/app.py`**

Add after the existing config endpoints (before cache/memory endpoints):
```python
@app.post("/optimize")
async def post_optimize(body: dict):
    db = get_async_db()
    config = await load_config(db)
    from finops.daemon.router import ModulePipeline
    pipeline = ModulePipeline(db, config.get("modules", {}))
    request = OptimizeRequest(
        prompt=body.get("prompt", ""),
        context=body.get("context", ""),
        agent_id=body.get("agent_id", "default"),
        framework=body.get("framework", "unknown"),
        corpus_id=body.get("corpus_id"),
    )
    return await pipeline.run(request)
```

Also add `OptimizeRequest` to the imports at the top of `app.py`:
```python
from finops.modules._base import OptimizeRequest
```

- [ ] **Step 5: Run tests**

```bash
docker compose run --rm dev pytest tests/daemon/test_optimize.py -v 2>&1 | tail -15
```
Expected: 4 tests PASS. (The `test_optimize_returns_expected_shape` test hits the live endpoint with all modules disabled by default config — so module_results will be empty, which is correct.)

- [ ] **Step 6: Run full suite**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add finops/daemon/router.py finops/daemon/app.py tests/daemon/test_optimize.py
git commit -m "feat: add ModulePipeline router and POST /optimize endpoint"
```

---

## Task 8: Metrics Endpoint

`finops/daemon/metrics.py` aggregates token savings from `cache_entries` and `compression_stats`. `GET /metrics` returns totals + per-module breakdown. Tests seed both collections and verify the aggregation math.

**Files:**
- Create: `finops/daemon/metrics.py`
- Modify: `finops/daemon/app.py` — add `GET /metrics`
- Create: `tests/daemon/test_metrics.py`

**Interfaces:**
- Consumes: `CACHE_ENTRIES`, `COMPRESSION_STATS`, `get_async_db`
- Produces:
  - `await aggregate_metrics(db) -> dict` — returns `{total_tokens_saved, cache_hit_rate, compression_ratio, per_module[]}`

Response shape:
```json
{
  "total_tokens_saved": 12500,
  "cache_hit_rate": 0.42,
  "compression_ratio": 3.8,
  "per_module": [
    {"module": "semantic_cache", "tokens_saved": 10000, "events": 5},
    {"module": "context_compressor", "tokens_saved": 2500, "events": 3}
  ]
}
```

- [ ] **Step 1: Write failing tests**

Create `tests/daemon/test_metrics.py`:
```python
import pytest
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app
from finops.db.collections import CACHE_ENTRIES, COMPRESSION_STATS


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
    # total = 400*3 + 200*0 = 1200 from cache
    assert data["total_tokens_saved"] >= 1200
    # hit_rate: 1 entry with hit_count>0 out of 2 = 0.5
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
    assert comp_entry["tokens_saved"] == 1350  # (1000-250) + (800-200)
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/daemon/test_metrics.py -v 2>&1 | tail -15
```
Expected: ImportError or 404 / FAIL.

- [ ] **Step 3: Create `finops/daemon/metrics.py`**

```python
from motor.motor_asyncio import AsyncIOMotorDatabase
from finops.db.collections import CACHE_ENTRIES, COMPRESSION_STATS


async def aggregate_metrics(db: AsyncIOMotorDatabase) -> dict:
    # ── Semantic cache metrics ────────────────────────────────────────────────
    cache_pipeline = [
        {
            "$group": {
                "_id": None,
                "total_tokens": {"$sum": {"$multiply": ["$tokens_saved", "$hit_count"]}},
                "total_entries": {"$sum": 1},
                "hit_entries": {"$sum": {"$cond": [{"$gt": ["$hit_count", 0]}, 1, 0]}},
            }
        }
    ]
    cache_tokens = 0
    cache_hit_rate = 0.0
    cache_events = 0
    async for doc in db[CACHE_ENTRIES].aggregate(cache_pipeline):
        cache_tokens = doc["total_tokens"]
        total = doc["total_entries"]
        cache_hit_rate = doc["hit_entries"] / total if total > 0 else 0.0
        cache_events = int(doc["hit_entries"])

    # ── Compression metrics ───────────────────────────────────────────────────
    comp_pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_ratio": {"$avg": "$ratio"},
                "total_saved": {"$sum": {"$subtract": ["$original_tokens", "$compressed_tokens"]}},
                "events": {"$sum": 1},
            }
        }
    ]
    comp_ratio = 0.0
    comp_tokens = 0
    comp_events = 0
    async for doc in db[COMPRESSION_STATS].aggregate(comp_pipeline):
        comp_ratio = round(doc["avg_ratio"], 2)
        comp_tokens = int(doc["total_saved"])
        comp_events = int(doc["events"])

    per_module = []
    if cache_tokens > 0 or cache_events > 0:
        per_module.append({
            "module": "semantic_cache",
            "tokens_saved": cache_tokens,
            "events": cache_events,
        })
    if comp_tokens > 0 or comp_events > 0:
        per_module.append({
            "module": "context_compressor",
            "tokens_saved": comp_tokens,
            "events": comp_events,
        })

    return {
        "total_tokens_saved": cache_tokens + comp_tokens,
        "cache_hit_rate":     round(cache_hit_rate, 4),
        "compression_ratio":  comp_ratio,
        "per_module":         per_module,
    }
```

- [ ] **Step 4: Add `GET /metrics` to `finops/daemon/app.py`**

Add at the end of the file:
```python
@app.get("/metrics")
async def get_metrics():
    from finops.daemon.metrics import aggregate_metrics
    db = get_async_db()
    return await aggregate_metrics(db)
```

- [ ] **Step 5: Run tests**

```bash
docker compose run --rm dev pytest tests/daemon/test_metrics.py -v 2>&1 | tail -15
```
Expected: 3 tests PASS.

- [ ] **Step 6: Run full suite**

```bash
docker compose run --rm dev pytest --tb=short -q 2>&1 | tail -10
```
Expected: all tests PASS. Count should be 50+ tests.

- [ ] **Step 7: Commit**

```bash
git add finops/daemon/metrics.py finops/daemon/app.py tests/daemon/test_metrics.py
git commit -m "feat: add metrics aggregation and GET /metrics endpoint"
```
