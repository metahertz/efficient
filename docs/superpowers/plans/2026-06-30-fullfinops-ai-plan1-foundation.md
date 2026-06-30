# fullFinOps-AI — Plan 1: Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the fullFinOps-AI project so that `finops start` launches a healthy FastAPI daemon with MongoDB connected, all vector indexes created, and `finops status` reports module on/off state.

**Architecture:** A local FastAPI daemon (port 7432) owns all MongoDB connections. The CLI manages the daemon process via a PID file. All six optimization modules share a `BaseModule` ABC. MongoDB 7.0+ with Atlas Search (mongot) is required — the daemon checks this on startup and exits with a clear error if not met.

**Tech Stack:** Python 3.11+, FastAPI, motor (async MongoDB), pymongo (index creation), Click (CLI), MongoDB Atlas Local Docker image (`mongodb/mongodb-atlas-local`), pytest + pytest-asyncio.

## Global Constraints

- Python ≥ 3.11 (uses `X | Y` union syntax, `match` statements)
- MongoDB ≥ 7.0 with `mongot` process (Atlas Search); use `mongodb/mongodb-atlas-local:latest` Docker image locally
- Default embedding model: `voyage-4-nano` (1024 dimensions, Voyage AI) — set in config, not used until Plan 2
- Default daemon port: 7432 — overridable via `FINOPS_PORT` env var
- Default MongoDB URI: `mongodb://localhost:27017` — overridable via `FINOPS_MONGODB_URI`
- Database name: `finops` (production), `finops_test` (tests) — set via `FINOPS_DB_NAME`
- No Mem0 dependency — memory stack built on `langchain-mongodb` (Plan 2)
- All async code uses `motor`; index creation uses sync `pymongo` (motor does not support `create_search_index`)
- `pytest-asyncio` mode: `auto` (set in `pyproject.toml`)

---

## File Structure

All files created in this plan:

```
pyproject.toml                        # package config, all deps, CLI entry point
docker-compose.yml                    # mongodb/mongodb-atlas-local service
.env.example                          # required env vars with placeholders
finops/__init__.py                    # empty
finops/db/__init__.py                 # empty
finops/db/client.py                   # motor + pymongo singletons; get_async_db(), get_sync_db()
finops/db/collections.py              # collection name string constants
finops/db/indexes.py                  # create_all_indexes(db) — idempotent, creates all vector+text indexes
finops/modules/__init__.py            # empty
finops/modules/_base.py               # OptimizeRequest, ModuleResult dataclasses; BaseModule(ABC)
finops/daemon/__init__.py             # empty
finops/daemon/config.py               # DEFAULT_CONFIG dict; load_config(), save_config()
finops/daemon/app.py                  # FastAPI app; lifespan (startup checks + index creation); /health, /config
finops/cli/__init__.py                # empty
finops/cli/main.py                    # Click CLI: start, stop, status
tests/__init__.py                     # empty
tests/conftest.py                     # sync_client, sync_db, async_client fixtures; env var setup
tests/db/__init__.py                  # empty
tests/db/test_client.py               # db name from env, async db type
tests/db/test_indexes.py              # idempotent creation, unique index enforcement
tests/modules/__init__.py             # empty
tests/modules/test_base.py            # subclass, process(), cannot instantiate ABC directly
tests/daemon/__init__.py              # empty
tests/daemon/test_health.py           # GET /health returns {status: ok, version}
tests/daemon/test_config.py           # GET /config returns defaults; PUT /config updates field
tests/cli/__init__.py                 # empty
tests/cli/test_cli.py                 # start writes PID, stop removes PID, status when down
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `finops/__init__.py`, `finops/db/__init__.py`, `finops/modules/__init__.py`, `finops/daemon/__init__.py`, `finops/cli/__init__.py`
- Create: `tests/__init__.py`, `tests/db/__init__.py`, `tests/modules/__init__.py`, `tests/daemon/__init__.py`, `tests/cli/__init__.py`

**Interfaces:**
- Produces: `finops` CLI entry point (wired in `[project.scripts]`); all package namespaces available for import

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "finops-ai"
version = "0.1.0"
description = "Token-saving developer toolkit for AI frameworks"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "motor>=3.6",
    "pymongo>=4.8",
    "click>=8.1",
    "httpx>=0.27",
    "voyageai>=0.3",
    "langchain-mongodb>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-core>=0.3",
    "python-dotenv>=1.0",
    "tree-sitter>=0.23",
    "llmlingua>=0.2",
    "rank-bm25>=0.2",
    "datasets>=2.20",
    "anthropic>=0.40",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "anyio>=4.0",
]

[project.scripts]
finops = "finops.cli.main:cli"

[tool.hatch.build.targets.wheel]
packages = ["finops"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  mongodb:
    image: mongodb/mongodb-atlas-local:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.runCommand('ping').ok"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  mongodb_data:
```

- [ ] **Step 3: Write `.env.example`**

```
FINOPS_MONGODB_URI=mongodb://localhost:27017
FINOPS_DB_NAME=finops
FINOPS_PORT=7432
VOYAGE_API_KEY=your_voyage_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

- [ ] **Step 4: Create all empty `__init__.py` files**

```bash
touch finops/__init__.py finops/db/__init__.py finops/modules/__init__.py \
      finops/daemon/__init__.py finops/cli/__init__.py \
      tests/__init__.py tests/db/__init__.py tests/modules/__init__.py \
      tests/daemon/__init__.py tests/cli/__init__.py
```

- [ ] **Step 5: Start MongoDB**

```bash
docker compose up -d
docker compose ps
```

Expected: `mongodb` service shows `Up` with `(healthy)` after ~30 seconds.

- [ ] **Step 6: Install the package in dev mode**

```bash
pip install -e ".[dev]"
```

Expected: installs without errors; `finops --help` prints help text.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml docker-compose.yml .env.example finops/ tests/
git commit -m "chore: scaffold project structure and dependencies"
```

---

### Task 2: MongoDB Client Singleton

**Files:**
- Create: `finops/db/client.py`
- Create: `finops/db/collections.py`
- Create: `tests/conftest.py`
- Create: `tests/db/test_client.py`

**Interfaces:**
- Produces:
  - `get_async_db(client=None) -> AsyncIOMotorDatabase`
  - `get_sync_db(client=None) -> Database`
  - `reset_clients() -> None` (testing only)
  - Collection name constants: `CACHE_ENTRIES`, `CONFIG`, `SEMANTIC_MEMORY`, etc.

- [ ] **Step 1: Write the failing tests**

```python
# tests/conftest.py
import os
import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from finops.db.client import reset_clients

MONGO_URI = os.getenv("FINOPS_TEST_MONGODB_URI", "mongodb://localhost:27017")
TEST_DB   = "finops_test"


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    os.environ["FINOPS_MONGODB_URI"] = MONGO_URI
    os.environ["FINOPS_DB_NAME"]     = TEST_DB


@pytest.fixture(scope="session")
def sync_client():
    client = MongoClient(MONGO_URI)
    yield client
    client.drop_database(TEST_DB)
    client.close()


@pytest.fixture
def sync_db(sync_client):
    db = sync_client[TEST_DB]
    yield db
    for name in db.list_collection_names():
        db[name].drop()


@pytest.fixture
async def async_client():
    reset_clients()
    os.environ["FINOPS_DB_NAME"] = TEST_DB
    client = AsyncIOMotorClient(MONGO_URI)
    yield client
    await client.drop_database(TEST_DB)
    client.close()
    reset_clients()
```

```python
# tests/db/test_client.py
import os
import pytest
from finops.db.client import get_sync_db, get_async_db, reset_clients


def test_sync_db_uses_env_db_name(sync_client, monkeypatch):
    monkeypatch.setenv("FINOPS_DB_NAME", "finops_test")
    reset_clients()
    db = get_sync_db()
    assert db.name == "finops_test"
    reset_clients()


async def test_async_db_name_matches_env(monkeypatch):
    monkeypatch.setenv("FINOPS_DB_NAME", "finops_test")
    reset_clients()
    db = get_async_db()
    assert db.name == "finops_test"
    reset_clients()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/db/test_client.py -v
```

Expected: `ImportError: cannot import name 'get_sync_db' from 'finops.db.client'`

- [ ] **Step 3: Write `finops/db/collections.py`**

```python
CODEBASE_NODES    = "codebase_nodes"
CACHE_ENTRIES     = "cache_entries"
WORKING_MEMORY    = "working_memory"
EPISODIC_MEMORY   = "episodic_memory"
SEMANTIC_MEMORY   = "semantic_memory"
COMPRESSION_STATS = "compression_stats"
CORPUS_CHUNKS     = "corpus_chunks"
BENCHMARK_RUNS    = "benchmark_runs"
CONFIG            = "config"
```

- [ ] **Step 4: Write `finops/db/client.py`**

```python
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database
import os

_async_client: AsyncIOMotorClient | None = None
_sync_client:  MongoClient | None = None


def get_async_client() -> AsyncIOMotorClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncIOMotorClient(
            os.getenv("FINOPS_MONGODB_URI", "mongodb://localhost:27017")
        )
    return _async_client


def get_sync_client() -> MongoClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(
            os.getenv("FINOPS_MONGODB_URI", "mongodb://localhost:27017")
        )
    return _sync_client


def get_async_db(client: AsyncIOMotorClient | None = None) -> AsyncIOMotorDatabase:
    if client is None:
        client = get_async_client()
    return client[os.getenv("FINOPS_DB_NAME", "finops")]


def get_sync_db(client: MongoClient | None = None) -> Database:
    if client is None:
        client = get_sync_client()
    return client[os.getenv("FINOPS_DB_NAME", "finops")]


def reset_clients() -> None:
    """Reset singletons. Call in tests only."""
    global _async_client, _sync_client
    _async_client = None
    _sync_client  = None
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/db/test_client.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add finops/db/client.py finops/db/collections.py tests/conftest.py tests/db/test_client.py
git commit -m "feat: add MongoDB client singleton and collection constants"
```

---

### Task 3: MongoDB Index Creation

**Files:**
- Create: `finops/db/indexes.py`
- Create: `tests/db/test_indexes.py`

**Interfaces:**
- Consumes: `get_sync_db()`, collection name constants from `finops/db/collections.py`
- Produces: `create_all_indexes(db: Database) -> None` — idempotent; safe to call on every startup

- [ ] **Step 1: Write the failing tests**

```python
# tests/db/test_indexes.py
import pytest
from pymongo.errors import DuplicateKeyError
from finops.db.indexes import create_all_indexes
from finops.db.collections import CACHE_ENTRIES, EPISODIC_MEMORY, CORPUS_CHUNKS


def test_create_indexes_is_idempotent(sync_db):
    create_all_indexes(sync_db)
    create_all_indexes(sync_db)  # must not raise


def test_unique_index_on_cache_prompt_hash(sync_db):
    create_all_indexes(sync_db)
    col = sync_db[CACHE_ENTRIES]
    col.insert_one({"prompt_hash": "abc123"})
    with pytest.raises(DuplicateKeyError):
        col.insert_one({"prompt_hash": "abc123"})


def test_ttl_index_on_cache_expires_at(sync_db):
    create_all_indexes(sync_db)
    indexes = {i["name"]: i for i in sync_db[CACHE_ENTRIES].list_indexes()}
    assert "expires_at_1" in indexes
    assert indexes["expires_at_1"].get("expireAfterSeconds") == 0


def test_text_index_on_corpus_chunks(sync_db):
    create_all_indexes(sync_db)
    indexes = {i["name"]: i for i in sync_db[CORPUS_CHUNKS].list_indexes()}
    assert any("text" in name for name in indexes)
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/db/test_indexes.py -v
```

Expected: `ImportError: cannot import name 'create_all_indexes'`

- [ ] **Step 3: Write `finops/db/indexes.py`**

```python
from pymongo import ASCENDING, TEXT
from pymongo.database import Database
from finops.db.collections import (
    CODEBASE_NODES, CACHE_ENTRIES, WORKING_MEMORY,
    EPISODIC_MEMORY, SEMANTIC_MEMORY, CORPUS_CHUNKS,
    COMPRESSION_STATS, BENCHMARK_RUNS,
)

EMBEDDING_DIMENSIONS = 1024
VECTOR_SIMILARITY    = "cosine"


def _search_index_exists(collection, name: str) -> bool:
    return any(idx["name"] == name for idx in collection.list_search_indexes())


def _create_vector_index(collection, name: str, field: str = "embedding") -> None:
    if _search_index_exists(collection, name):
        return
    collection.create_search_index({
        "name": name,
        "type": "vectorSearch",
        "definition": {
            "fields": [{
                "type": "vector",
                "path": field,
                "numDimensions": EMBEDDING_DIMENSIONS,
                "similarity": VECTOR_SIMILARITY,
            }]
        },
    })


def create_all_indexes(db: Database) -> None:
    # codebase_nodes
    col = db[CODEBASE_NODES]
    col.create_index([("repo_id", ASCENDING), ("symbol", ASCENDING)])
    col.create_index([("repo_id", ASCENDING), ("file_path", ASCENDING)])
    _create_vector_index(col, "codebase_vector_index")

    # cache_entries
    col = db[CACHE_ENTRIES]
    col.create_index([("prompt_hash", ASCENDING)], unique=True)
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "cache_vector_index")

    # working_memory
    col = db[WORKING_MEMORY]
    col.create_index([("agent_id", ASCENDING), ("session_id", ASCENDING)])

    # episodic_memory
    col = db[EPISODIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "episodic_vector_index")

    # semantic_memory
    col = db[SEMANTIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "semantic_vector_index")

    # compression_stats
    col = db[COMPRESSION_STATS]
    col.create_index([("created_at", ASCENDING)])

    # corpus_chunks
    col = db[CORPUS_CHUNKS]
    col.create_index([("corpus_id", ASCENDING)])
    col.create_index([("bm25_tokens", TEXT)])
    _create_vector_index(col, "corpus_vector_index")

    # benchmark_runs
    col = db[BENCHMARK_RUNS]
    col.create_index([("started_at", ASCENDING)])
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/db/test_indexes.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add finops/db/indexes.py tests/db/test_indexes.py
git commit -m "feat: add idempotent MongoDB index creation for all collections"
```

---

### Task 4: BaseModule ABC + Data Types

**Files:**
- Create: `finops/modules/_base.py`
- Create: `tests/modules/test_base.py`

**Interfaces:**
- Produces:
  - `OptimizeRequest(prompt, context, agent_id, framework, corpus_id=None)`
  - `ModuleResult(module, tokens_in, tokens_out, tokens_saved, latency_ms, detail)`
  - `BaseModule(ABC)` with abstract methods `process(request) -> tuple[OptimizeRequest, ModuleResult]` and `is_enabled() -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/modules/test_base.py
import pytest
from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult


class PassthroughModule(BaseModule):
    name = "passthrough"

    async def process(
        self, request: OptimizeRequest
    ) -> tuple[OptimizeRequest, ModuleResult]:
        result = ModuleResult(
            module=self.name, tokens_in=10, tokens_out=10,
            tokens_saved=0, latency_ms=0.5, detail="no-op",
        )
        return request, result

    def is_enabled(self) -> bool:
        return True


def test_module_subclass_instantiates():
    mod = PassthroughModule()
    assert mod.is_enabled()
    assert mod.name == "passthrough"


async def test_process_returns_original_request_and_result():
    mod = PassthroughModule()
    req = OptimizeRequest(
        prompt="hello", context="ctx", agent_id="agent1", framework="test"
    )
    req_out, result = await mod.process(req)
    assert req_out is req
    assert result.module == "passthrough"
    assert result.tokens_saved == 0
    assert result.latency_ms == 0.5


def test_cannot_instantiate_base_directly():
    with pytest.raises(TypeError):
        BaseModule()


def test_optimize_request_corpus_id_defaults_to_none():
    req = OptimizeRequest(prompt="p", context="c", agent_id="a", framework="f")
    assert req.corpus_id is None


def test_module_result_fields():
    r = ModuleResult(
        module="test", tokens_in=100, tokens_out=50,
        tokens_saved=50, latency_ms=12.3, detail="compressed"
    )
    assert r.tokens_saved == 50
    assert r.detail == "compressed"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/modules/test_base.py -v
```

Expected: `ImportError: cannot import name 'BaseModule'`

- [ ] **Step 3: Write `finops/modules/_base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OptimizeRequest:
    prompt:    str
    context:   str
    agent_id:  str
    framework: str
    corpus_id: str | None = None


@dataclass
class ModuleResult:
    module:       str
    tokens_in:    int
    tokens_out:   int
    tokens_saved: int
    latency_ms:   float
    detail:       str


class BaseModule(ABC):
    name: str = ""

    @abstractmethod
    async def process(
        self, request: OptimizeRequest
    ) -> tuple[OptimizeRequest, ModuleResult]:
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        ...
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/modules/test_base.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add finops/modules/_base.py tests/modules/test_base.py
git commit -m "feat: add BaseModule ABC with OptimizeRequest and ModuleResult types"
```

---

### Task 5: Daemon Skeleton — `/health` and `/config`

**Files:**
- Create: `finops/daemon/config.py`
- Create: `finops/daemon/app.py`
- Create: `tests/daemon/test_health.py`
- Create: `tests/daemon/test_config.py`

**Interfaces:**
- Consumes: `get_async_db()`, `get_sync_db()`, `create_all_indexes()`, collection `CONFIG`
- Produces:
  - `load_config(db) -> dict` — returns config doc (inserts defaults on first call)
  - `save_config(db, patch) -> dict` — merges patch into config, returns updated doc
  - FastAPI `app` with `GET /health`, `GET /config`, `PUT /config`

- [ ] **Step 1: Write the failing tests**

```python
# tests/daemon/test_health.py
import pytest
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app


async def test_health_returns_ok():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
```

```python
# tests/daemon/test_config.py
import pytest
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app


async def test_get_config_returns_defaults(async_client):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "modules" in data
    assert "semantic_cache" in data["modules"]
    assert data["modules"]["semantic_cache"]["enabled"] is True
    assert data["embedding_model"] == "voyage-4-nano"


async def test_put_config_disables_module(async_client):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.put(
            "/config",
            json={"modules": {"semantic_cache": {"enabled": False}}}
        )
    assert r.status_code == 200
    data = r.json()
    assert data["modules"]["semantic_cache"]["enabled"] is False


async def test_config_has_no_id_field(async_client):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/config")
    assert "_id" not in r.json()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/daemon/ -v
```

Expected: `ImportError: cannot import name 'app' from 'finops.daemon.app'`

- [ ] **Step 3: Write `finops/daemon/config.py`**

```python
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from finops.db.collections import CONFIG

DEFAULT_CONFIG: dict = {
    "_id": "global",
    "modules": {
        "codebase_graph":    {"enabled": True,  "repo_paths": []},
        "semantic_cache":    {"enabled": True,  "similarity_threshold": 0.92, "ttl_hours": 168},
        "agent_memory":      {"enabled": True,  "working_memory_turns": 20,
                              "episodic_ttl_days": 30, "semantic_ttl_days": 90},
        "context_compressor":{"enabled": True,  "token_threshold": 8000, "target_ratio": 4.0},
        "hybrid_retrieval":  {"enabled": False, "top_k": 5, "rrf_k": 60},
        "benchmark_runner":  {"enabled": True,  "judge_model": "claude-sonnet-4-6"},
    },
    "embedding_model":       "voyage-4-nano",
    "embedding_dimensions":  1024,
    "cost_per_input_token":  0.000003,
    "cost_per_output_token": 0.000015,
    "updated_at":            None,
}


async def load_config(db: AsyncIOMotorDatabase) -> dict:
    doc = await db[CONFIG].find_one({"_id": "global"})
    if doc is None:
        initial = {**DEFAULT_CONFIG, "updated_at": datetime.now(timezone.utc)}
        await db[CONFIG].insert_one(initial)
        return dict(initial)
    return dict(doc)


async def save_config(db: AsyncIOMotorDatabase, patch: dict) -> dict:
    await db[CONFIG].update_one(
        {"_id": "global"},
        {"$set": {**patch, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return await load_config(db)
```

- [ ] **Step 4: Write `finops/daemon/app.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from finops.db.client import get_async_db, get_sync_db
from finops.db.indexes import create_all_indexes
from finops.daemon.config import load_config, save_config

VERSION = "0.1.0"


def _check_prerequisites(sync_db) -> None:
    info = sync_db.command("buildInfo")
    major = int(info["version"].split(".")[0])
    if major < 7:
        raise SystemExit(
            f"ERROR: MongoDB >= 7.0 required, found {info['version']}.\n"
            "Run: docker run -p 27017:27017 mongodb/mongodb-atlas-local:latest"
        )
    try:
        list(sync_db["config"].list_search_indexes())
    except Exception as exc:
        raise SystemExit(
            "ERROR: MongoDB Atlas Search (mongot) not available.\n"
            "Run: docker run -p 27017:27017 mongodb/mongodb-atlas-local:latest\n"
            f"Detail: {exc}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_db = get_sync_db()
    _check_prerequisites(sync_db)
    create_all_indexes(sync_db)
    db = get_async_db()
    await load_config(db)
    yield


app = FastAPI(title="fullFinOps-AI Daemon", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}


@app.get("/config")
async def get_config():
    db = get_async_db()
    config = await load_config(db)
    config.pop("_id", None)
    return config


@app.put("/config")
async def put_config(patch: dict):
    db = get_async_db()
    config = await save_config(db, patch)
    config.pop("_id", None)
    return config
```

- [ ] **Step 5: Run tests — verify they pass**

```bash
pytest tests/daemon/ -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add finops/daemon/config.py finops/daemon/app.py \
        tests/daemon/test_health.py tests/daemon/test_config.py
git commit -m "feat: add FastAPI daemon skeleton with /health and /config endpoints"
```

---

### Task 6: CLI — `start`, `stop`, `status`

**Files:**
- Create: `finops/cli/main.py`
- Create: `tests/cli/test_cli.py`

**Interfaces:**
- Consumes: `finops.daemon.app:app` (started via uvicorn subprocess)
- Produces: `cli` Click group — entry point `finops` in `pyproject.toml`

- [ ] **Step 1: Write the failing tests**

```python
# tests/cli/test_cli.py
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from finops.cli.main import cli


def test_status_when_daemon_down():
    runner = CliRunner()
    with patch("httpx.get", side_effect=ConnectionRefusedError("refused")):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "daemon not running" in result.output


def test_start_writes_pid_file(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file), \
         patch("subprocess.Popen", return_value=mock_proc):
        result = runner.invoke(cli, ["start"])
    assert result.exit_code == 0
    assert pid_file.read_text().strip() == "12345"
    assert "12345" in result.output


def test_start_blocks_if_pid_file_exists(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("99999")
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file):
        result = runner.invoke(cli, ["start"])
    assert "already running" in result.output


def test_stop_removes_pid_file(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("12345")
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file), \
         patch("os.kill"):
        result = runner.invoke(cli, ["stop"])
    assert not pid_file.exists()
    assert "stopped" in result.output


def test_stop_when_no_daemon_running(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    runner = CliRunner()
    with patch("finops.cli.main.PID_FILE", pid_file):
        result = runner.invoke(cli, ["stop"])
    assert "No daemon running" in result.output


def test_status_shows_module_state():
    health_resp = MagicMock()
    health_resp.json.return_value = {"status": "ok", "version": "0.1.0"}
    config_resp = MagicMock()
    config_resp.json.return_value = {
        "modules": {
            "semantic_cache": {"enabled": True},
            "agent_memory":   {"enabled": False},
        }
    }
    runner = CliRunner()
    with patch("httpx.get", side_effect=[health_resp, config_resp]):
        result = runner.invoke(cli, ["status"])
    assert "daemon running" in result.output
    assert "ON" in result.output
    assert "OFF" in result.output
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/cli/test_cli.py -v
```

Expected: `ImportError: cannot import name 'cli' from 'finops.cli.main'`

- [ ] **Step 3: Write `finops/cli/main.py`**

```python
import os
import signal
import subprocess
from pathlib import Path

import click
import httpx

DAEMON_URL = os.getenv("FINOPS_DAEMON_URL", "http://localhost:7432")
PID_FILE   = Path.home() / ".finops" / "daemon.pid"


@click.group()
def cli():
    """fullFinOps-AI — token optimization toolkit."""
    pass


@cli.command()
def start():
    """Start the finops daemon in the background."""
    if PID_FILE.exists():
        click.echo("Daemon already running. Run 'finops stop' first.")
        return
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    port = os.getenv("FINOPS_PORT", "7432")
    proc = subprocess.Popen(
        ["uvicorn", "finops.daemon.app:app", "--host", "0.0.0.0", "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PID_FILE.write_text(str(proc.pid))
    click.echo(f"Daemon started (PID {proc.pid}) at {DAEMON_URL}")


@cli.command()
def stop():
    """Stop the finops daemon."""
    if not PID_FILE.exists():
        click.echo("No daemon running.")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        click.echo(f"Daemon stopped (PID {pid})")
    except ProcessLookupError:
        click.echo(f"Process {pid} not found — removing stale PID file.")
    PID_FILE.unlink(missing_ok=True)


@cli.command()
def status():
    """Show daemon health and module on/off state."""
    try:
        health = httpx.get(f"{DAEMON_URL}/health", timeout=2.0).json()
        click.echo(f"● daemon running  version={health['version']}")
        modules = httpx.get(f"{DAEMON_URL}/config", timeout=2.0).json().get("modules", {})
        for name, cfg in modules.items():
            state = "ON " if cfg.get("enabled") else "OFF"
            click.echo(f"  [{state}] {name}")
    except Exception:
        click.echo("○ daemon not running")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/cli/test_cli.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Run the full test suite**

```bash
pytest -v
```

Expected: all tests pass. Note: `test_indexes.py` tests require Docker MongoDB to be running.

- [ ] **Step 6: Smoke test the CLI end-to-end**

```bash
finops start
# wait 3 seconds for uvicorn to start
finops status
finops stop
finops status
```

Expected output:
```
Daemon started (PID <pid>) at http://localhost:7432
● daemon running  version=0.1.0
  [ON ] codebase_graph
  [ON ] semantic_cache
  [ON ] agent_memory
  [ON ] context_compressor
  [OFF] hybrid_retrieval
  [ON ] benchmark_runner
Daemon stopped (PID <pid>)
○ daemon not running
```

- [ ] **Step 7: Commit**

```bash
git add finops/cli/main.py tests/cli/test_cli.py
git commit -m "feat: add CLI start/stop/status commands"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Docker setup ✓, pyproject.toml ✓, MongoDB client ✓, all indexes ✓, BaseModule ABC ✓, daemon /health ✓, /config GET/PUT ✓, CLI start/stop/status ✓, MongoDB version check ✓, default config insertion ✓
- [x] **Placeholder scan:** No TBD/TODO in any step — all code is complete
- [x] **Type consistency:** `OptimizeRequest` and `ModuleResult` defined in Task 4 and referenced correctly; `get_async_db()` / `get_sync_db()` defined in Task 2 and used in Tasks 3, 5; `create_all_indexes(db)` defined in Task 3 and called in Task 5
- [x] **Interface chain:** Tasks 2 → 3 → 5 (client → indexes → daemon); Task 4 standalone; Task 6 depends on daemon import only

---

## What's Next

Plan 2 builds on this foundation and implements all five optimization modules plus the `/optimize` pipeline endpoint. Write Plan 2 after Plan 1's smoke test passes.
