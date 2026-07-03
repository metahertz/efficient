# fullFinOps-AI Plan 2: Optimization Modules + Strategy-Driven Pipeline (Working Version)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all five token-optimization modules (Semantic Cache, Codebase Graph, Hybrid Retrieval, Agent Memory, Context Compressor), a swappable **Strategy** abstraction that owns pipeline order / composition / cache-key / short-circuit rules, the strategy-driven `/optimize` **composing** pipeline, the loop-closing `/complete` LLM proxy (Anthropic + OpenAI), the `/metrics` endpoint, and a real `@pytest.mark.integration` suite that runs against live `mongot` with the real local embedding model — a genuinely functional optimizer, not merely mock-tested.

**Architecture:** Each module subclasses `BaseModule(ABC)` and receives an `AsyncIOMotorDatabase` + module-specific config dict at construction. A named, frozen `Strategy` (in `finops/daemon/strategies.py`) declares module `order`, `composition` mode, `cache_key` policy, and `short_circuit_on` rules; adding a strategy is a registry data entry, not a router change. The `ModulePipeline` in `finops/daemon/router.py` reads live config from MongoDB per request, selects the strategy, and runs enabled modules in the strategy's order. Augmenter modules (codebase_graph, hybrid_retrieval, agent_memory) **append** labeled sections to an accumulating context that begins from the caller's original context — they never clobber. The semantic_cache short-circuits before any composition on a hit. The compressor runs last over the assembled whole. Embeddings run **locally** via `sentence-transformers` loading the open-weight `voyageai/voyage-4-nano` (no API key). All heavy/external calls (embedding model, LLMLingua-2, LLM providers) are wrapped behind thin helpers so unit tests mock them; integration tests exercise the real model + live vector search.

**Tech Stack:** Python 3.11+, FastAPI, Motor (async MongoDB), `sentence-transformers>=5.0` (local `voyageai/voyage-4-nano`, asymmetric encode_query/encode_document), `langchain-anthropic` (fact-extraction LLM), `anthropic>=1.0` + `openai>=1.0` (`/complete` proxy providers), `llmlingua>=0.2` (context compression), `tree-sitter>=0.23` + `tree-sitter-python>=0.23` (codebase graph), `rank-bm25>=0.2` + MongoDB `$text` index (BM25 retrieval), MongoDB `$vectorSearch` (dense retrieval), pytest + pytest-asyncio (mode=auto), httpx AsyncClient + ASGITransport (daemon endpoint tests). `voyageai` HTTP SDK is an optional `[hosted]` extra (hosted upgrade path only).

## Global Constraints

- Container-only workflow: every command runs via `docker compose run --rm dev <cmd>`. Run every test command as `docker compose run --rm dev pytest ...`.
- Python ≥ 3.11; MongoDB ≥ 7.0 with mongot (Atlas Local image).
- **Embeddings run LOCALLY**: `voyageai/voyage-4-nano` via `sentence-transformers` with `trust_remote_code=True`, `truncate_dim=1024`, cosine similarity. 1024 dims — matches existing indexes, **no rebuild**. **NO API key for embeddings.** The model is **asymmetric**: use document encoding on the store side and query encoding on the lookup side.
- Single source of truth for vector geometry: `EMBEDDING_DIMENSIONS = 1024` and `VECTOR_SIMILARITY = "cosine"` in `finops/db/indexes.py`.
- Token counts in all `ModuleResult` fields are approximate (`len(text) // 4`). Do not add tiktoken.
- **Honest metrics.** `ModuleResult` distinguishes reduction from augmentation via `short_circuit`, `tokens_added`, and `baseline_tokens`. Reducers (compressor) report `tokens_saved = before − after`. Augmenters report `tokens_saved` against the naive baseline they replace and record `tokens_added` when they grow the payload. No clamped-to-zero fiction beyond `max(0, …)` on the honest delta.
- **Modules compose, never clobber.** Each augmenter appends a labeled section (`## Relevant Code`, `## Retrieved Docs`, `## Memory`) to an accumulating context that begins from the caller's original context. The compressor runs last over the assembled whole.
- **Cache key policy.** The cache keys on the incoming **prompt**, never the assembled/compressed context. Default `prompt+scope` (includes `agent_id`/`corpus_id`); a strategy may override to `prompt`-only for max hit rate.
- Unit tests mock all heavy/external calls (embedding model, LLMLingua-2, LLM providers). No real API keys or model downloads required for the unit gate. Integration tests (marked `@pytest.mark.integration`) use the REAL local model + live mongot.
- All new code follows the existing no-comments style (no docstrings, no inline comments unless WHY is non-obvious from code alone).
- Test DB name: `finops_test` (set by `set_test_env` fixture in `tests/conftest.py`).
- Test MongoDB URI: `mongodb://localhost:27017/?directConnection=true` (or `FINOPS_TEST_MONGODB_URI` env var, `mongodb://mongodb:27017` inside the container).
- `pytest-asyncio` mode is `auto` — async test functions are collected without `@pytest.mark.asyncio`.
- The `integration` marker is registered in `pyproject.toml`. Unit runs exclude integration via `-m "not integration"`.

---

## Task 1: Foundation — Local Embeddings, Deps, Plan-1 Fixes, Honest Metrics, Test Infra

Swaps the embedding stack to the local `sentence-transformers` model (no API key), adds the strategy/provider dependencies, fixes three carry-over Plan-1 issues (M1 lazy daemon URL, M2 name enforcement, M3 multi-stage Dockerfile), upgrades `ModuleResult` to honest-metric fields, adds pre-filter paths to the vector indexes, wires a HuggingFace model-cache volume + `finops warmup` command, and adds the async test fixtures + `integration` marker every later task depends on.

**Files:**
- Modify: `pyproject.toml` — deps: add `sentence-transformers>=5.0`, `openai>=1.0`, `tree-sitter-python>=0.23`; move `voyageai` to `[hosted]` extra; register `integration` marker.
- Create: `finops/modules/embeddings.py` — lazy singleton `SentenceTransformer` + asymmetric helpers.
- Modify: `finops/modules/_base.py` — `ModuleResult` honest-metric fields; `BaseModule.__init__` name enforcement.
- Modify: `finops/cli/main.py` — lazy `_daemon_url()` (M1); add `finops warmup` command.
- Modify: `finops/db/indexes.py` — `filter_paths` param on `_create_vector_index`; per-collection filter paths.
- Modify: `Dockerfile` — multi-stage `base` + `dev` targets (M3).
- Modify: `docker-compose.yml` — `target:` per service (M3); `hf_cache` named volume + `HF_HOME` env (M3 + local model cache).
- Modify: `tests/conftest.py` — add `finops_db` async fixture + `wait_for_queryable` helper.
- Modify: `tests/modules/test_base.py` — tests for name enforcement + new `ModuleResult` fields.
- Create: `tests/modules/test_embeddings.py` — mocked-model helper tests.

**Interfaces:**
- Produces:
  - `finops.modules.embeddings.embed_documents(texts: list[str]) -> list[list[float]]` — store side (document encoding), 1024-float lists.
  - `finops.modules.embeddings.embed_query(text: str) -> list[float]` — lookup side (query encoding), 1024-float list.
  - `finops.modules.embeddings.reset_model() -> None` — test helper.
  - `finops.modules.embeddings._get_model() -> SentenceTransformer` — lazy singleton (warmup + monkeypatch target).
  - `ModuleResult.short_circuit: bool = False`, `ModuleResult.tokens_added: int = 0`, `ModuleResult.baseline_tokens: int = 0`.
  - `finops_db` pytest fixture — `AsyncIOMotorDatabase` with all Atlas Search indexes pre-created.
  - `tests.conftest.wait_for_queryable(collection, index_name, timeout=90)` — polls `list_search_indexes()` for `queryable is True`.

- [ ] **Step 1: Write failing tests for M2 name enforcement + new ModuleResult fields**

Add to `tests/modules/test_base.py`:
```python
def test_subclass_without_name_raises():
    class Unnamed(BaseModule):
        async def process(self, request):
            return request, None
        def is_enabled(self):
            return True
    with pytest.raises(TypeError, match="must define a non-empty 'name'"):
        Unnamed()


def test_module_result_has_honest_metric_defaults():
    r = ModuleResult(
        module="x", tokens_in=0, tokens_out=0,
        tokens_saved=0, latency_ms=0.0, detail="",
    )
    assert r.short_circuit is False
    assert r.tokens_added == 0
    assert r.baseline_tokens == 0


def test_module_result_honest_metric_fields_can_be_set():
    r = ModuleResult(
        module="x", tokens_in=0, tokens_out=0,
        tokens_saved=0, latency_ms=0.0, detail="",
        short_circuit=True, tokens_added=42, baseline_tokens=1000,
    )
    assert r.short_circuit is True
    assert r.tokens_added == 42
    assert r.baseline_tokens == 1000
```

Ensure the imports at the top of `tests/modules/test_base.py` include:
```python
import pytest
from finops.modules._base import BaseModule, ModuleResult, OptimizeRequest
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_base.py -v 2>&1 | tail -20
```
Expected: the 3 new tests FAIL (TypeError not raised for `Unnamed`; `ModuleResult` rejects `short_circuit`/`tokens_added`/`baseline_tokens` kwargs with `TypeError: __init__() got an unexpected keyword argument`).

- [ ] **Step 3: Replace `finops/modules/_base.py` — honest metrics + name enforcement**

Replace the entire file `finops/modules/_base.py`:
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
    module:          str
    tokens_in:       int
    tokens_out:      int
    tokens_saved:    int
    latency_ms:      float
    detail:          str
    short_circuit:   bool = field(default=False)
    tokens_added:    int = field(default=0)
    baseline_tokens: int = field(default=0)


class BaseModule(ABC):
    name: str = ""

    def __init__(self):
        if not self.__class__.name:
            raise TypeError(
                f"{self.__class__.__name__} must define a non-empty 'name' class attribute"
            )

    @abstractmethod
    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        ...
```

- [ ] **Step 4: Run base tests — all should pass**

```bash
docker compose run --rm dev pytest tests/modules/test_base.py -v 2>&1 | tail -15
```
Expected: all base tests PASS (existing + 3 new).

- [ ] **Step 5: Update `pyproject.toml` — deps, extras, marker**

Replace the `dependencies` list, `[project.optional-dependencies]`, and `[tool.pytest.ini_options]` blocks in `pyproject.toml`:
```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "motor>=3.6",
    "pymongo>=4.8",
    "click>=8.1",
    "httpx>=0.27",
    "sentence-transformers>=5.0",
    "langchain-mongodb>=0.3",
    "langchain-anthropic>=0.3",
    "langchain-core>=0.3",
    "python-dotenv>=1.0",
    "tree-sitter>=0.23",
    "tree-sitter-python>=0.23",
    "llmlingua>=0.2",
    "rank-bm25>=0.2",
    "datasets>=2.20",
    "anthropic>=0.40",
    "openai>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "anyio>=4.0",
]
hosted = [
    "voyageai>=0.3",
]

[project.scripts]
finops = "finops.cli.main:cli"

[tool.hatch.build.targets.wheel]
packages = ["finops"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
    "integration: real mongot + real local embedding model (slow, downloads model on first run)",
]
```
Note: `anthropic>=0.40` already satisfies the `>=1.0` intent for the `/complete` provider (the installed line stays `>=0.40`; the async client used in Task 9 is available in that range). `openai>=1.0` is new.

- [ ] **Step 6: Create `finops/modules/embeddings.py` (local model, asymmetric helpers)**

```python
import os
from sentence_transformers import SentenceTransformer

_MODEL_ID = "voyageai/voyage-4-nano"
_DIM = 1024
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        name = os.getenv("FINOPS_EMBEDDING_MODEL", _MODEL_ID)
        _model = SentenceTransformer(name, trust_remote_code=True, truncate_dim=_DIM)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode_document(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vec = model.encode_query([text], normalize_embeddings=True)[0]
    return vec.tolist()


def reset_model() -> None:
    global _model
    _model = None
```

- [ ] **Step 7: VERIFY the installed sentence-transformers exposes `encode_query` / `encode_document`; fall back if not**

Run this probe inside the container (this also triggers the first model download; see warmup + hf_cache below):
```bash
docker compose run --rm dev python -c "from sentence_transformers import SentenceTransformer; print(hasattr(SentenceTransformer, 'encode_query'), hasattr(SentenceTransformer, 'encode_document'))"
```
Expected: `True True`.

If the probe prints `False False` (older sentence-transformers without the asymmetric methods), edit `finops/modules/embeddings.py` and replace ONLY the two helper bodies with the `prompt_name` variant below (keep `_get_model` and `reset_model` unchanged):
```python
def embed_documents(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    vecs = model.encode(texts, prompt_name="document", normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vec = model.encode([text], prompt_name="query", normalize_embeddings=True)[0]
    return vec.tolist()
```
Otherwise keep the `encode_document` / `encode_query` bodies from Step 6. Do not change the public signatures either way — all later tasks import `embed_documents` / `embed_query`.

- [ ] **Step 8: Write embeddings unit tests (model mocked via `_get_model`)**

Create `tests/modules/test_embeddings.py`:
```python
import numpy as np
from unittest.mock import MagicMock
import finops.modules.embeddings as emb


def _fake_model():
    m = MagicMock()
    m.encode_document.side_effect = lambda texts, **kw: [np.full(1024, 0.2) for _ in texts]
    m.encode_query.side_effect = lambda texts, **kw: [np.full(1024, 0.3) for _ in texts]
    return m


def test_embed_documents_returns_list_of_1024_floats(monkeypatch):
    monkeypatch.setattr(emb, "_get_model", _fake_model)
    out = emb.embed_documents(["a", "b"])
    assert len(out) == 2
    assert len(out[0]) == 1024
    assert isinstance(out[0][0], float)


def test_embed_query_returns_1024_floats(monkeypatch):
    monkeypatch.setattr(emb, "_get_model", _fake_model)
    out = emb.embed_query("hello")
    assert len(out) == 1024
    assert isinstance(out[0], float)


def test_embed_query_and_documents_use_distinct_encoders(monkeypatch):
    m = _fake_model()
    monkeypatch.setattr(emb, "_get_model", lambda: m)
    emb.embed_documents(["doc"])
    emb.embed_query("qry")
    m.encode_document.assert_called_once()
    m.encode_query.assert_called_once()


def test_reset_model_clears_singleton():
    emb._model = object()
    emb.reset_model()
    assert emb._model is None
```

```bash
docker compose run --rm dev pytest tests/modules/test_embeddings.py -v 2>&1 | tail -15
```
Expected: 4 tests PASS.

- [ ] **Step 9: Fix M1 — lazy `_daemon_url()` + add `finops warmup` in `finops/cli/main.py`**

Replace lines 1–10 of `finops/cli/main.py`:
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

In `start()`, replace the final echo line:
```python
    click.echo(f"Daemon started (PID {proc.pid}) at {_daemon_url()}")
```

In `status()`, replace the two `DAEMON_URL` references:
```python
        health = httpx.get(f"{_daemon_url()}/health", timeout=2.0).json()
        click.echo(f"● daemon running  version={health['version']}")
        modules = httpx.get(f"{_daemon_url()}/config", timeout=2.0).json().get("modules", {})
```

Add a `warmup` command before the `if __name__ == "__main__":` block. It fetches BOTH the embedding model and the LLMLingua-2 compressor model once (so the hf_cache volume is populated and later runs are fast). The compressor import is inside the function so `finops --help` stays cheap:
```python
@cli.command()
def warmup():
    """Download and cache local models (embeddings + compressor)."""
    from finops.modules.embeddings import _get_model
    click.echo("Loading embedding model (voyageai/voyage-4-nano)...")
    _get_model()
    click.echo("  embedding model ready.")
    from finops.modules.context_compressor import _get_compressor
    click.echo("Loading compressor model (LLMLingua-2)...")
    _get_compressor()
    click.echo("  compressor model ready.")
    click.echo("Warmup complete.")
```
Note: `_get_compressor` is created in Task 7. Until Task 7 lands, `finops warmup` will raise `ImportError` on the compressor line — that is expected and resolved when Task 7 completes. The embedding half works immediately.

- [ ] **Step 10: Run CLI tests to verify M1 fix (no regressions)**

```bash
docker compose run --rm dev pytest tests/cli/ -v 2>&1 | tail -15
```
Expected: all CLI tests PASS.

- [ ] **Step 11: Fix M3 — multi-stage Dockerfile**

Replace the entire `Dockerfile`:
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

- [ ] **Step 12: Update `docker-compose.yml` — targets, hf_cache volume, HF_HOME**

Replace the entire `docker-compose.yml`:
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
      timeout: 10s
      retries: 10
      start_period: 30s

  dev:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev
    volumes:
      - .:/workspace
      - hf_cache:/root/.cache/huggingface
    working_dir: /workspace
    command: sleep infinity
    environment:
      - FINOPS_MONGODB_URI=mongodb://mongodb:27017
      - FINOPS_DB_NAME=finops
      - FINOPS_TEST_MONGODB_URI=mongodb://mongodb:27017
      - HF_HOME=/root/.cache/huggingface
    depends_on:
      mongodb:
        condition: service_healthy

  daemon:
    build:
      context: .
      dockerfile: Dockerfile
      target: base
    ports:
      - "7432:7432"
    volumes:
      - hf_cache:/root/.cache/huggingface
    environment:
      - FINOPS_MONGODB_URI=mongodb://mongodb:27017
      - FINOPS_DB_NAME=finops
      - HF_HOME=/root/.cache/huggingface
    depends_on:
      mongodb:
        condition: service_healthy

volumes:
  mongodb_data:
  hf_cache:
```

- [ ] **Step 13: Update `finops/db/indexes.py` — `filter_paths` support + per-collection filters**

Replace `_create_vector_index` and update the calls inside `create_all_indexes`. Full replacement of `finops/db/indexes.py`:
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


def _create_vector_index(
    collection, name: str, field: str = "embedding",
    filter_paths: list[str] | None = None,
) -> None:
    if _search_index_exists(collection, name):
        return
    fields = [{
        "type": "vector",
        "path": field,
        "numDimensions": EMBEDDING_DIMENSIONS,
        "similarity": VECTOR_SIMILARITY,
    }]
    for path in (filter_paths or []):
        fields.append({"type": "filter", "path": path})
    collection.create_search_index({
        "name": name,
        "type": "vectorSearch",
        "definition": {"fields": fields},
    })


def create_all_indexes(db: Database) -> None:
    col = db[CODEBASE_NODES]
    col.create_index([("repo_id", ASCENDING), ("symbol", ASCENDING)])
    col.create_index([("repo_id", ASCENDING), ("file_path", ASCENDING)])
    _create_vector_index(col, "codebase_vector_index", filter_paths=["repo_id"])

    col = db[CACHE_ENTRIES]
    col.create_index([("prompt_hash", ASCENDING)], unique=True)
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "cache_vector_index")

    col = db[WORKING_MEMORY]
    col.create_index([("agent_id", ASCENDING), ("session_id", ASCENDING)])

    col = db[EPISODIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "episodic_vector_index", filter_paths=["agent_id"])

    col = db[SEMANTIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "semantic_vector_index", filter_paths=["agent_id"])

    col = db[COMPRESSION_STATS]
    col.create_index([("created_at", ASCENDING)])

    col = db[CORPUS_CHUNKS]
    col.create_index([("corpus_id", ASCENDING)])
    col.create_index([("bm25_tokens", TEXT)])
    _create_vector_index(col, "corpus_vector_index", filter_paths=["corpus_id"])

    col = db[BENCHMARK_RUNS]
    col.create_index([("started_at", ASCENDING)])
```

- [ ] **Step 14: Add `finops_db` fixture + `wait_for_queryable` helper to `tests/conftest.py`**

Append to `tests/conftest.py`:
```python
import time


@pytest.fixture
async def finops_db(async_client, sync_db):
    from finops.db.indexes import create_all_indexes
    create_all_indexes(sync_db)
    yield async_client[os.environ["FINOPS_DB_NAME"]]


def wait_for_queryable(collection, index_name, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for idx in collection.list_search_indexes():
            if idx["name"] == index_name and idx.get("queryable") is True:
                return
        time.sleep(1)
    raise TimeoutError(f"index {index_name} not queryable within {timeout}s")
```
Note: `finops_db` uses the async `async_client` for the yielded DB and the sync `sync_db` for index creation + teardown. `wait_for_queryable` takes a **sync** collection (`sync_db[NAME]`) so integration tests can poll before issuing async queries.

- [ ] **Step 15: Run the full unit suite — must be green**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -12
```
Expected: all unit tests PASS (Plan-1 28 + base/embeddings additions), 0 fail, 0 integration collected.

- [ ] **Step 16: Warm up the local models once (populates hf_cache volume)**

The embedding half works now; the compressor half becomes available after Task 7. Run the embedding warm-up now to seed the cache:
```bash
docker compose run --rm dev python -c "from finops.modules.embeddings import _get_model; _get_model(); print('embedding model cached')"
```
Expected: model downloads once, prints `embedding model cached`. Subsequent container runs reuse `hf_cache`.

- [ ] **Step 17: Commit**

```bash
git add pyproject.toml finops/modules/embeddings.py finops/modules/_base.py \
        finops/cli/main.py finops/db/indexes.py Dockerfile docker-compose.yml \
        tests/conftest.py tests/modules/test_base.py tests/modules/test_embeddings.py
git commit -m "feat: local embeddings, honest ModuleResult, M1/M2/M3 fixes, hf_cache, warmup, test infra"
```

---

## Task 2: Strategy Abstraction + `config.strategy`

Introduces the swappable `Strategy` dataclass that owns pipeline order, composition mode, cache-key policy, short-circuit rules, and per-module overrides. Two built-ins ship (`compose_then_compress` default; `cache_first_aggressive`). Adding a third is a registry data entry. `DEFAULT_CONFIG` gains a top-level `strategy` key.

**Files:**
- Create: `finops/daemon/strategies.py`
- Modify: `finops/daemon/config.py` — add `"strategy": "compose_then_compress"` to `DEFAULT_CONFIG`
- Create: `tests/daemon/test_strategies.py`

**Interfaces:**
- Consumes: nothing (pure data).
- Produces:
  - `finops.daemon.strategies.Strategy` — frozen dataclass `(name, order, composition, cache_key, short_circuit_on, overrides)`.
  - `finops.daemon.strategies.get_strategy(name: str | None) -> Strategy` — returns default for unknown/None.
  - `finops.daemon.strategies.list_strategies() -> list[str]`.
  - `COMPOSE_THEN_COMPRESS`, `CACHE_FIRST_AGGRESSIVE` module-level Strategy constants.

- [ ] **Step 1: Write failing tests**

Create `tests/daemon/test_strategies.py`:
```python
import pytest
import dataclasses
from finops.daemon.strategies import (
    Strategy, get_strategy, list_strategies,
    COMPOSE_THEN_COMPRESS, CACHE_FIRST_AGGRESSIVE,
)


def test_get_strategy_default_for_none():
    assert get_strategy(None) is COMPOSE_THEN_COMPRESS


def test_get_strategy_default_for_unknown():
    assert get_strategy("does-not-exist") is COMPOSE_THEN_COMPRESS


def test_get_strategy_returns_named():
    assert get_strategy("cache_first_aggressive") is CACHE_FIRST_AGGRESSIVE


def test_both_builtins_listed():
    names = list_strategies()
    assert "compose_then_compress" in names
    assert "cache_first_aggressive" in names


def test_strategy_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        COMPOSE_THEN_COMPRESS.name = "x"


def test_default_strategy_order_and_policy():
    s = COMPOSE_THEN_COMPRESS
    assert s.order[0] == "semantic_cache"
    assert s.order[-1] == "context_compressor"
    assert s.composition == "compose"
    assert s.cache_key == "prompt+scope"
    assert s.short_circuit_on == ("semantic_cache",)


def test_aggressive_strategy_overrides():
    s = CACHE_FIRST_AGGRESSIVE
    assert s.cache_key == "prompt"
    assert "context_compressor" not in s.order
    assert s.overrides["semantic_cache"]["similarity_threshold"] == 0.85


def test_config_default_has_strategy():
    from finops.daemon.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["strategy"] == "compose_then_compress"
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/daemon/test_strategies.py -v 2>&1 | tail -15
```
Expected: ImportError (module does not exist) / `test_config_default_has_strategy` KeyError.

- [ ] **Step 3: Create `finops/daemon/strategies.py`**

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Strategy:
    name:             str
    order:            tuple[str, ...]
    composition:      str
    cache_key:        str
    short_circuit_on: tuple[str, ...] = ()
    overrides:        dict = field(default_factory=dict)


COMPOSE_THEN_COMPRESS = Strategy(
    name="compose_then_compress",
    order=("semantic_cache", "codebase_graph", "hybrid_retrieval", "agent_memory", "context_compressor"),
    composition="compose",
    cache_key="prompt+scope",
    short_circuit_on=("semantic_cache",),
)

CACHE_FIRST_AGGRESSIVE = Strategy(
    name="cache_first_aggressive",
    order=("semantic_cache", "codebase_graph", "hybrid_retrieval", "agent_memory"),
    composition="compose",
    cache_key="prompt",
    short_circuit_on=("semantic_cache",),
    overrides={"semantic_cache": {"similarity_threshold": 0.85}},
)

_REGISTRY = {s.name: s for s in (COMPOSE_THEN_COMPRESS, CACHE_FIRST_AGGRESSIVE)}


def get_strategy(name: str | None) -> Strategy:
    return _REGISTRY.get(name or "", COMPOSE_THEN_COMPRESS)


def list_strategies() -> list[str]:
    return list(_REGISTRY)
```

- [ ] **Step 4: Add `strategy` to `DEFAULT_CONFIG` in `finops/daemon/config.py`**

In `finops/daemon/config.py`, add a top-level `"strategy"` key to `DEFAULT_CONFIG` immediately after the closing brace of `"modules"` (before `"embedding_model"`):
```python
    "strategy":              "compose_then_compress",
    "embedding_model":       "voyage-4-nano",
```

- [ ] **Step 5: Run tests**

```bash
docker compose run --rm dev pytest tests/daemon/test_strategies.py -v 2>&1 | tail -15
```
Expected: 8 tests PASS.

- [ ] **Step 6: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add finops/daemon/strategies.py finops/daemon/config.py tests/daemon/test_strategies.py
git commit -m "feat: add swappable Strategy abstraction and config.strategy"
```

---

## Task 3: Semantic Cache Module + `/cache/lookup` + `/cache/store`

Two-layer cache (SHA-256 exact match then vector similarity) driven by a `cache_key` policy. On a hit the module short-circuits (terminal): `short_circuit=True`, context replaced by the cached response, `tokens_saved` = the stored savings. `store()` upserts entries idempotently with a TTL. Lookups use `embed_query`; stores use `embed_documents([...])[0]` (asymmetric). `/cache/lookup` (GET) and `/cache/store` (POST) expose the two sides directly.

**Files:**
- Create: `finops/modules/semantic_cache.py`
- Modify: `finops/daemon/app.py` — add `GET /cache/lookup` and `POST /cache/store`
- Create: `tests/modules/test_semantic_cache.py`
- Create: `tests/daemon/test_cache_endpoints.py`

**Interfaces:**
- Consumes: `finops.modules.embeddings.embed_query`, `finops.modules.embeddings.embed_documents`, `finops.db.collections.CACHE_ENTRIES`, `BaseModule`, `OptimizeRequest`, `ModuleResult`.
- Produces:
  - `SemanticCache(db: AsyncIOMotorDatabase, config: dict)` — `config` may include `similarity_threshold`, `ttl_hours`, `cache_key`.
  - `SemanticCache._key_material(request) -> str` — builds the keyed string per policy.
  - `await cache.process(request) -> (OptimizeRequest, ModuleResult)` — on HIT sets `short_circuit=True`, replaces context with cached response.
  - `await cache.store(prompt, response, framework, model, tokens_saved, agent_id="", corpus_id="") -> None` — idempotent upsert with TTL.

- [ ] **Step 1: Write failing tests**

Create `tests/modules/test_semantic_cache.py`:
```python
import hashlib
import pytest
from datetime import datetime, timezone
from finops.modules.semantic_cache import SemanticCache
from finops.modules._base import OptimizeRequest
from finops.db.collections import CACHE_ENTRIES

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.semantic_cache.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.semantic_cache.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
def config():
    return {"similarity_threshold": 0.92, "ttl_hours": 168, "cache_key": "prompt+scope"}


@pytest.fixture
async def cache(finops_db, config):
    return SemanticCache(finops_db, config)


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


async def test_exact_hit_replaces_context_and_short_circuits(cache, finops_db, req):
    key = cache._key_material(req)
    prompt_hash = hashlib.sha256(key.encode()).hexdigest()
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
    assert new_req.context == "Python is a language"
    assert result.tokens_saved == 500
    assert result.short_circuit is True
    assert "exact" in result.detail
    doc = await finops_db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    assert doc["hit_count"] == 1


async def test_store_writes_entry(cache, finops_db, req):
    await cache.store(prompt=req.prompt, response="Python is great", framework="test",
                      model="claude", tokens_saved=200, agent_id=req.agent_id, corpus_id=req.corpus_id)
    key = cache._key_material(req)
    prompt_hash = hashlib.sha256(key.encode()).hexdigest()
    doc = await finops_db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    assert doc is not None
    assert doc["response"] == "Python is great"
    assert doc["tokens_saved"] == 200
    assert doc["hit_count"] == 0


async def test_store_is_idempotent(cache, finops_db, req):
    await cache.store(req.prompt, "resp", "test", "m", 100, agent_id=req.agent_id, corpus_id=req.corpus_id)
    await cache.store(req.prompt, "resp", "test", "m", 100, agent_id=req.agent_id, corpus_id=req.corpus_id)
    key = cache._key_material(req)
    prompt_hash = hashlib.sha256(key.encode()).hexdigest()
    count = await finops_db[CACHE_ENTRIES].count_documents({"prompt_hash": prompt_hash})
    assert count == 1


async def test_prompt_plus_scope_differs_by_corpus(finops_db):
    cache = SemanticCache(finops_db, {"cache_key": "prompt+scope"})
    r1 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c1")
    r2 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c2")
    assert cache._key_material(r1) != cache._key_material(r2)


async def test_prompt_only_ignores_corpus(finops_db):
    cache = SemanticCache(finops_db, {"cache_key": "prompt"})
    r1 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c1")
    r2 = OptimizeRequest(prompt="same", context="", agent_id="a", framework="f", corpus_id="c2")
    assert cache._key_material(r1) == cache._key_material(r2)
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_semantic_cache.py -v 2>&1 | tail -15
```
Expected: ImportError (module does not exist yet).

- [ ] **Step 3: Create `finops/modules/semantic_cache.py`**

```python
import hashlib
import time
from datetime import datetime, timezone, timedelta

from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_query, embed_documents
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
        self._cache_key = config.get("cache_key", "prompt+scope")

    def is_enabled(self) -> bool:
        return True

    def _key_material(self, request: OptimizeRequest) -> str:
        if self._cache_key == "prompt":
            return request.prompt
        return f"{request.prompt}|agent={request.agent_id}|corpus={request.corpus_id or ''}"

    def _key_material_raw(self, prompt: str, agent_id: str, corpus_id: str) -> str:
        if self._cache_key == "prompt":
            return prompt
        return f"{prompt}|agent={agent_id}|corpus={corpus_id or ''}"

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()
        key = self._key_material(request)
        prompt_hash = hashlib.sha256(key.encode()).hexdigest()

        entry = await self._db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
        if entry:
            await self._db[CACHE_ENTRIES].update_one(
                {"_id": entry["_id"]},
                {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}},
            )
            saved = entry.get("tokens_saved", 0)
            cached_req = OptimizeRequest(
                prompt=request.prompt,
                context=entry["response"],
                agent_id=request.agent_id,
                framework=request.framework,
                corpus_id=request.corpus_id,
            )
            return cached_req, ModuleResult(
                module=self.name,
                tokens_in=saved,
                tokens_out=0,
                tokens_saved=saved,
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail="exact hash hit",
                short_circuit=True,
                baseline_tokens=saved,
            )

        embedding = embed_query(key)
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
            saved = doc.get("tokens_saved", 0)
            cached_req = OptimizeRequest(
                prompt=request.prompt,
                context=doc["response"],
                agent_id=request.agent_id,
                framework=request.framework,
                corpus_id=request.corpus_id,
            )
            return cached_req, ModuleResult(
                module=self.name,
                tokens_in=saved,
                tokens_out=0,
                tokens_saved=saved,
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"semantic hit (similarity={score:.3f})",
                short_circuit=True,
                baseline_tokens=saved,
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
        agent_id: str = "",
        corpus_id: str = "",
    ) -> None:
        key = self._key_material_raw(prompt, agent_id, corpus_id)
        prompt_hash = hashlib.sha256(key.encode()).hexdigest()
        embedding = embed_documents([key])[0]
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self._ttl_hours)
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

- [ ] **Step 4: Run module tests**

```bash
docker compose run --rm dev pytest tests/modules/test_semantic_cache.py -v 2>&1 | tail -15
```
Expected: 6 tests PASS. (The exact-hit and key-policy tests exercise the hash path + store; they do not require a warm `$vectorSearch` index — semantic-similarity behavior is covered in the Task 11 integration suite.)

- [ ] **Step 5: Add `GET /cache/lookup` and `POST /cache/store` to `finops/daemon/app.py`**

Add `CACHE_ENTRIES` to the imports at the top of `app.py`:
```python
from finops.db.collections import CACHE_ENTRIES
```

Add after the `PUT /config` endpoint:
```python
@app.get("/cache/lookup")
async def cache_lookup(prompt_hash: str, embedding: list[float] | None = None):
    db = get_async_db()
    config = await load_config(db)
    cache_cfg = config.get("modules", {}).get("semantic_cache", {})
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


@app.post("/cache/store")
async def cache_store(body: dict):
    db = get_async_db()
    config = await load_config(db)
    cache_cfg = config.get("modules", {}).get("semantic_cache", {})
    strategy = get_strategy(config.get("strategy"))
    cache_cfg = {**cache_cfg, "cache_key": strategy.cache_key}
    from finops.modules.semantic_cache import SemanticCache
    cache = SemanticCache(db, cache_cfg)
    await cache.store(
        prompt=body.get("prompt", ""),
        response=body.get("response", ""),
        framework=body.get("framework", "unknown"),
        model=body.get("model", ""),
        tokens_saved=int(body.get("tokens_saved", 0)),
        agent_id=body.get("agent_id", ""),
        corpus_id=body.get("corpus_id", ""),
    )
    return {"stored": True}
```

Add `get_strategy` to the imports at the top of `app.py`:
```python
from finops.daemon.strategies import get_strategy
```

- [ ] **Step 6: Write and run cache endpoint tests**

Create `tests/daemon/test_cache_endpoints.py`:
```python
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
```

```bash
docker compose run --rm dev pytest tests/daemon/test_cache_endpoints.py -v 2>&1 | tail -12
```
Expected: 3 tests PASS.

- [ ] **Step 7: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add finops/modules/semantic_cache.py finops/daemon/app.py \
        tests/modules/test_semantic_cache.py tests/daemon/test_cache_endpoints.py
git commit -m "feat: semantic cache module with cache_key policy + /cache/lookup + /cache/store"
```

---

## Task 4: Codebase Graph Module (composing augmenter)

Parses Python source with Tree-Sitter, stores symbol nodes + document embeddings in `codebase_nodes`. Query by natural language or symbol name returns minimal code slices. `process` **appends** a `## Relevant Code` section to `request.context` (never overwrites). Honest metrics: `baseline_tokens` = token count of ALL indexed symbols for the repo (a documented proxy for the "read the whole codebase" alternative, since only snippets are stored), `tokens_added` = tokens of injected snippets, `tokens_saved = max(0, baseline_tokens - tokens_added)`. No-op (unchanged request) when no repo configured or no match.

**Files:**
- Create: `finops/modules/codebase_graph.py`
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/sample.py`
- Create: `tests/modules/test_codebase_graph.py`

**Interfaces:**
- Consumes: `embed_query`, `embed_documents`, `CODEBASE_NODES`, `BaseModule`, `OptimizeRequest`, `ModuleResult`.
- Produces:
  - `CodebaseGraph(db: AsyncIOMotorDatabase, config: dict)` — `config` may include `repo_paths`.
  - `await graph.index_file(repo_id: str, file_path: str, source: str) -> int` — count of symbols indexed.
  - `await graph.query(repo_id: str, query_text: str, k: int = 5) -> list[dict]`.
  - `await graph.process(request) -> (OptimizeRequest, ModuleResult)` — appends `## Relevant Code`; no-op if no repo/match.

- [ ] **Step 1: Create test fixture**

Create `tests/fixtures/__init__.py` (empty file).

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

- [ ] **Step 2: Write failing tests**

Create `tests/modules/test_codebase_graph.py`:
```python
import pytest
from pathlib import Path
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules._base import OptimizeRequest
from finops.db.collections import CODEBASE_NODES

FIXED_EMBEDDING = [0.1] * 1024
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "sample.py"


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.codebase_graph.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.codebase_graph.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
async def graph(finops_db):
    return CodebaseGraph(finops_db, {"repo_paths": []})


@pytest.fixture
def sample_source():
    return FIXTURE_PATH.read_text()


async def test_index_file_returns_symbol_count(graph, sample_source):
    count = await graph.index_file("repo1", "sample.py", sample_source)
    assert count >= 3


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
    req = OptimizeRequest(prompt="find add function", context="orig", agent_id="a1", framework="test")
    new_req, result = await graph.process(req)
    assert new_req is req
    assert new_req.context == "orig"
    assert result.tokens_saved == 0
    assert result.short_circuit is False
```

- [ ] **Step 3: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_codebase_graph.py -v 2>&1 | tail -15
```
Expected: ImportError (module does not exist yet).

- [ ] **Step 4: Create `finops/modules/codebase_graph.py`**

```python
import time
from datetime import datetime, timezone
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_query, embed_documents
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
        repo_id = self._repo_paths[0]
        results = await self.query(repo_id, request.prompt)
        if not results:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=(time.perf_counter() - t0) * 1000, detail="no symbols matched",
            )
        snippets = "\n\n".join(
            f"# {r['file_path']}:{r['line_start']}\n{r['source_snippet']}"
            for r in results
        )
        baseline_tokens = await self._repo_symbol_tokens(repo_id)
        tokens_added = _count_tokens(snippets)
        tokens_in = _count_tokens(request.context)
        section = "## Relevant Code\n" + snippets
        new_context = request.context + ("\n\n" if request.context else "") + section
        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=new_context,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=_count_tokens(new_context),
            tokens_saved=max(0, baseline_tokens - tokens_added),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"injected {len(results)} symbols (baseline={baseline_tokens} full-index tokens)",
            tokens_added=tokens_added,
            baseline_tokens=baseline_tokens,
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
        embeddings = embed_documents(snippets)
        now = datetime.now(timezone.utc)
        for sym, emb in zip(symbols, embeddings):
            await self._db[CODEBASE_NODES].update_one(
                {"repo_id": repo_id, "symbol": sym["symbol"], "file_path": file_path},
                {"$set": {**sym, "embedding": emb, "indexed_at": now}},
                upsert=True,
            )
        return len(symbols)

    async def query(self, repo_id: str, query_text: str, k: int = 5) -> list[dict]:
        embedding = embed_query(query_text)
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
            {"$project": {"embedding": 0}},
        ]
        results = []
        async for doc in self._db[CODEBASE_NODES].aggregate(pipeline):
            results.append(doc)
        return results

    async def _repo_symbol_tokens(self, repo_id: str) -> int:
        total = 0
        async for doc in self._db[CODEBASE_NODES].find(
            {"repo_id": repo_id}, {"source_snippet": 1}
        ):
            total += _count_tokens(doc.get("source_snippet", ""))
        return total
```

- [ ] **Step 5: Run module tests**

```bash
docker compose run --rm dev pytest tests/modules/test_codebase_graph.py -v 2>&1 | tail -15
```
Expected: 5 tests PASS. (`process` composition/append behavior with a live `$vectorSearch` result is validated in the Task 11 integration suite; here we verify indexing, metadata capture, and the no-repo no-op.)

- [ ] **Step 6: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add finops/modules/codebase_graph.py tests/fixtures/__init__.py \
        tests/fixtures/sample.py tests/modules/test_codebase_graph.py
git commit -m "feat: codebase graph module (Tree-Sitter, composing append, honest baseline)"
```

---

## Task 5: Hybrid Retrieval Module (composing augmenter)

BM25 via MongoDB `$text` index + dense via `$vectorSearch` + RRF fusion over `corpus_chunks`. `add_chunks` embeds with `embed_documents`; `_dense_search` embeds with `embed_query`. `process` **appends** a `## Retrieved Docs` section to `request.context` (never overwrites). Honest metrics: `baseline_tokens` = total tokens of ALL chunks in the corpus (full-corpus alternative), `tokens_added` = tokens of the top-k retrieved, `tokens_saved = max(0, baseline - added)`. No-op when no `corpus_id`.

**Files:**
- Create: `finops/modules/hybrid_retrieval.py`
- Create: `tests/modules/test_hybrid_retrieval.py`

**Interfaces:**
- Consumes: `embed_query`, `embed_documents`, `CORPUS_CHUNKS`, `BaseModule`, `OptimizeRequest`, `ModuleResult`.
- Produces:
  - `HybridRetrieval(db: AsyncIOMotorDatabase, config: dict)` — `config` may include `top_k`, `rrf_k`.
  - `await retrieval.process(request) -> (OptimizeRequest, ModuleResult)` — appends `## Retrieved Docs`; no-op if no `corpus_id`.
  - `await retrieval.add_chunks(corpus_id: str, chunks: list[dict]) -> int` — chunk shape `{"text": str, "source_file": str, "chunk_index": int, "metadata": dict}`.
  - `finops.modules.hybrid_retrieval._rrf_fusion(dense, sparse, k=60) -> list[dict]`.

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
    monkeypatch.setattr("finops.modules.hybrid_retrieval.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.hybrid_retrieval.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
def config():
    return {"top_k": 3, "rrf_k": 60}


@pytest.fixture
async def retrieval(finops_db, config):
    return HybridRetrieval(finops_db, config)


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
    doc = await finops_db[CORPUS_CHUNKS].find_one({"corpus_id": "corp1", "chunk_index": 0})
    assert doc["bm25_tokens"]  # non-empty tokenization


async def test_rrf_fusion_returns_ranked_results():
    from finops.modules.hybrid_retrieval import _rrf_fusion
    results_a = [{"_id": "a"}, {"_id": "b"}, {"_id": "c"}]
    results_b = [{"_id": "b"}, {"_id": "c"}, {"_id": "a"}]
    fused = _rrf_fusion(results_a, results_b, k=60)
    ids = [r["_id"] for r in fused]
    assert "b" in ids
    assert ids[0] in ("a", "b")
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_hybrid_retrieval.py -v 2>&1 | tail -15
```
Expected: ImportError (module does not exist yet).

- [ ] **Step 3: Create `finops/modules/hybrid_retrieval.py`**

```python
import time
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_query, embed_documents
from finops.db.collections import CORPUS_CHUNKS


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _rrf_fusion(dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
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
        baseline_tokens = await self._corpus_tokens(request.corpus_id)
        tokens_added = _count_tokens(retrieved)
        tokens_in = _count_tokens(request.context)
        section = "## Retrieved Docs\n" + retrieved
        new_context = request.context + ("\n\n" if request.context else "") + section
        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=new_context,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=_count_tokens(new_context),
            tokens_saved=max(0, baseline_tokens - tokens_added),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"retrieved {len(fused)} chunks (baseline={baseline_tokens} full-corpus tokens)",
            tokens_added=tokens_added,
            baseline_tokens=baseline_tokens,
        )

    async def add_chunks(self, corpus_id: str, chunks: list[dict]) -> int:
        texts = [c["text"] for c in chunks]
        embeddings = embed_documents(texts)
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
        embedding = embed_query(query)
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

    async def _corpus_tokens(self, corpus_id: str) -> int:
        total = 0
        async for doc in self._db[CORPUS_CHUNKS].find(
            {"corpus_id": corpus_id}, {"text": 1}
        ):
            total += _count_tokens(doc.get("text", ""))
        return total
```

- [ ] **Step 4: Run module tests**

```bash
docker compose run --rm dev pytest tests/modules/test_hybrid_retrieval.py -v 2>&1 | tail -15
```
Expected: 3 tests PASS. (`_rrf_fusion` is pure Python; `add_chunks` verifies writes + tokenization; live fusion ranking is validated in Task 11.)

- [ ] **Step 5: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add finops/modules/hybrid_retrieval.py tests/modules/test_hybrid_retrieval.py
git commit -m "feat: hybrid retrieval module (BM25+vector+RRF, composing append, honest baseline)"
```

---

## Task 6: Agent Memory Module + `/memory/retrieve` + `/memory/store`

Three-tier memory (working / episodic / semantic) backed by Motor + local embeddings. Retrieval uses `embed_query`; fact storage uses `embed_documents`. Episodic/semantic vector searches use the `agent_id` filter (indexes gained that filter in Task 1). `process` **appends** a `## Memory` section to `request.context` (never overwrites). Memory is an **augmenter**: `tokens_added` = tokens of injected memory; `baseline_tokens` = tokens of the FULL working-memory history for the agent (the naive "inject everything" alternative); `tokens_saved = max(0, baseline - added)`. Fact extraction uses `ChatAnthropic` (mocked in unit tests). `/memory/retrieve` and `/memory/store` expose the tiers directly.

**Files:**
- Create: `finops/modules/agent_memory.py`
- Modify: `finops/daemon/app.py` — add `POST /memory/retrieve` and `POST /memory/store`
- Create: `tests/modules/test_agent_memory.py`
- Create: `tests/daemon/test_memory.py`

**Interfaces:**
- Consumes: `embed_query`, `embed_documents`, `WORKING_MEMORY`, `EPISODIC_MEMORY`, `SEMANTIC_MEMORY`, `ChatAnthropic`, `BaseModule`, `OptimizeRequest`, `ModuleResult`.
- Produces:
  - `AgentMemory(db: AsyncIOMotorDatabase, config: dict)` — `config` may include `working_memory_turns`, `episodic_ttl_days`, `semantic_ttl_days`.
  - `await memory.process(request) -> (OptimizeRequest, ModuleResult)` — appends `## Memory`.
  - `await memory.store_turn(agent_id, session_id, turn, response) -> None`.
  - `await memory._get_working_memory(agent_id) -> list[dict]`, `_get_episodic_memory(agent_id, query) -> list[str]`, `_get_semantic_memory(agent_id, query) -> list[str]`.

- [ ] **Step 1: Write failing tests**

Create `tests/modules/test_agent_memory.py`:
```python
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
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/modules/test_agent_memory.py -v 2>&1 | tail -15
```
Expected: ImportError (module does not exist yet).

- [ ] **Step 3: Create `finops/modules/agent_memory.py`**

```python
import os
import time
from datetime import datetime, timezone, timedelta

from langchain_anthropic import ChatAnthropic
from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_query, embed_documents
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
                latency_ms=(time.perf_counter() - t0) * 1000, detail="no memory found",
            )

        memory_ctx = self._format_memory(working, episodic, semantic)
        baseline_tokens = await self._full_history_tokens(request.agent_id)
        tokens_added = _count_tokens(memory_ctx)
        tokens_in = _count_tokens(request.context)
        section = "## Memory\n" + memory_ctx
        new_context = request.context + ("\n\n" if request.context else "") + section
        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=new_context,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=_count_tokens(new_context),
            tokens_saved=max(0, baseline_tokens - tokens_added),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"working={len(working)}, episodic={len(episodic)}, semantic={len(semantic)}",
            tokens_added=tokens_added,
            baseline_tokens=baseline_tokens,
        )

    async def store_turn(self, agent_id: str, session_id: str, turn: str, response: str) -> None:
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

    async def _full_history_tokens(self, agent_id: str) -> int:
        total = 0
        async for doc in self._db[WORKING_MEMORY].find({"agent_id": agent_id}, {"messages": 1}):
            for m in doc.get("messages", []):
                total += _count_tokens(m.get("content", ""))
        return total

    async def _get_episodic_memory(self, agent_id: str, query: str) -> list[str]:
        embedding = embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "episodic_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 3,
                    "filter": {"agent_id": {"$eq": agent_id}},
                }
            },
        ]
        results = []
        async for doc in self._db[EPISODIC_MEMORY].aggregate(pipeline):
            results.append(doc["content"])
        return results

    async def _get_semantic_memory(self, agent_id: str, query: str) -> list[str]:
        embedding = embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "semantic_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 5,
                    "filter": {"agent_id": {"$eq": agent_id}},
                }
            },
        ]
        results = []
        async for doc in self._db[SEMANTIC_MEMORY].aggregate(pipeline):
            results.append(doc["fact"])
        return results

    async def _extract_and_store_facts(self, agent_id: str, turn: str, response: str) -> None:
        llm = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            max_tokens=256,
        )
        result = llm.invoke(_FACT_PROMPT.format(turn=turn, response=response))
        raw = result.content.strip()
        if not raw:
            return
        facts = [f.strip() for f in raw.splitlines() if f.strip()]
        fact_embeddings = embed_documents(facts)
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
                        "filter": {"agent_id": {"$eq": agent_id}},
                    }
                },
                {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
                {"$match": {"_score": {"$gte": _DEDUP_THRESHOLD}}},
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

    def _format_memory(self, working: list[dict], episodic: list[str], semantic: list[str]) -> str:
        parts = []
        if semantic:
            parts.append("### Known Facts\n" + "\n".join(f"- {f}" for f in semantic))
        if episodic:
            parts.append("### Recent Context\n" + "\n".join(f"- {e}" for e in episodic))
        if working:
            msgs = "\n".join(f"{m['role']}: {m['content']}" for m in working)
            parts.append("### Conversation\n" + msgs)
        return "\n\n".join(parts)
```

- [ ] **Step 4: Run module tests**

```bash
docker compose run --rm dev pytest tests/modules/test_agent_memory.py -v 2>&1 | tail -15
```
Expected: 5 tests PASS. (Working-memory path uses ordinary reads; episodic/semantic `$vectorSearch` recall is validated in Task 11.)

- [ ] **Step 5: Add `/memory/retrieve` and `/memory/store` to `finops/daemon/app.py`**

Add after the cache endpoints:
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
```

```bash
docker compose run --rm dev pytest tests/daemon/test_memory.py -v 2>&1 | tail -10
```
Expected: 2 tests PASS.

- [ ] **Step 7: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add finops/modules/agent_memory.py finops/daemon/app.py \
        tests/modules/test_agent_memory.py tests/daemon/test_memory.py
git commit -m "feat: agent memory module (composing append, agent_id filter) + /memory/* endpoints"
```

---

## Task 7: Context Compressor Module (reducer, runs last)

Wraps LLMLingua-2. It is a **reducer** that runs LAST over the ASSEMBLED context (`request.context` is the full composed block by then). Bypasses when `len(context)//4 < token_threshold`. On compress: `tokens_saved = before - after`, `tokens_added = 0`, `baseline_tokens = before`. Saves per-run stats to `compression_stats`. Wires `_get_compressor()` into `finops warmup` (referenced in Task 1) so the compressor model is prefetched into the hf_cache volume.

**Files:**
- Create: `finops/modules/context_compressor.py`
- Create: `tests/modules/test_context_compressor.py`

**Interfaces:**
- Consumes: `COMPRESSION_STATS`, `PromptCompressor` (LLMLingua-2), `BaseModule`, `OptimizeRequest`, `ModuleResult`.
- Produces:
  - `ContextCompressor(db: AsyncIOMotorDatabase, config: dict)` — `config` may include `token_threshold`, `target_ratio`.
  - `finops.modules.context_compressor._get_compressor() -> PromptCompressor` — lazy singleton (warmup target from Task 1).
  - `await compressor.process(request) -> (OptimizeRequest, ModuleResult)`.

- [ ] **Step 1: Write failing tests**

Create `tests/modules/test_context_compressor.py`:
```python
import pytest
from unittest.mock import MagicMock
from finops.modules.context_compressor import ContextCompressor
from finops.modules._base import OptimizeRequest
from finops.db.collections import COMPRESSION_STATS


@pytest.fixture(autouse=True)
def mock_compressor(monkeypatch):
    fake = MagicMock()
    def fake_compress(context_list, rate, force_tokens, **kw):
        text = context_list[0]
        words = text.split()
        compressed = " ".join(words[: max(1, len(words) // 4)])
        return {"compressed_prompt": compressed}
    fake.compress_prompt.side_effect = fake_compress
    monkeypatch.setattr("finops.modules.context_compressor._get_compressor", lambda: fake)


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
    assert result.tokens_added == 0
    assert "bypass" in result.detail


async def test_compresses_when_above_threshold(compressor, long_req):
    before = len(long_req.context) // 4
    new_req, result = await compressor.process(long_req)
    assert len(new_req.context) < len(long_req.context)
    assert result.tokens_saved > 0
    assert result.tokens_added == 0
    assert result.baseline_tokens == before


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
Expected: ImportError (module does not exist yet).

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
                baseline_tokens=original_tokens,
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
            detail=f"compressed {ratio:.1f}x ({original_tokens}->{compressed_tokens} tokens)",
            baseline_tokens=original_tokens,
        )
```

- [ ] **Step 4: Run module tests**

```bash
docker compose run --rm dev pytest tests/modules/test_context_compressor.py -v 2>&1 | tail -10
```
Expected: 3 tests PASS.

- [ ] **Step 5: Verify `finops warmup` now wires the compressor (from Task 1)**

Task 1 already added `finops warmup` calling `_get_compressor()`. Confirm the import now resolves (no need to download the compressor model in the unit gate — just that the symbol exists):
```bash
docker compose run --rm dev python -c "from finops.modules.context_compressor import _get_compressor; print('compressor symbol OK')"
```
Expected: prints `compressor symbol OK`.

Optionally prefetch both models into the hf_cache volume now (slow; downloads the LLMLingua-2 model):
```bash
docker compose run --rm dev finops warmup
```
Expected: prints embedding + compressor ready lines, then `Warmup complete.`

- [ ] **Step 6: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add finops/modules/context_compressor.py tests/modules/test_context_compressor.py
git commit -m "feat: context compressor module (LLMLingua-2 reducer, warmup wiring)"
```

---

## Task 8: Router (strategy-driven, composing) + `POST /optimize`

`ModulePipeline` in `finops/daemon/router.py` is driven by a `Strategy`: it instantiates all five modules with strategy-merged configs (strategy overrides layered onto module config; `cache_key` policy injected into `semantic_cache`), then runs enabled modules in the strategy's `order`. Augmenters append (compose); the cache short-circuits before composition. `POST /optimize` selects the strategy (body override → config default) and delegates.

**Files:**
- Create: `finops/daemon/router.py`
- Modify: `finops/daemon/app.py` — add `POST /optimize`
- Create: `tests/daemon/test_optimize.py`

**Interfaces:**
- Consumes: all five module classes, `get_strategy`, `Strategy`, `load_config`, `get_async_db`, `OptimizeRequest`, `ModuleResult`.
- Produces:
  - `ModulePipeline(db, module_configs: dict, strategy: Strategy)`.
  - `await pipeline.run(request: OptimizeRequest) -> dict` — returns `{optimized_prompt, optimized_context, cache_hit, strategy, tokens_saved, module_results[]}`.
  - `finops.daemon.router._result_dict(r: ModuleResult) -> dict`.
- Response `module_results[]` entries include `module, tokens_in, tokens_out, tokens_saved, tokens_added, baseline_tokens, latency_ms, detail`.

- [ ] **Step 1: Write failing tests**

Create `tests/daemon/test_optimize.py`:
```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app
from finops.modules._base import OptimizeRequest, ModuleResult
from finops.daemon.strategies import COMPOSE_THEN_COMPRESS, Strategy


def _make_append_module(name, section_label):
    mod = MagicMock()
    mod.name = name
    async def proc(req):
        section = f"## {section_label}\nfrom-{name}"
        new_ctx = req.context + ("\n\n" if req.context else "") + section
        new_req = OptimizeRequest(prompt=req.prompt, context=new_ctx,
                                  agent_id=req.agent_id, framework=req.framework,
                                  corpus_id=req.corpus_id)
        return new_req, ModuleResult(module=name, tokens_in=1, tokens_out=1,
                                     tokens_saved=0, latency_ms=1.0, detail="append",
                                     tokens_added=3, baseline_tokens=10)
    mod.process = proc
    return mod


def _make_cache_hit_module():
    mod = MagicMock()
    mod.name = "semantic_cache"
    async def hit(req):
        new_req = OptimizeRequest(prompt=req.prompt, context="cached response",
                                  agent_id=req.agent_id, framework=req.framework,
                                  corpus_id=req.corpus_id)
        return new_req, ModuleResult(module="semantic_cache", tokens_in=500, tokens_out=0,
                                     tokens_saved=500, latency_ms=5.0, detail="exact hash hit",
                                     short_circuit=True, baseline_tokens=500)
    mod.process = hit
    return mod


def _make_passthrough_module(name):
    mod = MagicMock()
    mod.name = name
    mod.process = AsyncMock(side_effect=lambda req: (
        req, ModuleResult(module=name, tokens_in=10, tokens_out=10,
                          tokens_saved=0, latency_ms=1.0, detail="pass")))
    return mod


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_optimize_returns_shape_with_strategy(client, finops_db):
    from finops.daemon.config import save_config
    await save_config(finops_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": False},
        "agent_memory": {"enabled": False}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    resp = await client.post("/optimize", json={
        "prompt": "What is Python?", "context": "some context",
        "agent_id": "a1", "framework": "test",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "optimized_prompt" in data
    assert "optimized_context" in data
    assert "tokens_saved" in data
    assert "module_results" in data
    assert data["strategy"] == "compose_then_compress"
    assert data["module_results"] == []


async def test_optimize_preserves_prompt(client, finops_db):
    from finops.daemon.config import save_config
    await save_config(finops_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": False},
        "agent_memory": {"enabled": False}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    resp = await client.post("/optimize", json={
        "prompt": "unique test prompt xyz", "context": "",
        "agent_id": "a1", "framework": "test",
    })
    assert resp.json()["optimized_prompt"] == "unique test prompt xyz"


async def test_pipeline_short_circuits_on_cache_hit(finops_db):
    from finops.daemon.router import ModulePipeline
    cache_mod = _make_cache_hit_module()
    other = _make_passthrough_module("context_compressor")
    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._strategy = COMPOSE_THEN_COMPRESS
    pipeline._modules = {"semantic_cache": cache_mod, "context_compressor": other}
    pipeline._enabled = {"semantic_cache": True, "context_compressor": True}
    req = OptimizeRequest(prompt="hi", context="ctx", agent_id="a", framework="f")
    result = await pipeline.run(req)
    assert result["cache_hit"] is True
    assert result["optimized_context"] == "cached response"
    assert result["tokens_saved"] == 500
    assert result["strategy"] == "compose_then_compress"
    other.process.assert_not_called()


async def test_pipeline_skips_disabled_modules(finops_db):
    from finops.daemon.router import ModulePipeline
    mod = _make_passthrough_module("context_compressor")
    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._strategy = COMPOSE_THEN_COMPRESS
    pipeline._modules = {"context_compressor": mod}
    pipeline._enabled = {"context_compressor": False}
    req = OptimizeRequest(prompt="hi", context="ctx", agent_id="a", framework="f")
    result = await pipeline.run(req)
    mod.process.assert_not_called()
    assert result["tokens_saved"] == 0


async def test_pipeline_composes_both_augmenters(finops_db):
    from finops.daemon.router import ModulePipeline
    graph = _make_append_module("codebase_graph", "Relevant Code")
    memory = _make_append_module("agent_memory", "Memory")
    strat = Strategy(name="two_aug", order=("codebase_graph", "agent_memory"),
                     composition="compose", cache_key="prompt+scope")
    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._strategy = strat
    pipeline._modules = {"codebase_graph": graph, "agent_memory": memory}
    pipeline._enabled = {"codebase_graph": True, "agent_memory": True}
    req = OptimizeRequest(prompt="hi", context="ORIG", agent_id="a", framework="f")
    result = await pipeline.run(req)
    ctx = result["optimized_context"]
    assert "ORIG" in ctx
    assert "## Relevant Code" in ctx
    assert "## Memory" in ctx
    assert result["cache_hit"] is False
    assert len(result["module_results"]) == 2
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/daemon/test_optimize.py -v 2>&1 | tail -15
```
Expected: ImportError (router.py does not exist yet).

- [ ] **Step 3: Create `finops/daemon/router.py`**

```python
from finops.daemon.strategies import get_strategy, Strategy
from finops.modules._base import OptimizeRequest, ModuleResult
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules.semantic_cache import SemanticCache
from finops.modules.agent_memory import AgentMemory
from finops.modules.context_compressor import ContextCompressor
from finops.modules.hybrid_retrieval import HybridRetrieval

_MODULE_CLASSES = {
    "codebase_graph":     CodebaseGraph,
    "semantic_cache":     SemanticCache,
    "agent_memory":       AgentMemory,
    "context_compressor": ContextCompressor,
    "hybrid_retrieval":   HybridRetrieval,
}


def _result_dict(r: ModuleResult) -> dict:
    return {
        "module": r.module, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
        "tokens_saved": r.tokens_saved, "tokens_added": r.tokens_added,
        "baseline_tokens": r.baseline_tokens, "latency_ms": r.latency_ms, "detail": r.detail,
    }


class ModulePipeline:
    def __init__(self, db, module_configs: dict, strategy: Strategy):
        self._strategy = strategy
        merged = {}
        for name in _MODULE_CLASSES:
            cfg = dict(module_configs.get(name, {}))
            cfg.update(strategy.overrides.get(name, {}))
            merged[name] = cfg
        merged["semantic_cache"]["cache_key"] = strategy.cache_key
        self._modules = {name: cls(db, merged[name]) for name, cls in _MODULE_CLASSES.items()}
        self._enabled = {name: module_configs.get(name, {}).get("enabled", False) for name in _MODULE_CLASSES}

    async def run(self, request: OptimizeRequest) -> dict:
        collected: list[ModuleResult] = []
        for name in self._strategy.order:
            if not self._enabled.get(name, False):
                continue
            request, result = await self._modules[name].process(request)
            collected.append(result)
            if name in self._strategy.short_circuit_on and result.short_circuit:
                return {
                    "optimized_prompt": request.prompt, "optimized_context": request.context,
                    "cache_hit": True, "strategy": self._strategy.name,
                    "tokens_saved": result.tokens_saved,
                    "module_results": [_result_dict(r) for r in collected],
                }
        return {
            "optimized_prompt": request.prompt, "optimized_context": request.context,
            "cache_hit": False, "strategy": self._strategy.name,
            "tokens_saved": sum(r.tokens_saved for r in collected),
            "module_results": [_result_dict(r) for r in collected],
        }
```

- [ ] **Step 4: Add `POST /optimize` to `finops/daemon/app.py`**

Add `OptimizeRequest` to the imports at the top of `app.py`:
```python
from finops.modules._base import OptimizeRequest
```
(`get_strategy` was already imported in Task 3.)

Add after the config endpoints (before the cache/memory endpoints):
```python
@app.post("/optimize")
async def post_optimize(body: dict):
    db = get_async_db()
    config = await load_config(db)
    strategy = get_strategy(body.get("strategy") or config.get("strategy"))
    from finops.daemon.router import ModulePipeline
    pipeline = ModulePipeline(db, config.get("modules", {}), strategy)
    request = OptimizeRequest(
        prompt=body.get("prompt", ""),
        context=body.get("context", ""),
        agent_id=body.get("agent_id", "default"),
        framework=body.get("framework", "unknown"),
        corpus_id=body.get("corpus_id"),
    )
    return await pipeline.run(request)
```

- [ ] **Step 5: Run tests**

```bash
docker compose run --rm dev pytest tests/daemon/test_optimize.py -v 2>&1 | tail -15
```
Expected: 5 tests PASS.

- [ ] **Step 6: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add finops/daemon/router.py finops/daemon/app.py tests/daemon/test_optimize.py
git commit -m "feat: strategy-driven composing ModulePipeline + POST /optimize"
```

---

## Task 9: `/complete` Proxy (Anthropic + OpenAI) — Closes the Loop

`finops/daemon/providers.py` exposes `call_llm(provider, model, prompt, context) -> (response_text, input_tokens, output_tokens)`. Anthropic via `anthropic.AsyncAnthropic`, OpenAI via `openai.AsyncOpenAI`; keys from env. `POST /complete` runs the `/optimize` pipeline, and — only on a cache MISS — calls the LLM, stores the response in the cache with the pipeline's `tokens_saved`, records the turn in memory, and returns the answer. On a cache HIT (short-circuit) it returns the cached response WITHOUT calling the LLM.

**Files:**
- Create: `finops/daemon/providers.py`
- Modify: `finops/daemon/app.py` — add `POST /complete`
- Create: `tests/daemon/test_complete.py`

**Interfaces:**
- Consumes: `anthropic.AsyncAnthropic`, `openai.AsyncOpenAI`, `get_strategy`, `ModulePipeline`, `SemanticCache`, `AgentMemory`, `OptimizeRequest`.
- Produces:
  - `await finops.daemon.providers.call_llm(provider: str, model: str, prompt: str, context: str) -> tuple[str, int, int]`.
  - `POST /complete` → `{response, tokens_saved, cache_hit, input_tokens, output_tokens, module_results}`.

- [ ] **Step 1: Create `finops/daemon/providers.py`**

```python
import os

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI


def _compose_message(prompt: str, context: str) -> str:
    if context:
        return f"{context}\n\n{prompt}"
    return prompt


async def call_llm(provider: str, model: str, prompt: str, context: str) -> tuple[str, int, int]:
    message = _compose_message(prompt, context)
    if provider == "anthropic":
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": message}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        return text, resp.usage.input_tokens, resp.usage.output_tokens
    if provider == "openai":
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message}],
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return text, usage.prompt_tokens, usage.completion_tokens
    raise ValueError(f"unknown provider: {provider}")
```

- [ ] **Step 2: Add `POST /complete` to `finops/daemon/app.py`**

Add after `POST /optimize`:
```python
@app.post("/complete")
async def post_complete(body: dict):
    db = get_async_db()
    config = await load_config(db)
    strategy = get_strategy(body.get("strategy") or config.get("strategy"))
    from finops.daemon.router import ModulePipeline
    pipeline = ModulePipeline(db, config.get("modules", {}), strategy)
    request = OptimizeRequest(
        prompt=body.get("prompt", ""),
        context=body.get("context", ""),
        agent_id=body.get("agent_id", "default"),
        framework=body.get("framework", "unknown"),
        corpus_id=body.get("corpus_id"),
    )
    optimized = await pipeline.run(request)

    if optimized["cache_hit"]:
        return {
            "response": optimized["optimized_context"],
            "tokens_saved": optimized["tokens_saved"],
            "cache_hit": True,
            "input_tokens": 0,
            "output_tokens": 0,
            "module_results": optimized["module_results"],
        }

    from finops.daemon.providers import call_llm
    response_text, input_tokens, output_tokens = await call_llm(
        provider=body.get("provider", "anthropic"),
        model=body.get("model", ""),
        prompt=optimized["optimized_prompt"],
        context=optimized["optimized_context"],
    )

    cache_cfg = {**config.get("modules", {}).get("semantic_cache", {}), "cache_key": strategy.cache_key}
    from finops.modules.semantic_cache import SemanticCache
    cache = SemanticCache(db, cache_cfg)
    # A future cache hit avoids the entire LLM round-trip, so the entry's savings
    # is the avoided token cost (input+output), not this call's optimization delta.
    await cache.store(
        prompt=request.prompt,
        response=response_text,
        framework=request.framework,
        model=body.get("model", ""),
        tokens_saved=input_tokens + output_tokens,
        agent_id=request.agent_id,
        corpus_id=request.corpus_id or "",
    )

    from finops.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, config.get("modules", {}).get("agent_memory", {}))
    await memory.store_turn(
        request.agent_id, body.get("session_id", "default"), request.prompt, response_text
    )

    return {
        "response": response_text,
        "tokens_saved": optimized["tokens_saved"],
        "cache_hit": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "module_results": optimized["module_results"],
    }
```

- [ ] **Step 3: Write and run `/complete` tests**

Create `tests/daemon/test_complete.py`:
```python
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
```

```bash
docker compose run --rm dev pytest tests/daemon/test_complete.py -v 2>&1 | tail -15
```
Expected: 2 tests PASS. (The second-call cache hit fires via the exact-hash path since embeddings are mocked and the `prompt+scope` key is identical.)

- [ ] **Step 4: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add finops/daemon/providers.py finops/daemon/app.py tests/daemon/test_complete.py
git commit -m "feat: /complete proxy (Anthropic+OpenAI) with cache+memory write-back"
```

---

## Task 10: Metrics Endpoint

`finops/daemon/metrics.py` aggregates token savings from `cache_entries` and `compression_stats`. `GET /metrics` returns totals + per-module breakdown. Codebase_graph / hybrid_retrieval / agent_memory savings are surfaced per-request via `/optimize` `module_results` (they are not persisted as standalone events here) — `/metrics` covers the two collections that persist events; the per-module list notes this. Tests seed both collections and verify aggregation math.

**Files:**
- Create: `finops/daemon/metrics.py`
- Modify: `finops/daemon/app.py` — add `GET /metrics`
- Create: `tests/daemon/test_metrics.py`

**Interfaces:**
- Consumes: `CACHE_ENTRIES`, `COMPRESSION_STATS`, `get_async_db`.
- Produces: `await finops.daemon.metrics.aggregate_metrics(db) -> {total_tokens_saved, cache_hit_rate, compression_ratio, per_module[]}`.

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
    assert data["total_tokens_saved"] >= 1200
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
```

- [ ] **Step 2: Run to verify failures**

```bash
docker compose run --rm dev pytest tests/daemon/test_metrics.py -v 2>&1 | tail -15
```
Expected: ImportError / 404 (metrics not wired yet).

- [ ] **Step 3: Create `finops/daemon/metrics.py`**

```python
from motor.motor_asyncio import AsyncIOMotorDatabase
from finops.db.collections import CACHE_ENTRIES, COMPRESSION_STATS


async def aggregate_metrics(db: AsyncIOMotorDatabase) -> dict:
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
        per_module.append({"module": "semantic_cache", "tokens_saved": cache_tokens, "events": cache_events})
    if comp_tokens > 0 or comp_events > 0:
        per_module.append({"module": "context_compressor", "tokens_saved": comp_tokens, "events": comp_events})

    return {
        "total_tokens_saved": cache_tokens + comp_tokens,
        "cache_hit_rate":     round(cache_hit_rate, 4),
        "compression_ratio":  comp_ratio,
        "per_module":         per_module,
    }
```
Note: codebase_graph, hybrid_retrieval, and agent_memory do not persist per-event savings documents; their honest per-request savings surface in `/optimize` and `/complete` `module_results`. If those need historical aggregation later, add an events collection — out of scope for this task.

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

- [ ] **Step 6: Run full unit suite**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all unit tests PASS.

- [ ] **Step 7: Commit**

```bash
git add finops/daemon/metrics.py finops/daemon/app.py tests/daemon/test_metrics.py
git commit -m "feat: metrics aggregation and GET /metrics endpoint"
```

---

## Task 11: Integration Test Suite (real mongot + real local embeddings)

Adds `@pytest.mark.integration` tests exercising the REAL local `voyageai/voyage-4-nano` model (no mock) against live `mongot`, using `wait_for_queryable` to block until each search index is queryable. Covers: (a) semantic cache paraphrase hit, (b) codebase graph symbol recall, (c) hybrid retrieval top-ranked chunk, (d) agent memory recall, (e) end-to-end `/complete` with a MOCKED provider but REAL embeddings + real vector search populating then hitting the cache. These are slow and download models on first run (mitigated by `finops warmup` + the `hf_cache` volume).

**Files:**
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/integration/test_integration_modules.py`

**Interfaces:**
- Consumes: real `finops.modules.embeddings` (no monkeypatch), `wait_for_queryable`, `sync_db` (for polling), `finops_db` (async queries), all module classes, `ModulePipeline`, real `POST /complete` app with only `call_llm` mocked.

- [ ] **Step 1: Ensure models are warm (first run downloads them)**

```bash
docker compose run --rm dev finops warmup 2>&1 | tail -5
```
Expected: embedding + compressor models cached (fast if hf_cache already populated by Task 1/Task 7).

- [ ] **Step 2: Write the integration tests**

Create `tests/integration/__init__.py` (empty file).

Create `tests/integration/test_integration_modules.py`:
```python
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from finops.modules.semantic_cache import SemanticCache
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules.hybrid_retrieval import HybridRetrieval
from finops.modules.agent_memory import AgentMemory
from finops.modules._base import OptimizeRequest
from finops.db.collections import (
    CACHE_ENTRIES, CODEBASE_NODES, CORPUS_CHUNKS,
)
from finops.daemon.app import app
from tests.conftest import wait_for_queryable

pytestmark = pytest.mark.integration


SAMPLE_SOURCE = """
def add(a, b):
    return a + b


def greet(name):
    return f"hello {name}"
"""


async def test_semantic_cache_paraphrase_hit(finops_db, sync_db):
    cache = SemanticCache(finops_db, {"similarity_threshold": 0.6, "cache_key": "prompt"})
    await cache.store(
        prompt="How do I reverse a list in Python?",
        response="Use lst[::-1] or reversed(lst).",
        framework="test", model="claude", tokens_saved=300,
    )
    wait_for_queryable(sync_db[CACHE_ENTRIES], "cache_vector_index")
    req = OptimizeRequest(
        prompt="What is the way to reverse a Python list?",
        context="orig", agent_id="a1", framework="test",
    )
    new_req, result = await cache.process(req)
    assert result.short_circuit is True
    assert "reversed" in new_req.context or "[::-1]" in new_req.context


async def test_codebase_graph_symbol_recall(finops_db, sync_db):
    graph = CodebaseGraph(finops_db, {"repo_paths": []})
    n = await graph.index_file("repoI", "mod.py", SAMPLE_SOURCE)
    assert n >= 2
    wait_for_queryable(sync_db[CODEBASE_NODES], "codebase_vector_index")
    results = await graph.query("repoI", "function that adds two numbers", k=3)
    symbols = [r["symbol"] for r in results]
    assert "add" in symbols


async def test_hybrid_retrieval_top_ranked(finops_db, sync_db):
    retrieval = HybridRetrieval(finops_db, {"top_k": 2, "rrf_k": 60})
    await retrieval.add_chunks("corpI", [
        {"text": "MongoDB is a document-oriented NoSQL database.", "source_file": "d.txt", "chunk_index": 0, "metadata": {}},
        {"text": "The Eiffel Tower is located in Paris, France.", "source_file": "d.txt", "chunk_index": 1, "metadata": {}},
    ])
    wait_for_queryable(sync_db[CORPUS_CHUNKS], "corpus_vector_index")
    req = OptimizeRequest(
        prompt="what kind of database is MongoDB?", context="", agent_id="a1",
        framework="test", corpus_id="corpI",
    )
    new_req, result = await retrieval.process(req)
    assert "## Retrieved Docs" in new_req.context
    assert "document-oriented" in new_req.context


async def test_agent_memory_recall(finops_db, sync_db, monkeypatch):
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content="")
    monkeypatch.setattr("finops.modules.agent_memory.ChatAnthropic", lambda **kw: fake)
    memory = AgentMemory(finops_db, {"working_memory_turns": 20})
    await memory.store_turn("agentI", "sess", "My favorite language is Rust.", "Noted, Rust it is.")
    working = await memory._get_working_memory("agentI")
    assert any("Rust" in m["content"] for m in working)
    req = OptimizeRequest(prompt="what language do I like?", context="ctx", agent_id="agentI", framework="test")
    new_req, result = await memory.process(req)
    assert "## Memory" in new_req.context
    assert "Rust" in new_req.context


async def test_end_to_end_complete_real_embeddings(finops_db, sync_db, monkeypatch):
    from finops.daemon.config import save_config
    await save_config(finops_db, {"modules": {
        "codebase_graph": {"enabled": False},
        "semantic_cache": {"enabled": True, "similarity_threshold": 0.6, "cache_key": "prompt"},
        "agent_memory": {"enabled": False},
        "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    monkeypatch.setattr("finops.daemon.providers.call_llm", AsyncMock(return_value=("real answer", 80, 40)))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        body = {"prompt": "Explain what a hash map is.", "agent_id": "e2e", "session_id": "s",
                "framework": "test", "provider": "anthropic", "model": "claude-x"}
        first = await c.post("/complete", json=body)
        assert first.json()["cache_hit"] is False
        assert first.json()["response"] == "real answer"

        wait_for_queryable(sync_db[CACHE_ENTRIES], "cache_vector_index")

        spy = AsyncMock(return_value=("SHOULD NOT CALL", 1, 1))
        monkeypatch.setattr("finops.daemon.providers.call_llm", spy)
        paraphrase = {**body, "prompt": "Can you explain what a hash map is?"}
        second = await c.post("/complete", json=paraphrase)
        data = second.json()
        assert data["cache_hit"] is True
        assert data["response"] == "real answer"
        spy.assert_not_called()
```
Note: integration thresholds are set lower (0.6) than the 0.92 default because cross-phrasing cosine scores vary; the point is to prove real semantic recall, not to tune production thresholds. If a paraphrase assertion is flaky on a given machine, lower the threshold further or tighten the paraphrase — do not mock the model.

- [ ] **Step 3: Run the integration suite (slow; live mongot + real model)**

```bash
docker compose run --rm dev pytest -m integration -v 2>&1 | tail -30
```
Expected: 5 integration tests PASS. First run downloads models if hf_cache is cold (minutes); subsequent runs are faster. Index-warm polling can take up to ~90s per index.

- [ ] **Step 4: Confirm the unit gate still excludes integration**

```bash
docker compose run --rm dev pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```
Expected: all unit tests PASS; 0 integration collected (fast).

- [ ] **Step 5: Run the complete suite (unit + integration) once**

```bash
docker compose run --rm dev pytest -q 2>&1 | tail -12
```
Expected: everything PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_integration_modules.py
git commit -m "test: integration suite (real mongot + real local voyage-4-nano embeddings)"
```

---

## Summary of Tasks

1. **Foundation** — local `sentence-transformers` embeddings (asymmetric, no API key), dep changes (`sentence-transformers`, `openai`, `tree-sitter-python`; `voyageai` → `[hosted]`), Plan-1 fixes (M1/M2/M3), honest `ModuleResult` fields, index `filter_paths`, `hf_cache` volume + `HF_HOME` + `finops warmup`, `finops_db` fixture + `wait_for_queryable`, `integration` marker.
2. **Strategy abstraction** — frozen `Strategy` dataclass + registry (`compose_then_compress`, `cache_first_aggressive`), `get_strategy`/`list_strategies`, `config.strategy` default.
3. **Semantic Cache** — cache_key policy (`prompt` / `prompt+scope`), asymmetric embed, short-circuit hit, idempotent TTL store; `/cache/lookup` + `/cache/store`.
4. **Codebase Graph** — Tree-Sitter Python, composing `## Relevant Code` append, honest full-index baseline.
5. **Hybrid Retrieval** — BM25 + vector + RRF, composing `## Retrieved Docs` append, honest full-corpus baseline.
6. **Agent Memory** — three-tier, agent_id-filtered vector search, composing `## Memory` append, full-history baseline; `/memory/retrieve` + `/memory/store`.
7. **Context Compressor** — LLMLingua-2 reducer, runs last, bypass under threshold, warmup wiring.
8. **Router** — strategy-driven composing `ModulePipeline` (overrides merge, cache_key injection, short-circuit) + `POST /optimize`.
9. **`/complete` proxy** — `call_llm` (Anthropic + OpenAI), pipeline → LLM → cache+memory write-back, cache-hit skips the LLM.
10. **Metrics** — `aggregate_metrics` over cache + compression collections + `GET /metrics`.
11. **Integration suite** — real mongot + real local embeddings, `wait_for_queryable`, five scenarios including end-to-end `/complete`.
