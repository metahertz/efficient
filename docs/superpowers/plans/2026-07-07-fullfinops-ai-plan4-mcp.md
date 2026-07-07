# fullFinOps-AI Plan 4 — Claude Code MCP Server (Python, dockerized)

For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan. Dispatch each task to a fresh implementer subagent, one task at a time, in order. Each subagent must follow superpowers:test-driven-development (write the failing test, run it, watch it fail, implement, run it, watch it pass, commit) and superpowers:verification-before-completion (run the exact commands and paste real output before claiming success). Route each implementer to Opus (judgment-heavy). Do not batch tasks; Task N+1 depends on Task N being merged.

## Goal

Expose the fullFinOps-AI daemon to Claude Code as an MCP server, so Claude Code can call the token-saving pipeline (`optimize_context`), index and look up code symbols (`index_codebase`, `lookup_symbol`), and read/write agent memory (`retrieve_memory`, `store_memory`) as native MCP tools. The MCP server is Python (`mcp` SDK), runs inside the existing container over stdio, and is a thin HTTP client to the daemon. Along the way, make `docker compose build` reliable for both the `dev` and `daemon` images by installing CPU-only torch before the editable install (the daemon image currently lacks the transformers fix and clean builds time out re-downloading ~2 GB of CUDA torch).

This is Revision 3 of the design spec ("Claude Code MCP Server (Plan 4, pulled forward)"), which supersedes the TypeScript row of the Plugin Layer table.

## Architecture

```
Claude Code  ──(spawns via `docker compose run --rm -T mcp`)──▶  finops/mcp/server.py
                                                                 (FastMCP, stdio transport)
                                                                        │
                                                                        │ finops/mcp/daemon_client.py
                                                                        │ httpx → FINOPS_DAEMON_URL (default http://daemon:7432)
                                                                        ▼
                                                        FastAPI Daemon (finops/daemon/app.py)
                                                        /optimize  /codebase/index  /codebase/query
                                                        /memory/retrieve  /memory/store
                                                                        │
                                                                        ▼
                                                        Modules (CodebaseGraph, AgentMemory, ...)
                                                                        │
                                                                        ▼
                                                        MongoDB Atlas Local (mongot vector search)
```

- The daemon runs continuously: `docker compose up -d daemon`.
- Claude Code launches the MCP server on demand; it is a short-lived process that talks stdio to Claude Code and HTTP to the daemon over the compose network.
- All five MCP tools are one-line delegations to `daemon_client` functions, which are one-line `httpx` POSTs to the daemon.

## Tech Stack

- Python ≥ 3.11.
- `mcp` SDK (`from mcp.server.fastmcp import FastMCP`), stdio transport (the default).
- `httpx` async client (already a dependency).
- FastAPI daemon (existing).
- MongoDB Atlas Local with `mongot` (existing), local `voyage-4-nano` embeddings via `sentence-transformers` + `transformers>=4.54,<4.58`.
- Docker Compose; CPU-only torch from the PyTorch CPU wheel index.
- pytest + pytest-asyncio (`asyncio_mode = "auto"`), `@pytest.mark.integration` for live-daemon smoke.

## Global Constraints

- **Container-only.** Every command runs inside the container. Tests run via `docker compose run --rm dev <cmd>` (e.g. `docker compose run --rm dev pytest -m "not integration" -q`). Never invoke host `python`/`pytest`/`pip` — there is no host virtualenv.
- **Python ≥ 3.11; pytest-asyncio `mode=auto`** (already set in `pyproject.toml`). Do not add `@pytest.mark.asyncio` decorators; async test functions are collected automatically.
- **MCP server is Python, dockerized, stdio transport.** Launched by Claude Code via `docker compose run --rm -T mcp`. Reaches the daemon at `FINOPS_DAEMON_URL` (default `http://daemon:7432`).
- **stdout discipline.** stdout is the MCP protocol channel. ALL logging goes to stderr via `logging.basicConfig(level=logging.INFO, stream=sys.stderr)`. Never `print()` to stdout anywhere in `finops/mcp/`. Any stray stdout write corrupts the JSON-RPC framing.
- **MCP tool docstrings are FUNCTIONAL and REQUIRED.** The docstring of each `@mcp.tool()` function is the tool description sent to the model. These docstrings are the sole exemption to the repo's no-comments/no-docstrings style. Do NOT add any other comments or docstrings to `finops/mcp/` or the daemon edits. Do NOT delete or reword the tool docstrings.
- **Unit tests mock the daemon.** Unit tests monkeypatch the `daemon_client` functions (or `httpx`) — no live daemon needed. Only `@pytest.mark.integration` tests use a live daemon + a real MCP client.
- **`mcp` dependency is verified.** `from mcp.server.fastmcp import FastMCP`; `m = FastMCP("finops")`; `@m.tool()` decorator; `m.run(transport="stdio")`. Pin `mcp>=1.2` in `pyproject.toml`. The exact introspection API for listing registered tools (`list_tools`, whether sync/async, return shape) varies across SDK minor versions — the implementer must verify it inside the built image and adapt the assertion, not assume.
- **No emojis in code, tests, or docs.**

---

### Task 1 — Reliable images (CPU torch) + `mcp` dependency

Make `docker compose build` reliable for both `dev` and `daemon` by installing CPU-only torch in the Dockerfile `base` stage BEFORE the first editable install, so `sentence-transformers` reuses the ~200 MB CPU wheel instead of pulling the ~2 GB CUDA build (which has been timing out on clean builds). The daemon runs CPU-only anyway. Also add the `mcp>=1.2` dependency to `pyproject.toml`. Verify in a FRESHLY built image (not the current hand-patched `dev`) that `transformers` is in `[4.54, 4.58)` and that a real 1024-dim embedding is produced, then confirm the existing unit suite (86 tests) still passes.

**Files:**
- Modify: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/pyproject.toml`
- Modify: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/Dockerfile`
- Test: no new automated test file; verification is the fresh-image checks plus the existing `pytest -m "not integration"` suite.

**Interfaces:**
- Consumes: PyTorch CPU wheel index `https://download.pytorch.org/whl/cpu`; existing `sentence-transformers>=5.0`, `transformers>=4.54,<4.58` pins.
- Produces: `dev` and `daemon` images that (a) import `transformers` at a version `>= 4.54` and `< 4.58`, and (b) can run `from finops.modules.embeddings import embed_query; embed_query("hi")` returning a list of 1024 floats. `mcp>=1.2` importable in both.

**TDD steps:**

- [ ] Add the `mcp` dependency. Edit `pyproject.toml`, in the `[project].dependencies` list, add `"mcp>=1.2",` after the `"openai>=1.0",` line:

```toml
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "motor>=3.6",
    "pymongo>=4.8",
    "click>=8.1",
    "httpx>=0.27",
    "sentence-transformers>=5.0",
    "transformers>=4.54,<4.58",
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
    "mcp>=1.2",
]
```

- [ ] Add the CPU-torch install to the Dockerfile `base` stage BEFORE the first `pip install -e "."`. Edit `Dockerfile` so the `base` stage reads exactly:

```dockerfile
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10

RUN --mount=type=cache,target=/root/.cache/pip pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
RUN mkdir -p finops && touch finops/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip pip install -e "."

COPY . .
RUN --mount=type=cache,target=/root/.cache/pip pip install -e "."

EXPOSE 7432
CMD uvicorn finops.daemon.app:app --host 0.0.0.0 --port ${FINOPS_PORT:-7432}

FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/pip pip install -e ".[dev]"
```

  The existing cache-mount + `PIP_RETRIES=10` + `PIP_DEFAULT_TIMEOUT=180` pattern is preserved on every pip line, which handles transient timeouts on the CPU-torch download (still ~200 MB, but far smaller and more reliable than CUDA torch).

- [ ] FALLBACK VARIANT (only if the pinned-index torch install fails resolution). If `sentence-transformers` later rejects the torch version that `--index-url .../cpu` resolves to, do NOT keep the standalone `pip install torch` line. Instead let pip resolve torch as a normal transitive dependency of `-e "."`, but still prefer the CPU index, by removing the standalone torch line and adding an env var before the editable installs:

```dockerfile
ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10 PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
RUN mkdir -p finops && touch finops/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip pip install -e "."
```

  Implementer decision: build with the primary variant first (standalone CPU-torch line). If and only if the build errors on a torch/sentence-transformers version conflict, switch to this `PIP_EXTRA_INDEX_URL` variant and keep whichever produces a working image. Report which variant you kept.

- [ ] Build both images (no cache reliance for correctness; the cache mount is only a speed aid). From the repo root:

```
docker compose build daemon dev
```

  This must complete without timing out. If it times out on the torch download, re-run the same command (the pip cache mount resumes) — but the standalone CPU wheel is small enough that this should build in one pass.

- [ ] Verify the FRESHLY built `daemon` image (target `base`) carries the transformers fix — use the `daemon` service specifically, because it is the image that previously lacked the fix. It has no MongoDB dependency for these checks, so run the check directly:

```
docker compose run --rm --no-deps daemon python -c "import transformers, mcp; v=transformers.__version__; parts=[int(x) for x in v.split('.')[:2]]; assert (4,54) <= tuple(parts) < (4,58), v; print('transformers', v, 'ok; mcp import ok')"
```

  Expect it to print a `transformers 4.5x.y ok; mcp import ok` line and exit 0.

- [ ] Verify a real 1024-dim embedding on the freshly built `daemon` image (this exercises sentence-transformers + torch + the transformers fix end to end; it downloads the model on first run into the shared `hf_cache` volume):

```
docker compose run --rm --no-deps daemon python -c "from finops.modules.embeddings import embed_query; v=embed_query('hi'); assert isinstance(v, list) and len(v)==1024, len(v); print('embed_query ok, dim', len(v))"
```

  Expect `embed_query ok, dim 1024` and exit 0.

- [ ] Run the existing unit suite in the freshly built `dev` image to confirm the build changes did not regress anything. This needs MongoDB, so use the default `dev` service (which depends on `mongodb`):

```
docker compose run --rm dev pytest -m "not integration" -q
```

  Expect the existing suite (about 86 tests) to pass. If the count differs from 86, report the actual number — do not assume.

- [ ] Commit. Message: `build: install CPU-only torch before editable install; add mcp dependency`. Include which Dockerfile variant was kept.

---

### Task 2 — Daemon codebase endpoints (`/codebase/index`, `/codebase/query`)

Add two endpoints to the existing FastAPI daemon that expose the `CodebaseGraph` module over HTTP: `POST /codebase/index` walks a mounted path and indexes every `.py` file, and `POST /codebase/query` runs a vector symbol/NL lookup and returns code slices. The query endpoint projects explicit fields (never returns the raw Mongo document) because the documents carry a non-JSON-serializable `_id` (ObjectId) and an `embedding` array we do not want to ship over the wire. Tests use the existing `httpx.AsyncClient` + `ASGITransport` + `finops_db` pattern with embeddings monkeypatched; they assert shape/counts, not retrieval content (a cold vector index may return an empty `results` list — warm retrieval is covered by the Task 5 integration smoke).

**Files:**
- Modify: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/finops/daemon/app.py`
- Test: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/tests/daemon/test_codebase_endpoints.py`

**Interfaces:**
- Consumes: `get_async_db()`, `load_config(db)` (existing imports in `app.py`); `CodebaseGraph(db, cg_cfg)` with `async index_file(repo_id, file_path, source) -> int` and `async query(repo_id, query_text, k=5) -> list[dict]` (from `finops/modules/codebase_graph.py`). `CodebaseGraph` reads its config from `config["modules"]["codebase_graph"]`.
- Produces:
  - `POST /codebase/index { repo_id?, path? }` → `{ "repo_id": str, "indexed_files": int, "indexed_symbols": int }`.
  - `POST /codebase/query { repo_id?, query?, k? }` → `{ "repo_id": str, "results": [ { "symbol", "type", "file_path", "line_start", "line_end", "source_snippet" }, ... ] }`.

**TDD steps:**

- [ ] Write the failing test file `tests/daemon/test_codebase_endpoints.py`. It monkeypatches the embedding functions where `CodebaseGraph` imports them (`finops.modules.codebase_graph.embed_documents` / `.embed_query`), points `/codebase/index` at the existing `tests/fixtures` directory (which contains `sample.py` with two functions, one class, and its methods), and asserts index counts and query shape:

```python
import pytest
from httpx import AsyncClient, ASGITransport
from finops.daemon.app import app

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("finops.modules.codebase_graph.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("finops.modules.codebase_graph.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
async def client(finops_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_codebase_index_indexes_fixture_dir(client):
    resp = await client.post("/codebase/index", json={"repo_id": "r1", "path": "tests/fixtures"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "r1"
    assert data["indexed_files"] >= 1
    assert data["indexed_symbols"] > 0


async def test_codebase_query_returns_results_shape(client):
    await client.post("/codebase/index", json={"repo_id": "r2", "path": "tests/fixtures"})
    resp = await client.post("/codebase/query", json={"repo_id": "r2", "query": "add two numbers", "k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "r2"
    assert isinstance(data["results"], list)
    for r in data["results"]:
        assert set(r.keys()) == {"symbol", "type", "file_path", "line_start", "line_end", "source_snippet"}


async def test_codebase_query_defaults_are_safe(client):
    resp = await client.post("/codebase/query", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["repo_id"] == "default"
    assert isinstance(data["results"], list)
```

  Note: `path` is relative (`tests/fixtures`); the daemon runs with working dir `/workspace` (the mounted repo), so `Path("tests/fixtures").rglob("*.py")` resolves against the repo root. The `test_codebase_query_returns_results_shape` test asserts the KEY SET of each result, not that any results exist — a freshly created vector index may not be queryable yet, so `results` may be `[]`. That is acceptable here; content correctness is the Task 5 integration concern.

- [ ] Run the test and watch it fail (the endpoints do not exist yet → 404):

```
docker compose run --rm dev pytest tests/daemon/test_codebase_endpoints.py -q
```

  Expect failures (404 / KeyError). Confirm the failure is "endpoint missing", not an import error.

- [ ] Implement the two endpoints. Append to `finops/daemon/app.py`, after the `memory_store` endpoint and before `get_metrics` (any position among the route definitions is fine; place them together for readability):

```python
@app.post("/codebase/index")
async def codebase_index(body: dict):
    from pathlib import Path
    repo_id = body.get("repo_id", "default")
    path = body.get("path", "")
    db = get_async_db()
    config = await load_config(db)
    cg_cfg = config.get("modules", {}).get("codebase_graph", {})
    from finops.modules.codebase_graph import CodebaseGraph
    graph = CodebaseGraph(db, cg_cfg)
    root = Path(path)
    files = 0
    symbols = 0
    for py in root.rglob("*.py"):
        try:
            source = py.read_text(encoding="utf-8")
        except Exception:
            continue
        n = await graph.index_file(repo_id, str(py.relative_to(root)), source)
        if n:
            files += 1
            symbols += n
    return {"repo_id": repo_id, "indexed_files": files, "indexed_symbols": symbols}


@app.post("/codebase/query")
async def codebase_query(body: dict):
    repo_id = body.get("repo_id", "default")
    query = body.get("query", "")
    k = int(body.get("k", 5))
    db = get_async_db()
    config = await load_config(db)
    cg_cfg = config.get("modules", {}).get("codebase_graph", {})
    from finops.modules.codebase_graph import CodebaseGraph
    graph = CodebaseGraph(db, cg_cfg)
    results = await graph.query(repo_id, query, k)
    out = [{
        "symbol": r.get("symbol"), "type": r.get("type"),
        "file_path": r.get("file_path"), "line_start": r.get("line_start"),
        "line_end": r.get("line_end"), "source_snippet": r.get("source_snippet"),
    } for r in results]
    return {"repo_id": repo_id, "results": out}
```

  No comments, no docstrings (these are not MCP tools). The explicit field projection in `/codebase/query` is deliberate: it strips the ObjectId `_id` and the `embedding` array so the response is JSON-serializable and small.

- [ ] Run the test and watch it pass:

```
docker compose run --rm dev pytest tests/daemon/test_codebase_endpoints.py -q
```

  Expect all three tests to pass.

- [ ] Run the full non-integration suite to confirm no regression:

```
docker compose run --rm dev pytest -m "not integration" -q
```

  Expect the previous count plus the 3 new tests to pass.

- [ ] Commit. Message: `feat(daemon): add /codebase/index and /codebase/query endpoints`.

---

### Task 3 — MCP daemon client (`finops/mcp/daemon_client.py`)

Create the `finops/mcp` package and its `daemon_client` module: thin async `httpx` wrappers, one per daemon endpoint the MCP tools need. `_base_url()` reads `FINOPS_DAEMON_URL` (default `http://daemon:7432`) so the same code works on the compose network and can be pointed elsewhere via env. Unit tests monkeypatch `httpx.AsyncClient` with a fake that records the POST path + payload and returns canned JSON, asserting each function posts the right path with the right body and returns parsed JSON, and that the `FINOPS_DAEMON_URL` override is honored.

**Files:**
- Create: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/finops/mcp/__init__.py` (empty)
- Create: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/finops/mcp/daemon_client.py`
- Create: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/tests/mcp/__init__.py` (empty)
- Test: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/tests/mcp/test_daemon_client.py`

**Interfaces:**
- Consumes: env var `FINOPS_DAEMON_URL`; daemon endpoints `/optimize`, `/codebase/index`, `/codebase/query`, `/memory/retrieve`, `/memory/store`.
- Produces (all `async`, all return `dict` = parsed JSON body):
  - `_base_url() -> str`
  - `optimize(prompt, context="", agent_id="default", corpus_id=None, strategy=None) -> dict`
  - `codebase_index(repo_id, path) -> dict`
  - `codebase_query(query, repo_id, k=5) -> dict`
  - `memory_retrieve(agent_id, query) -> dict`
  - `memory_store(agent_id, session_id, turn, response) -> dict`

**TDD steps:**

- [ ] Create the empty package markers `finops/mcp/__init__.py` and `tests/mcp/__init__.py` (both zero-byte). These must exist before the test imports `finops.mcp.daemon_client`.

- [ ] Write the failing test `tests/mcp/test_daemon_client.py`. It replaces `httpx.AsyncClient` with a fake async-context-manager client that records the last `base_url`, `post` path, and `json` payload and returns a fake response whose `.json()` yields a sentinel; it asserts payload shape and that `FINOPS_DAEMON_URL` flows into `base_url`:

```python
import pytest
from finops.mcp import daemon_client


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    calls = []

    def __init__(self, base_url=None, timeout=None):
        self.base_url = base_url
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, path, json=None):
        _FakeClient.calls.append({"base_url": self.base_url, "path": path, "json": json})
        return _FakeResponse({"echoed": path})


@pytest.fixture(autouse=True)
def patch_httpx(monkeypatch):
    _FakeClient.calls = []
    monkeypatch.setattr("finops.mcp.daemon_client.httpx.AsyncClient", _FakeClient)


def _last():
    return _FakeClient.calls[-1]


def test_base_url_default(monkeypatch):
    monkeypatch.delenv("FINOPS_DAEMON_URL", raising=False)
    assert daemon_client._base_url() == "http://daemon:7432"


def test_base_url_override(monkeypatch):
    monkeypatch.setenv("FINOPS_DAEMON_URL", "http://localhost:9999")
    assert daemon_client._base_url() == "http://localhost:9999"


async def test_optimize_posts_optimize(monkeypatch):
    monkeypatch.setenv("FINOPS_DAEMON_URL", "http://localhost:9999")
    out = await daemon_client.optimize("p", "c", agent_id="a1", corpus_id="corp", strategy="s1")
    call = _last()
    assert call["base_url"] == "http://localhost:9999"
    assert call["path"] == "/optimize"
    assert call["json"] == {"prompt": "p", "context": "c", "agent_id": "a1",
                            "corpus_id": "corp", "strategy": "s1", "framework": "claude-code-mcp"}
    assert out == {"echoed": "/optimize"}


async def test_optimize_defaults():
    out = await daemon_client.optimize("hello")
    call = _last()
    assert call["json"] == {"prompt": "hello", "context": "", "agent_id": "default",
                            "corpus_id": None, "strategy": None, "framework": "claude-code-mcp"}
    assert out == {"echoed": "/optimize"}


async def test_codebase_index_posts_index():
    await daemon_client.codebase_index("r1", "/workspace")
    call = _last()
    assert call["path"] == "/codebase/index"
    assert call["json"] == {"repo_id": "r1", "path": "/workspace"}


async def test_codebase_query_posts_query():
    await daemon_client.codebase_query("find add", "r1", k=3)
    call = _last()
    assert call["path"] == "/codebase/query"
    assert call["json"] == {"repo_id": "r1", "query": "find add", "k": 3}


async def test_memory_retrieve_posts_retrieve():
    await daemon_client.memory_retrieve("a1", "what did I say")
    call = _last()
    assert call["path"] == "/memory/retrieve"
    assert call["json"] == {"agent_id": "a1", "query": "what did I say"}


async def test_memory_store_posts_store():
    await daemon_client.memory_store("a1", "s1", "turn text", "response text")
    call = _last()
    assert call["path"] == "/memory/store"
    assert call["json"] == {"agent_id": "a1", "session_id": "s1",
                            "turn": "turn text", "response": "response text"}
```

- [ ] Run the test and watch it fail (module does not exist yet):

```
docker compose run --rm dev pytest tests/mcp/test_daemon_client.py -q
```

  Expect a collection/import error (`ModuleNotFoundError: finops.mcp.daemon_client`).

- [ ] Implement `finops/mcp/daemon_client.py` exactly:

```python
import os
import httpx


def _base_url() -> str:
    return os.getenv("FINOPS_DAEMON_URL", "http://daemon:7432")


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60.0) as c:
        r = await c.post(path, json=payload)
        r.raise_for_status()
        return r.json()


async def optimize(prompt: str, context: str = "", agent_id: str = "default",
                   corpus_id: str | None = None, strategy: str | None = None) -> dict:
    return await _post("/optimize", {"prompt": prompt, "context": context,
                                     "agent_id": agent_id, "corpus_id": corpus_id,
                                     "strategy": strategy, "framework": "claude-code-mcp"})


async def codebase_index(repo_id: str, path: str) -> dict:
    return await _post("/codebase/index", {"repo_id": repo_id, "path": path})


async def codebase_query(query: str, repo_id: str, k: int = 5) -> dict:
    return await _post("/codebase/query", {"repo_id": repo_id, "query": query, "k": k})


async def memory_retrieve(agent_id: str, query: str) -> dict:
    return await _post("/memory/retrieve", {"agent_id": agent_id, "query": query})


async def memory_store(agent_id: str, session_id: str, turn: str, response: str) -> dict:
    return await _post("/memory/store", {"agent_id": agent_id, "session_id": session_id,
                                         "turn": turn, "response": response})
```

  No comments, no docstrings (these are not MCP tools).

- [ ] Run the test and watch it pass:

```
docker compose run --rm dev pytest tests/mcp/test_daemon_client.py -q
```

  Expect all tests to pass.

- [ ] Commit. Message: `feat(mcp): add daemon_client httpx wrappers`.

---

### Task 4 — MCP server + tools (`finops/mcp/server.py`)

Create the FastMCP server exposing five tools, each a one-line delegation to a `daemon_client` function. Logging is configured to stderr only (stdout is the protocol channel). The tool docstrings ARE the model-facing descriptions and are required. `main()` runs stdio transport; `python -m finops.mcp.server` starts it.

Tool-set refinement vs spec Revision 3: `cache_lookup` is DROPPED and `index_codebase` is ADDED. Rationale — the raw `/cache/lookup` endpoint keys on a `prompt_hash` (plus optional embedding) that would not match the default `prompt+scope` cache-key policy the optimizer writes with, so exposing it as a standalone tool would almost always miss; the cache benefit is already delivered inside `optimize_context` via the pipeline's cache-hit short-circuit. `index_codebase` is added because it is a prerequisite for `lookup_symbol` (you must index a repo before you can look symbols up). Net tool set (5): `optimize_context`, `index_codebase`, `lookup_symbol`, `retrieve_memory`, `store_memory`.

Unit tests monkeypatch the `daemon_client` functions with `AsyncMock`s, call each tool coroutine directly, and assert delegation + return value. They also assert all five tools are registered on the `mcp` object — the exact introspection API must be verified in the built image and adapted.

**Files:**
- Create: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/finops/mcp/server.py`
- Test: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/tests/mcp/test_server.py`

**Interfaces:**
- Consumes: `finops.mcp.daemon_client` (Task 3). `from mcp.server.fastmcp import FastMCP`.
- Produces:
  - Module-level `mcp = FastMCP("finops")` with five `@mcp.tool()` coroutines: `optimize_context`, `index_codebase`, `lookup_symbol`, `retrieve_memory`, `store_memory` (signatures below, each returning `dict`).
  - `main() -> None` calling `mcp.run(transport="stdio")`.
  - `__main__` guard calling `main()`.

**TDD steps:**

- [ ] Verify the SDK's tool-introspection API inside the built image BEFORE writing the test, so the registration assertion targets the real API. Run:

```
docker compose run --rm --no-deps dev python -c "import asyncio; from mcp.server.fastmcp import FastMCP; m=FastMCP('t'); \
import inspect; print('has list_tools:', hasattr(m,'list_tools')); \
print('list_tools is coroutine fn:', hasattr(m,'list_tools') and inspect.iscoroutinefunction(m.list_tools))"
```

  Record whether `list_tools` exists and whether it is async. Two supported cases:
  - `list_tools` exists and is async → the test uses `tools = await mcp.list_tools()` and reads each entry's `.name`.
  - `list_tools` is absent or not async in this SDK build → fall back to asserting the five decorated tool functions are importable and callable (the decorator returns the wrapped function object), i.e. `assert callable(server.optimize_context)` etc. Choose the branch that matches the recorded output and delete the other branch from the test before running it. Report which branch you used.

- [ ] Write the failing test `tests/mcp/test_server.py`. It monkeypatches each `daemon_client` function with an `AsyncMock`, imports the server module, calls each tool coroutine directly, and asserts delegation and return value; plus the registration assertion (use the branch chosen above):

```python
import pytest
from unittest.mock import AsyncMock
from finops.mcp import server


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    m = {
        "optimize": AsyncMock(return_value={"tool": "optimize"}),
        "codebase_index": AsyncMock(return_value={"tool": "codebase_index"}),
        "codebase_query": AsyncMock(return_value={"tool": "codebase_query"}),
        "memory_retrieve": AsyncMock(return_value={"tool": "memory_retrieve"}),
        "memory_store": AsyncMock(return_value={"tool": "memory_store"}),
    }
    for name, mock in m.items():
        monkeypatch.setattr(server.daemon_client, name, mock)
    return m


async def test_optimize_context_delegates(patch_client):
    out = await server.optimize_context("p", "c", agent_id="a1", corpus_id="cp", strategy="s1")
    patch_client["optimize"].assert_awaited_once_with("p", "c", "a1", "cp", "s1")
    assert out == {"tool": "optimize"}


async def test_index_codebase_delegates(patch_client):
    out = await server.index_codebase("r1", "/workspace")
    patch_client["codebase_index"].assert_awaited_once_with("r1", "/workspace")
    assert out == {"tool": "codebase_index"}


async def test_lookup_symbol_delegates(patch_client):
    out = await server.lookup_symbol("find add", "r1", k=3)
    patch_client["codebase_query"].assert_awaited_once_with("find add", "r1", 3)
    assert out == {"tool": "codebase_query"}


async def test_retrieve_memory_delegates(patch_client):
    out = await server.retrieve_memory("a1", "what did I say")
    patch_client["memory_retrieve"].assert_awaited_once_with("a1", "what did I say")
    assert out == {"tool": "memory_retrieve"}


async def test_store_memory_delegates(patch_client):
    out = await server.store_memory("a1", "s1", "turn", "resp")
    patch_client["memory_store"].assert_awaited_once_with("a1", "s1", "turn", "resp")
    assert out == {"tool": "memory_store"}


async def test_all_five_tools_registered():
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {"optimize_context", "index_codebase", "lookup_symbol",
                     "retrieve_memory", "store_memory"}
```

  If the introspection probe showed `list_tools` is unavailable/not-async, replace `test_all_five_tools_registered` with:

```python
def test_all_five_tools_exist():
    for name in ("optimize_context", "index_codebase", "lookup_symbol",
                 "retrieve_memory", "store_memory"):
        assert callable(getattr(server, name))
```

  Keep exactly one of these two registration tests.

- [ ] Run the test and watch it fail (module does not exist yet):

```
docker compose run --rm dev pytest tests/mcp/test_server.py -q
```

  Expect `ModuleNotFoundError: finops.mcp.server`.

- [ ] Implement `finops/mcp/server.py` exactly (docstrings are the functional tool descriptions and are REQUIRED; no other comments/docstrings):

```python
import logging
import sys

from mcp.server.fastmcp import FastMCP

from finops.mcp import daemon_client

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("finops")


@mcp.tool()
async def optimize_context(prompt: str, context: str = "", agent_id: str = "default",
                           corpus_id: str | None = None, strategy: str | None = None) -> dict:
    """Route a prompt and its context through the finops optimization pipeline (semantic cache, codebase graph, retrieval, memory, compression). Returns the trimmed optimized_context, tokens_saved, cache_hit, and per-module detail."""
    return await daemon_client.optimize(prompt, context, agent_id, corpus_id, strategy)


@mcp.tool()
async def index_codebase(repo_id: str, path: str) -> dict:
    """Index a repository directory (.py files) into the codebase graph so its symbols can be looked up later. Returns counts of indexed files and symbols."""
    return await daemon_client.codebase_index(repo_id, path)


@mcp.tool()
async def lookup_symbol(query: str, repo_id: str, k: int = 5) -> dict:
    """Retrieve the most relevant code slices for a symbol name or natural-language description, instead of reading whole files. Returns up to k matching symbols with their source snippets."""
    return await daemon_client.codebase_query(query, repo_id, k)


@mcp.tool()
async def retrieve_memory(agent_id: str, query: str) -> dict:
    """Retrieve working, episodic, and semantic memory relevant to a query for an agent."""
    return await daemon_client.memory_retrieve(agent_id, query)


@mcp.tool()
async def store_memory(agent_id: str, session_id: str, turn: str, response: str) -> dict:
    """Store a conversation turn and extract durable facts into the agent's long-term memory."""
    return await daemon_client.memory_store(agent_id, session_id, turn, response)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

  IMPORTANT: `@mcp.tool()` may return a wrapped tool object rather than the raw async function in some SDK versions, which would break `await server.optimize_context(...)` in the delegation tests. Verify by running the introspection probe result together with a quick check:

```
docker compose run --rm --no-deps dev python -c "import inspect, finops.mcp.server as s; print('optimize_context is coroutine fn:', inspect.iscoroutinefunction(s.optimize_context))"
```

  If it prints `True`, the delegation tests work as written. If it prints `False` (the decorator wrapped the callable), the implementer must adapt the delegation tests to call through the SDK's tool-invocation API (e.g. `await mcp.call_tool("optimize_context", {...})` and assert the mock was awaited) instead of calling the module attribute directly — verify the exact call-tool signature in the SDK and adapt. Do NOT change `server.py` to work around this; adapt the test. Report which path was used.

- [ ] Run the test and watch it pass:

```
docker compose run --rm dev pytest tests/mcp/test_server.py -q
```

  Expect all six (or five, if you used the callable-exists registration variant) tests to pass.

- [ ] Confirm stdout discipline: the module must not print to stdout at import time. Verify nothing is written to stdout on import (only stderr logging is allowed):

```
docker compose run --rm --no-deps dev python -c "import finops.mcp.server" 1>/tmp/out.txt 2>/tmp/err.txt; test ! -s /tmp/out.txt && echo "STDOUT CLEAN" || (echo "STDOUT DIRTY:"; cat /tmp/out.txt)
```

  Expect `STDOUT CLEAN`.

- [ ] Run the full non-integration suite:

```
docker compose run --rm dev pytest -m "not integration" -q
```

  Expect all prior tests plus the new server tests to pass.

- [ ] Commit. Message: `feat(mcp): add FastMCP server with five daemon-backed tools`.

---

### Task 5 — Compose `mcp` service + Claude Code config + end-to-end smoke

Add the `mcp` compose service that runs `python -m finops.mcp.server` over stdio and depends on the daemon; write a short README covering how to run the daemon and register the server with Claude Code; and add an integration smoke test (marked `@pytest.mark.integration`) that spawns the MCP server over stdio with the MCP client SDK, initializes, lists tools (asserting the five names), and calls `optimize_context` against a LIVE daemon. Because the smoke needs a live daemon and the real embedding model, it is integration-only and documented as such. A documented fallback path is provided if the stdio client proves environment-fragile.

**Files:**
- Modify: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/docker-compose.yml`
- Create: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/finops-mcp-README.md`
- Create: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/tests/integration/__init__.py` (empty, only if `tests/integration/` does not already exist)
- Test: `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/tests/integration/test_mcp_smoke.py`

**Interfaces:**
- Consumes: the `dev`-target image (Task 1), `finops/mcp/server.py` (Task 4), the daemon service and its `/optimize` endpoint. MCP client SDK (`mcp.client.stdio`, `mcp.client.session` — verify exact import paths in the installed SDK).
- Produces:
  - Compose service `mcp` (stdio server, `depends_on: daemon`, `FINOPS_DAEMON_URL=http://daemon:7432`).
  - `finops-mcp-README.md` with run/registration instructions.
  - Integration smoke asserting the five tool names are advertised and that `optimize_context` returns a dict containing `optimized_context` and `tokens_saved`.

**TDD steps:**

- [ ] Add the `mcp` service to `docker-compose.yml`. Insert this block after the `daemon` service and before the top-level `volumes:` key (match the existing two-space indentation):

```yaml
  mcp:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev
    command: python -m finops.mcp.server
    environment:
      - FINOPS_DAEMON_URL=http://daemon:7432
    volumes:
      - .:/workspace
    depends_on:
      - daemon
```

  Verify the compose file still parses:

```
docker compose config >/dev/null && echo "COMPOSE OK"
```

  Expect `COMPOSE OK`.

- [ ] Write `finops-mcp-README.md`. Content (fill in the absolute repo path where shown; on this machine it is `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI`):

```markdown
# fullFinOps-AI — Claude Code MCP Server

The finops MCP server exposes the token-saving daemon to Claude Code as native tools:
`optimize_context`, `index_codebase`, `lookup_symbol`, `retrieve_memory`, `store_memory`.

It is a Python (FastMCP) server that runs inside the project's Docker container over
stdio and talks HTTP to the finops daemon. No Node toolchain is required.

## Prerequisites

1. Build the images (installs CPU-only torch + all deps):

       docker compose build daemon dev

2. Start the daemon and MongoDB (the daemon must be running before Claude Code
   launches the MCP server — the server reaches it at FINOPS_DAEMON_URL):

       docker compose up -d daemon

   Confirm it is healthy:

       curl -s http://localhost:7432/health

## Register with Claude Code

The MCP server is launched on demand by Claude Code via
`docker compose run --rm -T mcp`. The `-T` disables pseudo-TTY allocation, which is
required for clean stdio JSON-RPC framing.

### Option A — `claude mcp add`

    claude mcp add finops -- docker compose -f /Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/docker-compose.yml run --rm -T mcp

### Option B — `~/.claude.json` (or the project's `.mcp.json`) snippet

    {
      "mcpServers": {
        "finops": {
          "command": "docker",
          "args": [
            "compose",
            "-f",
            "/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/docker-compose.yml",
            "run",
            "--rm",
            "-T",
            "mcp"
          ]
        }
      }
    }

Use the absolute path to this repo's `docker-compose.yml`. Because the MCP server
reads `FINOPS_DAEMON_URL` (baked into the `mcp` compose service as
`http://daemon:7432`) and joins the compose network, it reaches the running daemon
container directly.

## Notes

- The daemon must be up first (`docker compose up -d daemon`); the MCP process is
  short-lived and exits when Claude Code closes the connection.
- stdout is the MCP protocol channel; the server logs only to stderr.
- Index a repo before using `lookup_symbol`: call `index_codebase(repo_id, path)`
  where `path` is a directory inside the mounted `/workspace`.
```

- [ ] Ensure `tests/integration/` is a package. Check whether the directory and its `__init__.py` already exist; create an empty `tests/integration/__init__.py` only if missing:

```
docker compose run --rm --no-deps dev python -c "import os; p='tests/integration/__init__.py'; print('exists' if os.path.exists(p) else 'missing')"
```

  If it prints `missing`, create the empty file `tests/integration/__init__.py`.

- [ ] Verify the MCP client SDK import paths inside the built image BEFORE writing the smoke test (they vary across SDK versions):

```
docker compose run --rm --no-deps dev python -c "from mcp.client.stdio import stdio_client, StdioServerParameters; from mcp.client.session import ClientSession; print('client imports ok')"
```

  If this errors, discover the correct paths (`docker compose run --rm --no-deps dev python -c "import mcp.client, pkgutil; print([m.name for m in pkgutil.iter_modules(mcp.client.__path__)])"`) and adapt the imports in the smoke test accordingly. Report the exact imports used.

- [ ] Write the integration smoke `tests/integration/test_mcp_smoke.py`. It spawns the server as a subprocess over stdio, initializes a session, lists tools (asserting the five names), and calls `optimize_context` against the live daemon. It is marked `@pytest.mark.integration` so the default `-m "not integration"` gate skips it:

```python
import pytest
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

pytestmark = pytest.mark.integration


async def _run_session(callback):
    params = StdioServerParameters(command="python", args=["-m", "finops.mcp.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await callback(session)


async def test_mcp_lists_five_tools():
    async def cb(session):
        return await session.list_tools()
    result = await _run_session(cb)
    names = {t.name for t in result.tools}
    assert names == {"optimize_context", "index_codebase", "lookup_symbol",
                     "retrieve_memory", "store_memory"}


async def test_mcp_optimize_context_roundtrip():
    async def cb(session):
        return await session.call_tool("optimize_context", {
            "prompt": "What is Python?", "context": "some context", "agent_id": "smoke",
        })
    result = await _run_session(cb)
    payload = result.structuredContent if getattr(result, "structuredContent", None) else None
    if payload is None:
        import json
        payload = json.loads(result.content[0].text)
    assert "optimized_context" in payload
    assert "tokens_saved" in payload
```

  Notes for the implementer:
  - The `ClientSession` / `stdio_client` API and the shape of `call_tool`'s return (`.tools`, `.content[0].text`, `.structuredContent`) vary across SDK versions. Verify against the installed SDK (the import probe above plus `dir(result)`) and adapt these accessors. The test defends against both `structuredContent` and text-content shapes; keep whichever the SDK actually produces and simplify.
  - Spawning `python -m finops.mcp.server` from inside the `dev` container works because `dev` mounts the repo and has the package installed. The spawned server reads `FINOPS_DAEMON_URL`; inside the `dev` container the default `http://daemon:7432` resolves on the compose network to the running daemon.

- [ ] Run the integration smoke against a live daemon. First start the daemon (it may take a moment on first run to download the embedding model into the shared `hf_cache` volume), then run only the integration marker:

```
docker compose up -d daemon
docker compose run --rm dev pytest -m integration tests/integration/test_mcp_smoke.py -q
```

  Expect both smoke tests to pass. If the stdio client proves environment-fragile inside the container (e.g. subprocess stdio buffering or SDK API mismatch you cannot resolve), take the FALLBACK path below and clearly report that you did so.

- [ ] FALLBACK (only if the stdio-client smoke cannot be made reliable). Replace the stdio smoke with (a) a lighter live-daemon smoke that calls the `daemon_client` functions directly against the running daemon, and (b) a documented manual verification in the README. The lighter smoke:

```python
import pytest
from finops.mcp import daemon_client

pytestmark = pytest.mark.integration


async def test_daemon_client_optimize_live():
    out = await daemon_client.optimize("What is Python?", "some context", agent_id="smoke")
    assert "optimized_context" in out
    assert "tokens_saved" in out
```

  Run it as:

```
docker compose up -d daemon
docker compose run --rm dev pytest -m integration tests/integration/test_mcp_smoke.py -q
```

  And add to `finops-mcp-README.md` a "Manual verification" section: start the daemon, run `claude mcp add ...` (Option A above), then in Claude Code ask it to call the `optimize_context` tool and confirm a `tokens_saved` field comes back. State in your final report which path (full stdio smoke vs. fallback) you used and why.

- [ ] Confirm the default gate still excludes the integration test and the rest of the suite is green:

```
docker compose run --rm dev pytest -m "not integration" -q
```

  Expect the full non-integration suite to pass and the integration smoke to be deselected.

- [ ] Commit. Message: `feat(mcp): add compose mcp service, Claude Code config docs, and integration smoke`. Note in the commit body whether the full stdio smoke or the daemon_client fallback was used.

---

## Done criteria

- `docker compose build daemon dev` completes reliably; both images carry `transformers` in `[4.54, 4.58)`, CPU torch, and importable `mcp`.
- The daemon exposes `POST /codebase/index` and `POST /codebase/query`.
- `finops/mcp/daemon_client.py` and `finops/mcp/server.py` exist; the server advertises exactly five tools (`optimize_context`, `index_codebase`, `lookup_symbol`, `retrieve_memory`, `store_memory`), logs only to stderr, and runs stdio.
- The `mcp` compose service launches the server; `finops-mcp-README.md` documents daemon startup and Claude Code registration.
- All non-integration tests pass in the container; the integration smoke passes against a live daemon (or the documented fallback does).
