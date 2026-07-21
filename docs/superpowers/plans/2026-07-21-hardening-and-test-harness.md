# Hardening & Test Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the daemon (auth, localhost binding, async offloading, input validation, path allowlisting), repair the dev/test environment, add live-daemon and MCP-protocol integration harnesses, and reconcile docs. Stop before Claude Code e2e (Layer C).

**Architecture:** The FastAPI daemon (`finops/daemon/app.py`) gains an opt-in bearer-token dependency and Pydantic request models; blocking ML/LLM calls move off the event loop via `asyncio.to_thread`. A new `tests/integration/conftest.py` fixture boots a real uvicorn subprocess against the compose `mongodb-test` instance (port 27018), and the MCP stdio smoke tests are fixed to spawn `sys.executable` pointed at that live daemon.

**Tech Stack:** FastAPI/Pydantic v2, motor/pymongo, httpx, pytest + pytest-asyncio (`asyncio_mode = "auto"` — async tests need no decorator), MCP Python SDK stdio client, Docker Compose (`mongodb/mongodb-atlas-local`).

## Global Constraints

- Python `>=3.11`; the repo venv must be rebuilt with `transformers>=4.54,<4.58` (pyproject pin — transformers 5.x breaks voyage-4-nano remote code).
- Auth is **opt-in**: when `FINOPS_API_TOKEN` is unset, behavior is unchanged (backwards compatible for local dev). When set, all endpoints except `/health`, `/metrics`, and `/dashboard*` require `Authorization: Bearer <token>`.
- Non-integration tests must keep passing: run with `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest -m "not integration" -q` (requires `docker compose up -d mongodb-test`).
- Integration tests are marked `@pytest.mark.integration` (marker exists in `pyproject.toml`).
- Do NOT touch `finops/mcp/server.py` tool signatures, module pipeline logic, or the dashboard.
- Commit after every task; conventional-commit messages.

---

### Task 1: Bearer-token auth dependency

**Files:**
- Create: `finops/daemon/auth.py`
- Modify: `finops/daemon/app.py:43` (FastAPI constructor)
- Test: `tests/daemon/test_auth.py`

**Interfaces:**
- Produces: `finops.daemon.auth.require_token(request: Request) -> None` (FastAPI dependency; raises `HTTPException(401)`). Task 2's clients send `Authorization: Bearer $FINOPS_API_TOKEN`.

- [ ] **Step 1: Write the failing test**

```python
# tests/daemon/test_auth.py
import httpx
import pytest
from httpx import ASGITransport


@pytest.fixture
async def client(finops_db):
    from finops.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_no_token_configured_allows_all(client, monkeypatch):
    monkeypatch.delenv("FINOPS_API_TOKEN", raising=False)
    r = await client.get("/config")
    assert r.status_code == 200


async def test_token_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config")
    assert r.status_code == 401


async def test_wrong_token_rejected(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


async def test_correct_token_accepted(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config", headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 200


async def test_health_and_metrics_exempt(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/metrics")).status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/daemon/test_auth.py -v`
Expected: `test_token_required_when_configured` and `test_wrong_token_rejected` FAIL (200 != 401); others pass.

- [ ] **Step 3: Implement auth module and wire it in**

```python
# finops/daemon/auth.py
import os
import secrets

from fastapi import HTTPException, Request

# Paths a browser or liveness probe hits without credentials.
_EXEMPT_PREFIXES = ("/health", "/metrics", "/dashboard")


async def require_token(request: Request) -> None:
    expected = os.getenv("FINOPS_API_TOKEN", "")
    if not expected:
        return
    path = request.url.path
    if any(path == p or path.startswith(p + "/") or path.startswith(p) for p in _EXEMPT_PREFIXES):
        return
    header = request.headers.get("authorization", "")
    provided = header.removeprefix("Bearer ").strip()
    if not (provided and secrets.compare_digest(provided, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token (FINOPS_API_TOKEN)")
```

In `finops/daemon/app.py`, change line 43 (and add imports at top):

```python
from fastapi import Depends, FastAPI
from finops.daemon.auth import require_token
# ...
app = FastAPI(title="efficient Daemon", lifespan=lifespan, dependencies=[Depends(require_token)])
```

(The dashboard router included at line 46 inherits the dependency; its paths are exempt by prefix.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/daemon/ -v`
Expected: all PASS (existing daemon tests unaffected because `FINOPS_API_TOKEN` is unset).

- [ ] **Step 5: Commit**

```bash
git add finops/daemon/auth.py finops/daemon/app.py tests/daemon/test_auth.py
git commit -m "feat(daemon): opt-in bearer-token auth via FINOPS_API_TOKEN"
```

---

### Task 2: Localhost binding + token-aware clients (CLI, compose, MCP client, hooks)

**Files:**
- Modify: `finops/cli/main.py:36-41` (bind host), `docker-compose.yml:56-57` and `docker-compose.yml:60-63` (port binding + token passthrough), `finops/mcp/daemon_client.py:9-13`, `examples/claude-hooks/recall-memory.sh`, `examples/claude-hooks/reindex-on-edit.sh`, `examples/claude-hooks/efficient-autoindex.sh`
- Test: `tests/cli/test_cli.py` (extend), `tests/mcp/test_daemon_client.py` (extend)

**Interfaces:**
- Consumes: `FINOPS_API_TOKEN` convention from Task 1.
- Produces: CLI env override `FINOPS_HOST` (default `127.0.0.1`); `daemon_client._auth_headers() -> dict` used by `_post`.

- [ ] **Step 1: Write failing tests**

Add to `tests/cli/test_cli.py` (match the file's existing mocking style for `subprocess.Popen` — it already mocks it; assert on the argv passed):

```python
def test_start_binds_localhost_by_default(tmp_path, monkeypatch):
    import finops.cli.main as cli_main
    monkeypatch.setattr(cli_main, "PID_FILE", tmp_path / "daemon.pid")
    monkeypatch.delenv("FINOPS_HOST", raising=False)
    calls = {}

    class FakeProc:
        pid = 4242

    def fake_popen(argv, **kwargs):
        calls["argv"] = argv
        return FakeProc()

    monkeypatch.setattr(cli_main.subprocess, "Popen", fake_popen)
    from click.testing import CliRunner
    result = CliRunner().invoke(cli_main.cli, ["start"])
    assert result.exit_code == 0
    host_idx = calls["argv"].index("--host") + 1
    assert calls["argv"][host_idx] == "127.0.0.1"
```

Add to `tests/mcp/test_daemon_client.py`:

```python
async def test_post_sends_bearer_token_when_set(monkeypatch):
    import finops.mcp.daemon_client as dc
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    captured = {}

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"ok": True}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, path, json): return FakeResponse()

    monkeypatch.setattr(dc.httpx, "AsyncClient", FakeClient)
    await dc._post("/x", {})
    assert captured["headers"].get("Authorization") == "Bearer sekret"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/cli/test_cli.py::test_start_binds_localhost_by_default tests/mcp/test_daemon_client.py::test_post_sends_bearer_token_when_set -v`
Expected: FAIL (`0.0.0.0` != `127.0.0.1`; no `headers` kwarg).

- [ ] **Step 3: Implement**

`finops/cli/main.py` — replace lines 36-41:

```python
    port = os.getenv("FINOPS_PORT", "7432")
    host = os.getenv("FINOPS_HOST", "127.0.0.1")
    proc = subprocess.Popen(
        ["uvicorn", "finops.daemon.app:app", "--host", host, "--port", port],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
```

`finops/mcp/daemon_client.py` — replace lines 9-13:

```python
def _auth_headers() -> dict:
    token = os.getenv("FINOPS_API_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60.0, headers=_auth_headers()) as c:
        r = await c.post(path, json=payload)
        r.raise_for_status()
        return r.json()
```

`docker-compose.yml` — daemon service: bind the published port to loopback and pass the token through (empty default keeps auth off):

```yaml
    ports:
      - "127.0.0.1:7432:7432"
    environment:
      - FINOPS_MONGODB_URI=mongodb://mongodb:27017
      - FINOPS_DB_NAME=finops
      - HF_HOME=/root/.cache/huggingface
      - FINOPS_API_TOKEN=${FINOPS_API_TOKEN:-}
```

Also add `- FINOPS_API_TOKEN=${FINOPS_API_TOKEN:-}` to the `mcp` service `environment:` list.

Each of the three hook scripts: right after the shebang/comments, add

```bash
AUTH_ARGS=()
[ -n "${FINOPS_API_TOKEN:-}" ] && AUTH_ARGS=(-H "Authorization: Bearer $FINOPS_API_TOKEN")
```

and insert `"${AUTH_ARGS[@]}"` into every `curl` invocation that hits the daemon, e.g. in `recall-memory.sh`:

```bash
resp=$(curl -s -m 10 -X POST http://localhost:7432/memory/retrieve \
  "${AUTH_ARGS[@]}" \
  -H 'content-type: application/json' \
  -d "$(jq -n --arg a project --arg q "$q" '{agent_id:$a, query:$q}')" 2>/dev/null) || exit 0
```

Same pattern for the `/codebase/index-file` curls in `reindex-on-edit.sh` and `efficient-autoindex.sh` (the `/health` probe in `efficient-autoindex.sh` needs no header — exempt path — but adding it is harmless).

- [ ] **Step 4: Run tests + shellcheck-by-hand**

Run: `venv/bin/python -m pytest tests/cli tests/mcp -v` — expected: all PASS.
Run: `bash -n examples/claude-hooks/recall-memory.sh examples/claude-hooks/reindex-on-edit.sh examples/claude-hooks/efficient-autoindex.sh` — expected: no output (syntax OK).

- [ ] **Step 5: Commit**

```bash
git add finops/cli/main.py finops/mcp/daemon_client.py docker-compose.yml examples/claude-hooks/
git commit -m "feat: localhost-by-default binding + bearer-token support in CLI, compose, MCP client, hooks"
```

---

### Task 3: Offload blocking ML/LLM work with asyncio.to_thread

**Files:**
- Modify: `finops/modules/semantic_cache.py:66,124`, `finops/modules/agent_memory.py:112,132,177-180`, `finops/modules/context_compressor.py:55-59`
- Test: `tests/modules/test_event_loop_not_blocked.py`

**Interfaces:**
- Consumes: existing module-level `embed_query`/`embed_documents` imports (tests monkeypatch these names on each module — keep the imports, wrap at call sites only).
- Produces: no signature changes; all `process`/`store`/`store_turn` remain `async` with identical returns.

- [ ] **Step 1: Write the failing test**

```python
# tests/modules/test_event_loop_not_blocked.py
import asyncio
import time

from finops.modules._base import OptimizeRequest
from finops.modules.semantic_cache import SemanticCache


async def test_slow_embedding_does_not_block_event_loop(finops_db, monkeypatch):
    def slow_embed_query(text):
        time.sleep(0.5)  # simulates model inference on CPU
        return [0.1] * 1024

    import finops.modules.semantic_cache as sc
    monkeypatch.setattr(sc, "embed_query", slow_embed_query)

    cache = SemanticCache(finops_db, {})
    request = OptimizeRequest(prompt="p", context="", agent_id="a", framework="t", corpus_id=None)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.02)
            ticks += 1

    await asyncio.gather(cache.process(request), heartbeat())
    # If embed_query ran on the event loop, the heartbeat stalls for 0.5s
    # and completes far fewer iterations during the overlap.
    assert ticks >= 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/modules/test_event_loop_not_blocked.py -v`
Expected: FAIL (ticks near 0 — the sync sleep blocks the loop). Note: `finops_db` requires mongodb-test up.

- [ ] **Step 3: Implement**

Add `import asyncio` to the imports of all three modules, then:

`semantic_cache.py:66`: `embedding = await asyncio.to_thread(embed_query, key)`
`semantic_cache.py:124`: `embedding = (await asyncio.to_thread(embed_documents, [key]))[0]`

`agent_memory.py:112` and `agent_memory.py:132`: `embedding = await asyncio.to_thread(embed_query, query)`
`agent_memory.py:177-180` (in `_extract_and_store_facts`):

```python
        facts = await asyncio.to_thread(self._extract_facts, turn, response)
        if not facts:
            return
        fact_embeddings = await asyncio.to_thread(embed_documents, facts)
```

`context_compressor.py:55-59`:

```python
        rate = 1.0 / self._target_ratio
        result = await asyncio.to_thread(
            lambda: _get_compressor().compress_prompt(
                [request.context],
                rate=rate,
                force_tokens=["\n", "?"],
            )
        )
```

(`_get_compressor()` stays inside the thread on purpose — first-call model load is also blocking.)

- [ ] **Step 4: Run full module + daemon suites**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/modules tests/daemon -q`
Expected: all PASS (existing monkeypatched-embedding tests still work — `to_thread` calls the patched module attribute).

- [ ] **Step 5: Commit**

```bash
git add finops/modules/semantic_cache.py finops/modules/agent_memory.py finops/modules/context_compressor.py tests/modules/test_event_loop_not_blocked.py
git commit -m "fix(modules): run embeddings, compression, and fact-extraction LLM calls off the event loop"
```

---

### Task 4: Pydantic request models, config-key validation, index-path allowlist

**Files:**
- Create: `finops/daemon/schemas.py`
- Modify: `finops/daemon/app.py` (all `body: dict` handlers + `/codebase/index` allowlist + `/cache/lookup` k-free), `finops/daemon/config.py` (patch validation)
- Test: `tests/daemon/test_validation.py`

**Interfaces:**
- Produces: `finops.daemon.schemas` models listed below; `finops.daemon.config.validate_patch(patch: dict) -> None` (raises `ValueError` on `$`/`.` keys or unknown top-level keys). Existing JSON request/response shapes unchanged for valid input.

- [ ] **Step 1: Write the failing tests**

```python
# tests/daemon/test_validation.py
import httpx
import pytest
from httpx import ASGITransport


@pytest.fixture
async def client(finops_db):
    from finops.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_config_rejects_operator_keys(client):
    r = await client.put("/config", json={"$set": {"strategy": "x"}})
    assert r.status_code == 422


async def test_config_rejects_dotted_keys(client):
    r = await client.put("/config", json={"modules.semantic_cache.enabled": False})
    assert r.status_code == 422


async def test_config_rejects_unknown_top_level_key(client):
    r = await client.put("/config", json={"evil": 1})
    assert r.status_code == 422


async def test_config_accepts_valid_patch(client):
    r = await client.put("/config", json={"strategy": "compose_then_compress"})
    assert r.status_code == 200


async def test_codebase_query_bounds_k(client):
    r = await client.post("/codebase/query", json={"repo_id": "r", "query": "q", "k": 10**9})
    assert r.status_code == 422
    r = await client.post("/codebase/query", json={"repo_id": "r", "query": "q", "k": "x"})
    assert r.status_code == 422


async def test_codebase_index_rejects_unlisted_path(client, tmp_path):
    r = await client.post("/codebase/index", json={"repo_id": "r", "path": str(tmp_path)})
    assert r.status_code == 403


async def test_codebase_index_allows_configured_root(client, tmp_path, monkeypatch):
    monkeypatch.setenv("FINOPS_ALLOWED_INDEX_ROOTS", str(tmp_path))
    (tmp_path / "m.py").write_text("def f():\n    return 1\n")
    r = await client.post("/codebase/index", json={"repo_id": "r", "path": str(tmp_path)})
    assert r.status_code == 200
    assert r.json()["indexed_files"] == 1


async def test_optimize_rejects_non_string_prompt(client):
    r = await client.post("/optimize", json={"prompt": ["not", "a", "string"]})
    assert r.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/daemon/test_validation.py -v`
Expected: FAIL (raw dicts accept everything; unlisted path returns 200 today).

- [ ] **Step 3: Implement**

```python
# finops/daemon/schemas.py
from pydantic import BaseModel, Field


class OptimizeBody(BaseModel):
    prompt: str = ""
    context: str = ""
    agent_id: str = "default"
    framework: str = "unknown"
    corpus_id: str | None = None
    strategy: str | None = None


class CompleteBody(OptimizeBody):
    provider: str = "anthropic"
    model: str = ""
    session_id: str = "default"


class CacheStoreBody(BaseModel):
    prompt: str = ""
    response: str = ""
    framework: str = "unknown"
    model: str = ""
    tokens_saved: int = 0
    agent_id: str = ""
    corpus_id: str = ""


class MemoryRetrieveBody(BaseModel):
    agent_id: str = "default"
    query: str = ""


class MemoryStoreBody(BaseModel):
    agent_id: str = "default"
    session_id: str = "default"
    turn: str = ""
    response: str = ""


class CodebaseIndexBody(BaseModel):
    repo_id: str = "default"
    path: str = ""


class CodebaseQueryBody(BaseModel):
    repo_id: str = "default"
    query: str = ""
    k: int = Field(5, ge=1, le=50)


class CodebaseIndexFileBody(BaseModel):
    repo_id: str = "default"
    file_path: str = ""
    source: str = ""


class CodebaseReferencesBody(BaseModel):
    repo_id: str = "default"
    symbol: str = ""
```

`finops/daemon/config.py` — add after `DEFAULT_CONFIG`:

```python
_PATCHABLE_KEYS = {
    "modules", "strategy", "embedding_model", "embedding_dimensions",
    "cost_per_input_token", "cost_per_output_token",
}


def _check_keys(obj: dict) -> None:
    for k, v in obj.items():
        if not isinstance(k, str) or k.startswith("$") or "." in k:
            raise ValueError(f"invalid config key: {k!r}")
        if isinstance(v, dict):
            _check_keys(v)


def validate_patch(patch: dict) -> None:
    unknown = set(patch) - _PATCHABLE_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    _check_keys(patch)
```

`finops/daemon/app.py` — rewrite the handlers' signatures to consume the models (each handler body then reads `body.prompt` etc. instead of `body.get(...)`; keep all downstream logic identical). Representative diffs:

```python
from fastapi import Depends, FastAPI, HTTPException
from finops.daemon import schemas
from finops.daemon.config import load_config, save_config, validate_patch


@app.put("/config")
async def put_config(patch: dict):
    patch.pop("_id", None)
    patch.pop("updated_at", None)
    try:
        validate_patch(patch)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db = get_async_db()
    config = await save_config(db, patch)
    config.pop("_id", None)
    return config


@app.post("/optimize")
async def post_optimize(body: schemas.OptimizeBody):
    db = get_async_db()
    config = await load_config(db)
    strategy = get_strategy(body.strategy or config.get("strategy"))
    from finops.daemon.router import ModulePipeline
    pipeline = ModulePipeline(db, config.get("modules", {}), strategy)
    request = OptimizeRequest(
        prompt=body.prompt, context=body.context, agent_id=body.agent_id,
        framework=body.framework, corpus_id=body.corpus_id,
    )
    result = await pipeline.run(request)
    from finops.daemon.metrics import record_module_events
    await record_module_events(db, result["module_results"])
    return result
```

Apply the same mechanical conversion to `/complete` (`schemas.CompleteBody`; `body.get("provider", "anthropic")` → `body.provider`, `body.get("model", "")` → `body.model`, `body.get("session_id", "default")` → `body.session_id`), `/cache/store` (`CacheStoreBody`), `/memory/retrieve` (`MemoryRetrieveBody`), `/memory/store` (`MemoryStoreBody`), `/codebase/query` (`CodebaseQueryBody` — delete the `int(body.get("k", 5))` cast), `/codebase/index-file` (`CodebaseIndexFileBody`), `/codebase/references` (`CodebaseReferencesBody`).

`/codebase/index` gets the model **plus** the allowlist:

```python
import os
from pathlib import Path


def _allowed_index_roots(cg_cfg: dict) -> list[Path]:
    roots = [Path(p).resolve() for p in cg_cfg.get("repo_paths", [])]
    env = os.getenv("FINOPS_ALLOWED_INDEX_ROOTS", "")
    roots += [Path(p).resolve() for p in env.split(":") if p]
    return roots


@app.post("/codebase/index")
async def codebase_index(body: schemas.CodebaseIndexBody):
    db = get_async_db()
    config = await load_config(db)
    cg_cfg = config.get("modules", {}).get("codebase_graph", {})
    root = Path(body.path).resolve() if body.path else None
    if root is None or not root.is_dir():
        return {"repo_id": body.repo_id, "indexed_files": 0, "indexed_symbols": 0}
    allowed = _allowed_index_roots(cg_cfg)
    if not any(root == r or root.is_relative_to(r) for r in allowed):
        raise HTTPException(
            status_code=403,
            detail="path not under an allowed index root "
                   "(configure modules.codebase_graph.repo_paths or FINOPS_ALLOWED_INDEX_ROOTS)",
        )
    from finops.modules.codebase_graph import CodebaseGraph
    graph = CodebaseGraph(db, cg_cfg)
    await graph.clear_repo(body.repo_id)
    files = 0
    symbols = 0
    for py in root.rglob("*.py"):
        try:
            source = py.read_text(encoding="utf-8")
        except Exception:
            continue
        n = await graph.index_file(body.repo_id, str(py.relative_to(root)), source)
        if n:
            files += 1
            symbols += n
    return {"repo_id": body.repo_id, "indexed_files": files, "indexed_symbols": symbols}
```

Compose note: add `- FINOPS_ALLOWED_INDEX_ROOTS=/workspace` to the `dev` service environment (the daemon service has no source mount, so it gets no roots — `index_codebase` via MCP then requires explicit configuration, which is the intent).

- [ ] **Step 4: Run the daemon suite; fix fallout**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/daemon -v`
Expected: `test_validation.py` PASSES. Existing codebase-endpoint tests that POST `/codebase/index` with a real tmp dir will now get 403 — update those tests to set `FINOPS_ALLOWED_INDEX_ROOTS` (monkeypatch) to the tmp path. Any test PUTting unknown config keys must switch to a valid key. Do not weaken the new checks to make old tests pass.

- [ ] **Step 5: Run the full non-integration suite**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest -m "not integration" -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add finops/daemon/schemas.py finops/daemon/app.py finops/daemon/config.py docker-compose.yml tests/daemon/
git commit -m "feat(daemon): Pydantic request validation, config patch whitelist, codebase-index path allowlist"
```

---

### Task 5: Repair venv, test-runner.sh, drop verify_scaffold.py

**Files:**
- Modify: `test-runner.sh` (full rewrite)
- Delete: `verify_scaffold.py` (redundant with `tests/test_scaffold.py`)
- No new tests — this task's deliverable is a working environment, verified by running the suite.

**Interfaces:**
- Produces: `./test-runner.sh` (unit) and `./test-runner.sh --integration`; a rebuilt `venv/` whose interpreter path points at this repo.

- [ ] **Step 1: Rebuild the venv in place**

```bash
cd /Users/matt.johnson/ClaudeCodeRepo/efficient
rm -rf venv
python3.12 -m venv venv
venv/bin/pip install -e ".[dev]"
```

Verify: `venv/bin/python -c "import transformers; print(transformers.__version__)"` — expected: a `4.5x` version `<4.58`. Verify: `venv/bin/pytest --version` runs (shebang fixed).

- [ ] **Step 2: Rewrite test-runner.sh**

```bash
#!/usr/bin/env bash
# Run the test suite against the dedicated mongodb-test instance (port 27018).
# Usage: ./test-runner.sh [--integration]
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d --wait mongodb-test
export FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true"

if [ "${1:-}" = "--integration" ]; then
    exec venv/bin/python -m pytest -v
else
    exec venv/bin/python -m pytest -m "not integration" -q
fi
```

```bash
chmod +x test-runner.sh
rm verify_scaffold.py
```

- [ ] **Step 3: Run it**

Run: `./test-runner.sh`
Expected: full non-integration suite PASSES (~134+ tests).

- [ ] **Step 4: Commit**

```bash
git add test-runner.sh
git rm verify_scaffold.py
git commit -m "chore: rewrite test-runner.sh for this repo + compose mongodb-test; drop verify_scaffold.py"
```

(venv/ is untracked — confirm with `git status` before committing.)

---

### Task 6: Layer A — live-daemon integration fixture and tests

**Files:**
- Create: `tests/integration/conftest.py`, `tests/integration/test_live_daemon.py`

**Interfaces:**
- Produces: session fixture `live_daemon -> str` (base URL of a real uvicorn subprocess wired to mongodb-test, DB `finops_live_test`). Task 7 consumes it.

- [ ] **Step 1: Write the fixture and failing tests**

```python
# tests/integration/conftest.py
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from pymongo import MongoClient

MONGO_URI = os.getenv("FINOPS_TEST_MONGODB_URI", "mongodb://localhost:27018/?directConnection=true")
LIVE_DB = "finops_live_test"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_daemon():
    port = _free_port()
    env = {
        **os.environ,
        "FINOPS_MONGODB_URI": MONGO_URI,
        "FINOPS_DB_NAME": LIVE_DB,
    }
    env.pop("FINOPS_API_TOKEN", None)  # keep the harness auth-free
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "finops.daemon.app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120  # lifespan creates search indexes; first run is slow
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"daemon exited early with code {proc.returncode}")
            try:
                if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise TimeoutError("live daemon did not become healthy within 120s")
            time.sleep(0.5)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=15)
        MongoClient(MONGO_URI, directConnection=True).drop_database(LIVE_DB)
```

```python
# tests/integration/test_live_daemon.py
import httpx
import pytest

pytestmark = pytest.mark.integration


def test_lifespan_boots_and_health_reports_ok(live_daemon):
    r = httpx.get(f"{live_daemon}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_config_seeded_by_lifespan(live_daemon):
    r = httpx.get(f"{live_daemon}/config", timeout=5.0)
    assert r.status_code == 200
    assert "modules" in r.json()


def test_optimize_end_to_end(live_daemon):
    r = httpx.post(
        f"{live_daemon}/optimize",
        json={"prompt": "What is a mutex?", "context": "", "agent_id": "live-harness"},
        timeout=300.0,  # first call loads the embedding model
    )
    assert r.status_code == 200
    body = r.json()
    assert "optimized_context" in body
    assert "module_results" in body


def test_validation_enforced_on_live_daemon(live_daemon):
    r = httpx.post(f"{live_daemon}/codebase/query",
                   json={"repo_id": "r", "query": "q", "k": 10**9}, timeout=5.0)
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify current state**

Run: `docker compose up -d --wait mongodb-test && FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/integration/test_live_daemon.py -v`
Expected: PASS if Tasks 1-5 are done (this is harness code — the "failing" phase is any boot error it surfaces; fix forward). First run downloads voyage-4-nano (~minutes).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/conftest.py tests/integration/test_live_daemon.py
git commit -m "test: live-daemon integration harness (real uvicorn + mongodb-test) exercising lifespan"
```

---

### Task 7: Layer B — fix MCP stdio smoke tests and add per-tool roundtrips

**Files:**
- Modify: `tests/integration/test_mcp_smoke.py` (fix interpreter + wire to `live_daemon`)
- Create: `tests/integration/test_mcp_tools.py`

**Interfaces:**
- Consumes: `live_daemon` fixture (Task 6); `finops.mcp.server` module entry point (unchanged).

- [ ] **Step 1: Fix the smoke test**

Replace `tests/integration/test_mcp_smoke.py` lines 1-15 with:

```python
import json
import os
import sys

import pytest
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

pytestmark = pytest.mark.integration


async def _run_session(daemon_url, callback):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "finops.mcp.server"],
        env={**os.environ, "FINOPS_DAEMON_URL": daemon_url},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await callback(session)
```

and thread the fixture through both tests: `async def test_mcp_lists_seven_tools(live_daemon):` → `await _run_session(live_daemon, cb)` (same for the roundtrip test).

- [ ] **Step 2: Write the per-tool roundtrip tests**

```python
# tests/integration/test_mcp_tools.py
import json

import pytest

from tests.integration.test_mcp_smoke import _run_session

pytestmark = pytest.mark.integration

SAMPLE_SOURCE = '''\
def greet(name):
    return helper(name)


def helper(name):
    return f"hello {name}"
'''


def _payload(result):
    payload = getattr(result, "structuredContent", None)
    if isinstance(payload, dict):
        return payload
    return json.loads(result.content[0].text)


async def test_memory_store_then_retrieve(live_daemon):
    async def cb(session):
        await session.call_tool("store_memory", {
            "agent_id": "mcp-harness", "session_id": "s1",
            "turn": "remember the port is 7432", "response": "noted: port 7432",
        })
        return await session.call_tool("retrieve_memory", {
            "agent_id": "mcp-harness", "query": "which port",
        })
    payload = _payload(await _run_session(live_daemon, cb))
    assert "working" in payload
    assert any("7432" in m.get("content", "") for m in payload["working"])


async def test_reindex_then_lookup_then_references(live_daemon):
    async def cb(session):
        await session.call_tool("reindex_file", {
            "repo_id": "mcp-harness", "file_path": "sample.py", "source": SAMPLE_SOURCE,
        })
        lookup = await session.call_tool("lookup_symbol", {
            "query": "greet", "repo_id": "mcp-harness",
        })
        refs = await session.call_tool("find_references", {
            "repo_id": "mcp-harness", "symbol": "helper",
        })
        return lookup, refs
    lookup, refs = await _run_session(live_daemon, cb)
    lookup_payload = _payload(lookup)
    assert any(r.get("symbol") == "greet" for r in lookup_payload.get("results", []))
    refs_payload = _payload(refs)
    assert "callers" in refs_payload
```

Note: `lookup_symbol` uses vector search — if the `code_symbols` search index needs time to become queryable, reuse `tests.conftest.wait_for_queryable` against the live DB's collection before asserting (import `MongoClient` + the collection name from `finops.db.collections`); add it only if the test proves flaky.

- [ ] **Step 3: Run the Layer B suite**

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest tests/integration/test_mcp_smoke.py tests/integration/test_mcp_tools.py -v`
Expected: all PASS (daemon spawned once per session by the fixture).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_mcp_smoke.py tests/integration/test_mcp_tools.py
git commit -m "test: MCP stdio protocol harness — fixed interpreter/env wiring + per-tool roundtrips"
```

---

### Task 8: Docs reconciliation

**Files:**
- Create: `README.md`
- Modify: `efficient-mcp-README.md:4,32,43`, `finops/daemon/config.py:16` (drop `benchmark_runner`), `examples/claude-hooks/README.md` (token note)
- Test: existing suite (config-shape assertions may reference `benchmark_runner`)

- [ ] **Step 1: Drop the phantom module from DEFAULT_CONFIG**

Delete `finops/daemon/config.py:16` (`"benchmark_runner": {...}`). Then:

Run: `grep -rn benchmark_runner finops tests dashboard docs/superpowers/plans` and update any test asserting its presence (assert absence instead) and any dashboard copy listing it.

Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest -m "not integration" -q` — expected: all PASS. Note: an existing Mongo config doc will still contain the stale key until the DB is reset; that's acceptable (config load merges nothing — document in README that `docker compose down -v` resets config).

- [ ] **Step 2: Fix efficient-mcp-README.md**

- Line 4: list all **7** tools: `optimize_context`, `index_codebase`, `lookup_symbol`, `find_references`, `retrieve_memory`, `store_memory`, `reindex_file`.
- Lines 32 and 43: replace `/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/docker-compose.yml` with `<absolute path to this repo>/docker-compose.yml` and show the concrete example as `$(pwd)/docker-compose.yml` from the repo root.
- Add a `## Security` section: daemon binds `127.0.0.1` by default (`FINOPS_HOST` to override); set `FINOPS_API_TOKEN` to require a bearer token (compose passes it through to daemon and mcp services); `index_codebase` only indexes paths under `modules.codebase_graph.repo_paths` or `FINOPS_ALLOWED_INDEX_ROOTS`.
- In Notes, mention that `/codebase/index-file` (used by the hooks) is the mount-free path and needs no allowlist.

- [ ] **Step 3: Write the root README.md**

Content requirements (write it fully, ~60-90 lines): project name `efficient` (package `finops`, CLI `efficient` — explain the naming split); one-paragraph description (token-saving daemon + 5 modules + MCP integration for Claude Code); Quick start (`docker compose build daemon dev && docker compose up -d daemon`, `curl -s http://localhost:7432/health`); local dev (`python3.12 -m venv venv && venv/bin/pip install -e ".[dev]"`, `venv/bin/python -m finops.cli.main start` or `efficient start`, first-run `efficient warmup` note); Testing (`./test-runner.sh`, `./test-runner.sh --integration`, requires Docker; integration downloads voyage-4-nano on first run); Security section (same three env vars as Step 2); Claude Code integration pointers to `efficient-mcp-README.md`, `examples/claude-hooks/README.md`, and `scripts/install-to-project.sh`; note that `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` are only needed for `/complete` and fact extraction (see `.env.example`); link to `docs/superpowers/specs/2026-06-30-fullfinops-ai-design.md` with a caveat that Rev 2/3 headers supersede parts of the body.

- [ ] **Step 4: Add token note to hooks README**

In `examples/claude-hooks/README.md`, add one paragraph: if the daemon has `FINOPS_API_TOKEN` set, export the same variable in the environment Claude Code runs in — the hooks attach it as a bearer header automatically; also mention running `efficient warmup` once before first SessionStart to avoid a slow cold index.

- [ ] **Step 5: Verify and commit**

Run: `grep -rn fullFinOps-AI README.md efficient-mcp-README.md examples scripts` — expected: no matches.
Run: `FINOPS_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" venv/bin/python -m pytest -m "not integration" -q` — expected: all PASS.

```bash
git add README.md efficient-mcp-README.md finops/daemon/config.py examples/claude-hooks/README.md tests dashboard
git commit -m "docs: root README, 7-tool + auth docs, drop phantom benchmark_runner config"
```

---

## Final verification

- [ ] `./test-runner.sh` → all non-integration tests pass.
- [ ] `./test-runner.sh --integration` → integration suite passes (mongodb-test up; models cached after first run).
- [ ] `FINOPS_API_TOKEN=x venv/bin/python -m uvicorn finops.daemon.app:app --port 7433` then `curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:7433/config` → `401`; with `-H 'Authorization: Bearer x'` → `200`.
