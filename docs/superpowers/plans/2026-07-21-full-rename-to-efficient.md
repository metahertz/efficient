# Full Rename to `efficient` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the tool internally from `finops` to `efficient` — package, imports, env vars, DB names, PID dir, docker, hooks, docs — as one clean-break sweep per the approved spec (`docs/superpowers/specs/2026-07-21-full-rename-to-efficient-design.md`).

**Architecture:** `git mv finops efficient`, then a case-preserving textual sweep (`FINOPS` → `EFFICIENT`, `finops` → `efficient`) over all tracked files except the historical `docs/superpowers/` and `.superpowers/` trees, followed by a manual prose pass on current-facing docs. The existing 169-test suite is the correctness gate; there is no new behavior, so no new unit tests — the "failing test" is the final grep audit and the suite run against the renamed package.

**Tech Stack:** git, perl one-liner (portable in-place edit on macOS), pip editable install, pytest, docker compose.

## Global Constraints

- Clean break: no `FINOPS_*` fallback reads anywhere.
- Historical docs untouched: `docs/superpowers/plans/*`, `docs/superpowers/specs/2026-06-30-*`, `.superpowers/sdd/*`.
- DB names become `efficient`, `efficient_test`, `efficient_live_test`; PID file `~/.efficient/daemon.pid`.
- Full suite (`./test-runner.sh --integration`) must pass: 169 tests.
- Mongo collection names and docker service/volume names unchanged.

---

### Task 1: Mechanical sweep + suite gate

**Files:**
- Rename: `finops/` → `efficient/` (git mv)
- Modify (via sweep): everything `git ls-files` returns except `docs/superpowers/**` and `.superpowers/**` — includes `tests/**`, `pyproject.toml`, `docker-compose.yml`, `Dockerfile`, `.env.example`, `.devcontainer/devcontainer.json`, `examples/claude-hooks/*`, `scripts/install-to-project.sh`, `dashboard/*`, `test-runner.sh`, `README.md`, `efficient-mcp-README.md`

**Interfaces:**
- Produces: package `efficient` (`efficient.daemon.app:app`, `efficient.cli.main:cli`, `efficient.mcp.server`), env vars `EFFICIENT_*`, DBs `efficient*`. Task 2 relies on these names in docs.

- [ ] **Step 1: Rename the package**

```bash
git mv finops efficient
```

- [ ] **Step 2: Case-preserving sweep over tracked files (history excluded)**

```bash
git ls-files -z \
  | grep -zv '^docs/superpowers/' \
  | grep -zv '^\.superpowers/' \
  | xargs -0 grep -lZ -i finops -- 2>/dev/null \
  | xargs -0 perl -pi -e 's/FINOPS/EFFICIENT/g; s/finops/efficient/g; s/FinOps/Efficient/g'
```

- [ ] **Step 3: Refresh the editable install (package dir changed)**

```bash
venv/bin/pip install -qe ".[dev]"
venv/bin/python -c "import efficient; print(efficient.__name__)"
```

Expected: `efficient`.

- [ ] **Step 4: Audit — old names must be gone from live code**

```bash
grep -riIn finops . \
  --exclude-dir=venv --exclude-dir=.git --exclude-dir=.pytest_cache \
  --exclude-dir=.superpowers --exclude-dir=docs | grep -v '^Binary'
```

Expected: no output. Also confirm no stray `uvicorn finops...` or `~/.finops` survived:

```bash
grep -rn "\.finops\|finops\." efficient tests scripts examples dashboard 2>/dev/null
```

Expected: no output.

- [ ] **Step 5: Run the full suite**

```bash
docker compose up -d --wait mongodb-test
./test-runner.sh --integration
```

Expected: `169 passed`. (test-runner.sh now exports `EFFICIENT_TEST_MONGODB_URI`; tests/conftest.py reads the same name post-sweep.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor!: rename package/env/db/paths finops -> efficient (clean break)"
```

---

### Task 2: Docs prose pass + CLI smoke + push

**Files:**
- Modify: `README.md`, `efficient-mcp-README.md`, `examples/claude-hooks/README.md`, `examples/claude-hooks/CLAUDE.md` (prose review only — sweep already changed tokens)

**Interfaces:**
- Consumes: names from Task 1 (`EFFICIENT_*`, `efficient.cli.main:cli`, DB `efficient`).

- [ ] **Step 1: Review the swept docs for awkward prose**

Read the post-sweep diff of the four docs. Fix by hand:
- README's former naming-split paragraph ("the package is called `finops`...") now reads redundantly — replace with one line: `> Historical note: before 2026-07-21 the package and env vars were named "finops" (project "fullFinOps-AI"); dated documents under docs/superpowers/ retain that name.`
- Any sentence the mechanical sweep made ungrammatical (e.g. "Efficient" mid-sentence where prose meant the FinOps discipline, not the tool).

- [ ] **Step 2: CLI smoke against the test MongoDB**

```bash
EFFICIENT_MONGODB_URI="mongodb://localhost:27018/?directConnection=true" \
EFFICIENT_DB_NAME=efficient_cli_smoke EFFICIENT_PORT=7434 venv/bin/efficient start
sleep 8
EFFICIENT_DAEMON_URL=http://127.0.0.1:7434 venv/bin/efficient status
EFFICIENT_PORT=7434 venv/bin/efficient stop
```

Expected: `Daemon started (PID ...)`, `● daemon running version=0.1.0` with module list, `Daemon stopped`. Then clean up: drop `efficient_cli_smoke` via mongosh or pymongo one-liner:

```bash
venv/bin/python -c "from pymongo import MongoClient; MongoClient('mongodb://localhost:27018/?directConnection=true', directConnection=True).drop_database('efficient_cli_smoke')"
```

- [ ] **Step 3: Final audit + non-integration suite re-run**

```bash
grep -riIn finops . --exclude-dir=venv --exclude-dir=.git --exclude-dir=.pytest_cache --exclude-dir=.superpowers --exclude-dir=docs | grep -v 'Historical note'
./test-runner.sh
```

Expected: empty grep; `155 passed`.

- [ ] **Step 4: Commit and push**

```bash
git add README.md efficient-mcp-README.md examples/claude-hooks/
git commit -m "docs: prose cleanup after finops -> efficient rename"
git push
```

---

## Final verification

- [ ] `./test-runner.sh --integration` → 169 passed on the renamed tree.
- [ ] `venv/bin/efficient --help` lists start/stop/status/warmup.
- [ ] Grep audit empty outside `docs/superpowers/` + `.superpowers/`.
