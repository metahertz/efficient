# Full Rename to `efficient` — Design

**Date:** 2026-07-21
**Status:** Approved

## Goal

Eliminate the `finops` name everywhere it is functionally load-bearing, so the tool is `efficient` internally and externally. Clean break: no compatibility fallbacks. Dated historical documents keep the old name.

## Decisions (user-approved)

1. **Clean break** — `FINOPS_*` env vars, `~/.finops`, and default DB names are renamed with no fallback reads. Anyone with an existing setup re-exports vars and re-indexes.
2. **History preserved** — `docs/superpowers/plans/*`, the 2026-06-30 design spec, and `.superpowers/sdd/*` are dated records and are not rewritten.

## Rename map

| Old | New |
|---|---|
| `finops/` package (and all `import finops` / `from finops...`) | `efficient/` |
| `pyproject` `packages = ["finops"]`, script `"finops.cli.main:cli"` | `["efficient"]`, `"efficient.cli.main:cli"` |
| `FINOPS_MONGODB_URI`, `FINOPS_DB_NAME`, `FINOPS_TEST_MONGODB_URI`, `FINOPS_DAEMON_URL`, `FINOPS_API_TOKEN`, `FINOPS_HOST`, `FINOPS_PORT`, `FINOPS_ALLOWED_INDEX_ROOTS`, `FINOPS_EMBEDDING_MODEL` | same names with `EFFICIENT_` prefix |
| default DB `finops`; test DBs `finops_test`, `finops_live_test`; auth-check scratch `finops_authcheck` | `efficient`, `efficient_test`, `efficient_live_test` |
| PID file `~/.finops/daemon.pid` | `~/.efficient/daemon.pid` |

Untouched: Mongo collection names (already generic), docker service/volume names (already neutral), dated historical markdown.

## Files in scope

- `efficient/` (renamed package): all modules referencing env vars, DB names, imports.
- `tests/`: all imports, `conftest.py` env plumbing, integration conftest/fixtures.
- `pyproject.toml`, `docker-compose.yml`, `Dockerfile` (if it names the package), `.env.example`, `.devcontainer/devcontainer.json`.
- `examples/claude-hooks/*.sh`, `examples/claude-hooks/README.md`, `examples/claude-hooks/CLAUDE.md`, `scripts/install-to-project.sh`.
- `README.md`, `efficient-mcp-README.md` (drop the "package is called finops" naming-split explanation; replace with a one-line historical note), `dashboard/` (any finops strings), `test-runner.sh`.

## Approach

Big-bang mechanical rename in one reviewed sweep: `git mv finops efficient`, scripted replace for imports/env vars/DB names/PID dir, manual pass over compose/hooks/scripts/docs, then refresh the editable install.

## Verification

1. `venv/bin/pip install -e ".[dev]"` (package dir changed).
2. `./test-runner.sh --integration` → 169 passed.
3. Live CLI smoke: `efficient start` / `status` / `stop` against a running MongoDB.
4. `grep -riI finops . --exclude-dir=venv --exclude-dir=.git --exclude-dir=docs --exclude-dir=.superpowers --exclude-dir=.pytest_cache` → empty (plus a check that remaining `docs/` hits are only in dated history).

## Risks

- Missed dynamic references (monkeypatch strings like `"finops.modules.agent_memory.ChatAnthropic"` in tests) — covered by the suite, which fails loudly on import paths.
- Stale editable install pointing at the old package dir — step 1 of verification.
- Local daemon state: existing `~/.finops` PID files and the old `finops` DB are orphaned, not migrated — acceptable per clean break.
