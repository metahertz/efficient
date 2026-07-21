# efficient

A local, MongoDB-backed daemon that saves AI tokens: codebase graph indexing,
semantic caching, agent memory, context compression, and hybrid retrieval,
combined into one pipeline and exposed over HTTP to every major AI framework —
including a native MCP integration for Claude Code. The Python package is
`efficient` and the installed CLI is `efficient` (the repo and CLI go by the
shorter, user-facing name; the package keeps its original internal name to
avoid a disruptive rename of the code).

## Quick start

```bash
docker compose build daemon dev
docker compose up -d daemon
curl -s http://localhost:7432/health
```

This builds the daemon image (CPU-only torch + all deps baked in), starts the
daemon and its MongoDB, and confirms the daemon is answering on `127.0.0.1:7432`.

## Local development

```bash
python3.12 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/python -m efficient.cli.main start   # or, once installed: efficient start
```

On first run, call `efficient warmup` once to pre-load the local embedding
model — without it, the first codebase index or memory lookup pays a slow
cold-start cost.

## Testing

```bash
./test-runner.sh                 # non-integration suite
./test-runner.sh --integration   # + integration suite
```

Both require Docker (the test suite runs against a `mongodb-test` container on
port 27018). The integration suite additionally downloads the `voyage-4-nano`
embedding model on its first run and caches it for subsequent runs.

## Security

- The daemon binds to `127.0.0.1` by default; set `EFFICIENT_HOST` to override.
- Set `EFFICIENT_API_TOKEN` to require a bearer token on daemon requests
  (`/health`, `/metrics`, and `/dashboard*` stay exempt so basic liveness
  checks and the dashboard keep working without a token). `docker compose`
  passes `EFFICIENT_API_TOKEN` through to both the `daemon` and `mcp` services.
- `index_codebase` only indexes paths under `modules.codebase_graph.repo_paths`
  (the daemon's Mongo-backed config) or `EFFICIENT_ALLOWED_INDEX_ROOTS` (a
  colon-separated list of additional allowed roots). Note that resetting the
  database (`docker compose down -v`) also resets this config back to its
  defaults.

## Claude Code integration

- `efficient-mcp-README.md` — registering the MCP server (7 tools:
  `optimize_context`, `index_codebase`, `lookup_symbol`, `find_references`,
  `retrieve_memory`, `store_memory`, `reindex_file`).
- `examples/claude-hooks/README.md` — CLAUDE.md + hooks that make Claude Code
  use those tools automatically (auto-index, auto-reindex-on-edit, read
  steering, memory recall).
- `scripts/install-to-project.sh` — one-command install of the above into any
  project.

`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` are optional and only needed for the
`/complete` endpoint and agent-memory fact extraction — see `.env.example`.
Everything else (embeddings, codebase indexing, caching) runs locally with no
API key.

## Design

The full design spec lives at
`docs/superpowers/specs/2026-06-30-fullefficient-ai-design.md`. Read it alongside
its own revision headers (Rev 2, Rev 3, …) near the top — later revisions
supersede conflicting statements in the original body, so the headers are the
more current source of truth where they disagree.
