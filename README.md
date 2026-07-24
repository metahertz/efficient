# efficient

A local, MongoDB-backed daemon that saves AI tokens: codebase graph indexing,
semantic caching, agent memory, context compression, and hybrid retrieval,
combined into one pipeline and exposed over HTTP to every major AI framework —
including a native MCP integration for Claude Code. The Python package, CLI,
and env-var prefix (`EFFICIENT_*`) all share the name `efficient`.

> Historical note: before 2026-07-21 the package and env vars were named
> `finops` (project "fullFinOps-AI"); dated documents under `docs/superpowers/`
> retain that name.

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

The integration ships as a Claude Code plugin (this repo doubles as its
marketplace). In Claude Code, run:

```
/plugin marketplace add metahertz/efficient
/plugin install efficient@efficient
```

The plugin bundles the MCP server (7 tools: `optimize_context`,
`index_codebase`, `lookup_symbol`, `find_references`, `retrieve_memory`,
`store_memory`, `reindex_file`), four hooks (auto-index on session start,
reindex on edit, large-read steering, memory recall), a usage skill, and a
background monitor that auto-starts the daemon stack via
`plugin/docker-compose.yml` (first install builds images — allow a few
minutes). To require auth, export `EFFICIENT_API_TOKEN` before starting
Claude Code — the daemon, MCP server, and hooks all pick it up.

For manual (non-plugin) MCP registration, see `efficient-mcp-README.md`; the
hook scripts live in `plugin/scripts/` if you want to wire them yourself.

### Gateway mode (measure Claude Code's model traffic)

Launch Claude Code through the daemon and every model call is proxied
verbatim (streaming preserved, nothing mutated) while efficient measures
usage — tokens, Anthropic prompt-cache utilization, duplicate requests:

```bash
efficient claude            # sets ANTHROPIC_BASE_URL and execs claude
```

The dashboard's Gateway panel shows the running totals. Upstream defaults to
`https://api.anthropic.com`; override with `EFFICIENT_GATEWAY_UPSTREAM`. v1 is
read-only measurement — cache serving and compression decisions will be built
on what these numbers show.

### Memory-tool backend (Agent SDK)

efficient can be the storage backend for the Anthropic API memory tool
(`memory_20250818`): durable, agent-scoped, and every write is embedded so
memory becomes vector-searchable (surfaced via `/memory/retrieve`, the
`retrieve_memory` MCP tool, and the recall hook):

```python
from anthropic import Anthropic
from efficient.sdk import EfficientMemoryTool

client = Anthropic()
runner = client.beta.messages.tool_runner(
    model="claude-sonnet-5", max_tokens=1024,
    tools=[EfficientMemoryTool(agent_id="support-bot")],
    messages=[{"role": "user", "content": "Remember: we deploy on Fridays."}],
)
```

The daemon must be running; `EFFICIENT_DAEMON_URL`/`EFFICIENT_API_TOKEN` are
honored.

### OpenAI-compatible shim (aider, Continue, Cline, Open WebUI, …)

Point any OpenAI-base-URL client at the daemon to get semantic caching:

```
OPENAI_BASE_URL=http://localhost:7432/v1
```

`POST /v1/chat/completions` runs requests through the semantic cache — a hit is
served locally with no upstream call; a miss is forwarded verbatim to an
OpenAI-compatible upstream (`EFFICIENT_OPENAI_UPSTREAM`, default
`https://api.openai.com/v1`) using the caller's bearer key, and the response is
cached. v1 is cache-only (no compression) and buffers streaming into SSE frames.

### Retrieval corpus (RAG)

Seed a hybrid (BM25 + vector) retrieval corpus, then query it via
`optimize_context(corpus_id=…)` or the `add_corpus` MCP tool:

```
POST /corpus/add-chunks {"corpus_id": "docs", "chunks": [{"text": "...", "source_file": "a.md"}]}
```

Or auto-ingest whole directories — drop notes/docs into watched folders and
efficient keeps a corpus in sync. Configure `~/.efficient/watch.json`:

```json
{"watches": [{"path": "~/notes", "corpus_id": "notes"}]}
```

then `efficient watch --once` (one pass) or `efficient watch` (continuous,
re-ingests on change, removes chunks on delete). Text formats
(.md/.txt/.rst/.mdx) by default.

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
