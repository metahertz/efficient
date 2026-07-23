# Memory-Tool Backend — Design

**Date:** 2026-07-23
**Status:** Approved

## Goal

Make efficient a storage backend for the Anthropic API **memory tool** (`memory_20250818`): Agent SDK builders plug in `EfficientMemoryTool` and get durable, **vector-searchable**, multi-agent memory behind the official tool interface — the niche where our memory module beats Claude's file-based native memory.

## Components

1. **Daemon endpoint `POST /memory/tool`** — executes memory-tool commands server-side against a new `memory_files` collection. Body: `{agent_id="default", command, ...command fields}`. Commands and semantics mirror Anthropic's reference filesystem implementation:
   - `view {path, view_range?}` — directory (prefix) listing, or file content with `N: line` numbering
   - `create {path, file_text}` — create/overwrite; embeds content for vector recall (asyncio.to_thread)
   - `str_replace {path, old_str, new_str}` — old_str must occur exactly once; re-embeds
   - `insert {path, insert_line, insert_text}` — 0-based line insert; re-embeds
   - `delete {path}` — file, or prefix-delete for directories
   - `rename {old_path, new_path}` — file or prefix move
   - `clear_all` — wipe the agent's file store
   - Validation: paths must start with `/memories`, no `..` segments → error result.
   - Response: `{ok: true, result: "<text>"}` or `{ok: false, error: "<text>"}` (HTTP 200 either way — tool errors are content for the model, not transport failures).

2. **Storage**: `memory_files` docs `{agent_id, path, content, embedding, created_at, updated_at}`; unique index `(agent_id, path)`; vector index `memory_files_vector_index` (filter `agent_id`, content embedded truncated to 2000 chars).

3. **Vector recall**: `/memory/retrieve` response gains a `files` section — top-3 memory files by embedding similarity `{path, snippet}` — so both the recall hook and `retrieve_memory` MCP tool surface memory-tool content automatically.

4. **SDK helper `efficient.sdk.EfficientMemoryTool`** — subclass of `anthropic.lib.tools.BetaAbstractMemoryTool` (sync); constructor `(daemon_url=env EFFICIENT_DAEMON_URL or http://localhost:7432, agent_id="default")`; each command method POSTs to `/memory/tool` (bearer header via EFFICIENT_API_TOKEN) and returns the `result` string, or the `error` string (the model self-corrects on error text). Usage:

```python
from anthropic import Anthropic
from efficient.sdk import EfficientMemoryTool

client = Anthropic()
runner = client.beta.messages.tool_runner(
    model="claude-sonnet-5", max_tokens=1024,
    tools=[EfficientMemoryTool(agent_id="support-bot")],
    betas=["context-management-2025-06-27"],
    messages=[...],
)
```

## Non-goals

- No Claude Code-side change (its native auto-memory stays authoritative there); indexing native memory files is a separate roadmap item.
- No versioning/audit trail (Managed Agents territory).

## Testing

- `tests/daemon/test_memory_tool.py`: all 7 commands, overwrite, str_replace uniqueness error, path traversal rejection, view_range, retrieve `files` section (mocked embeddings).
- `tests/sdk/test_memory_tool.py`: command→payload mapping and result/error passthrough with the HTTP call stubbed; tool registers with the anthropic runner types (`to_dict()` shape).
