# efficient — Claude Code integration (CLAUDE.md + hooks)

Get Claude Code to use the `efficient` MCP tools automatically. `CLAUDE.md`
*suggests* (the model decides); hooks *enforce/automate* (deterministic).
Together they cover all current capabilities.

**The 7 MCP tools:** `optimize_context`, `index_codebase`, `lookup_symbol`,
`find_references`, `reindex_file`, `retrieve_memory`, `store_memory`.

## One-command install (recommended)
From the efficient repo, install everything into your project — no manual copying:
```bash
scripts/install-to-project.sh /path/to/your/project
```
It writes `CLAUDE.md`, `.claude/hooks/*` (chmod +x), merges the hooks into
`.claude/settings.json`, registers the MCP server in `.mcp.json`, and (if the
daemon is up) does an initial codebase index. Idempotent. Requires `jq`.
Then just ensure the daemon is running and open Claude Code:
```bash
docker compose up -d daemon      # in the efficient repo
```

## Manual install (if you prefer)
```bash
cp CLAUDE.md <your-project>/CLAUDE.md
mkdir -p <your-project>/.claude/hooks
cp *.sh <your-project>/.claude/hooks/ && chmod +x <your-project>/.claude/hooks/*.sh
# merge settings.json into <your-project>/.claude/settings.json
# add the efficient server to <your-project>/.mcp.json (see install script for the shape)
```
**Prereqs:** `jq` on your host; the daemon running. All hooks use
`repo_id`/`agent_id` = `"project"` — keep it consistent with CLAUDE.md.

## What gets wired up
- **CLAUDE.md** — tells Claude *when* to use each of the 7 tools (targeted
  `lookup_symbol` instead of whole-file reads, `find_references` for impact
  analysis, `store_memory` for durable decisions, `optimize_context` for large
  context).
- **Hook A — PreToolUse `steer-large-reads.sh`** — denies `Read` of code files
  > 400 lines and points Claude to `lookup_symbol`/`find_references`. Switch
  `permissionDecision` `"deny"` → `"ask"` for a human prompt instead.
- **Hook B — SessionStart `efficient-autoindex.sh`** — indexes the project's
  `.py` files on new sessions (async). **Mount-free** — sends file contents to
  `/codebase/index-file`, so no docker bind mount is needed. Whole-file deletes
  are reconciled on the next full index.
- **Hook C — PostToolUse `reindex-on-edit.sh`** — after Claude edits a `.py`
  file (`Edit|Write|MultiEdit`), re-indexes just that file (clean per-file
  replace) so `lookup_symbol`/`find_references` stay accurate mid-session.
- **Hook D — UserPromptSubmit `recall-memory.sh`** — queries `retrieve_memory`
  each prompt and injects relevant semantic/episodic memory as context. Pairs
  with `store_memory` (model-driven per CLAUDE.md).

Extend the `case`/`find` filters in the hook scripts (currently `.py`, matching
the codebase-graph extractor) as more language extractors land.

## Auth

If the daemon has `EFFICIENT_API_TOKEN` set, export the same variable in the
environment Claude Code runs in — the hooks pick it up and attach it as a
bearer `Authorization` header automatically on every daemon request. Also run
`efficient warmup` once before your first SessionStart so the autoindex hook
doesn't hit a slow cold index on the first prompt.

## Verify
- **Read steer:** ask Claude to read a 500-line file → denied, pivots to `lookup_symbol`.
- **Indexing:** `curl -s localhost:7432/metrics` → `codebase_graph` events climb; edit a file and confirm `find_references` reflects it.
- **Memory:** `store_memory(...)` a fact, then a new prompt mentioning it → the recall hook injects it.
