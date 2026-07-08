# efficient — Claude Code integration (CLAUDE.md + hooks)

Get Claude Code to actually use the `efficient` MCP tools. `CLAUDE.md` *suggests*
(the model decides); hooks *enforce/automate* (deterministic). Together they
cover all current capabilities. Copy these into the project you point Claude
Code at.

**The 7 MCP tools:** `optimize_context`, `index_codebase`, `lookup_symbol`,
`find_references`, `reindex_file`, `retrieve_memory`, `store_memory`.

## Install
```bash
# CLAUDE.md -> project root (merge if you already have one)
cp CLAUDE.md <your-project>/CLAUDE.md

# hooks
mkdir -p <your-project>/.claude/hooks
cp steer-large-reads.sh efficient-autoindex.sh reindex-on-edit.sh recall-memory.sh \
   <your-project>/.claude/hooks/
chmod +x <your-project>/.claude/hooks/*.sh
# merge settings.json into <your-project>/.claude/settings.json
```
**Prereqs:** `jq` on your host; the daemon running (`docker compose up -d daemon`).
All hooks use `repo_id`/`agent_id` = `"project"` — keep it consistent with CLAUDE.md.

## CLAUDE.md
Tells Claude *when* to use each of the 7 tools (targeted symbol lookup instead of
whole-file reads, `find_references` for impact analysis, `store_memory` for
durable decisions, `optimize_context` for large context, etc.). This is the
primary lever — the hooks below make the high-value paths automatic/enforced.

## Hook A — PreToolUse `steer-large-reads.sh`
Denies `Read` of code files > 400 lines and tells Claude to use `lookup_symbol` /
`find_references` instead. Self-contained. Switch `permissionDecision` from
`"deny"` to `"ask"` for a human prompt instead of hard enforcement.

## Hook B — SessionStart `efficient-autoindex.sh`
Indexes the repo into the codebase graph on new sessions (async, non-blocking).
**Requires the project mounted into the daemon container** — add to the `daemon`
(and `mcp`) service in `docker-compose.yml`:
```yaml
    volumes:
      - /ABS/PATH/TO/YOUR/PROJECT:/repo:ro
```
then `docker compose up -d daemon`. Does a whole-repo clean replace, so
deleted/renamed files drop out of the graph.

## Hook C — PostToolUse `reindex-on-edit.sh`
Keeps the graph fresh mid-session: after Claude edits a `.py` file
(`Edit|Write|MultiEdit`), re-indexes just that file (clean per-file replace).
**No mount needed** — sends the file contents in the request body. Extend the
`case` in the script (e.g. add `*.js) ;;`) as more language extractors land.

## Hook D — UserPromptSubmit `recall-memory.sh`
Auto-recall: on each prompt, queries `retrieve_memory` and injects relevant
semantic/episodic memory as context, so prior facts/decisions surface without the
model having to ask. Pairs with `store_memory` (which the model calls per
CLAUDE.md) — memory only surfaces if it was stored.

## Verify
- **Read steer:** ask Claude to read a 500-line file → denied, pivots to `lookup_symbol`.
- **Auto-index / reindex:** `curl -s localhost:7432/metrics` → `codebase_graph` events climb; edit a file and confirm `find_references` reflects it.
- **Memory:** `store_memory(...)` a fact, start a new prompt mentioning it → the recall hook injects it.
