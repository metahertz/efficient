# efficient — Claude Code hooks (examples)

Deterministic ways to get Claude Code to actually use the `efficient` MCP tools
(CLAUDE.md *suggests*; hooks *enforce*). Copy these into the project you point
Claude Code at.

## Install
```bash
mkdir -p <your-project>/.claude/hooks
cp steer-large-reads.sh efficient-autoindex.sh reindex-on-edit.sh <your-project>/.claude/hooks/
chmod +x <your-project>/.claude/hooks/*.sh
# merge settings.json into <your-project>/.claude/settings.json
```
**Prereqs:** `jq` on your host; the daemon running (`docker compose up -d daemon`).

## Hook A — PreToolUse `steer-large-reads.sh`
Denies `Read` of code files > 400 lines and tells Claude to use
`lookup_symbol` / `find_references` instead. Self-contained (reads the host
file directly). Switch `permissionDecision` from `"deny"` to `"ask"` for a
human-in-the-loop prompt instead of hard enforcement. Tune the threshold/exts.

## Hook B — SessionStart `efficient-autoindex.sh`
Indexes the repo into the codebase graph on new sessions (async, non-blocking).
**Requires the project mounted into the daemon container** — the daemon indexes
from its own filesystem. In `docker-compose.yml` add to the `daemon` (and `mcp`)
service:
```yaml
    volumes:
      - /ABS/PATH/TO/YOUR/PROJECT:/repo:ro
```
then `docker compose up -d daemon`. Uses a fixed `repo_id: "project"` — use the
same `repo_id` in your CLAUDE.md so `lookup_symbol(query, repo_id="project")`
matches.

## Hook C — PostToolUse `reindex-on-edit.sh`
Keeps the graph fresh as you edit. SessionStart indexing is only a **snapshot at
session start**; this PostToolUse hook (matcher `Edit|Write|MultiEdit`) re-indexes
each edited `.py` file the moment Claude changes it, so `lookup_symbol` and
`find_references` stay accurate mid-session. It POSTs the file's current contents
to `/codebase/index-file`, which does a clean per-file replace (deleted symbols
are removed, moved symbols are not duplicated).

**No mount needed** — unlike the SessionStart autoindex, this hook sends the file
contents in the request body, so the daemon does not read from its own
filesystem. Uses the same fixed `repo_id: "project"` (paths are sent relative to
`$CLAUDE_PROJECT_DIR`). **Requires `jq` + the daemon running.** Currently handles
`.py` files only (matches the codebase-graph extractor); extend the `case` in the
script as more language extractors are added.
