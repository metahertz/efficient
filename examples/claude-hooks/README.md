# efficient — Claude Code hooks (examples)

Deterministic ways to get Claude Code to actually use the `efficient` MCP tools
(CLAUDE.md *suggests*; hooks *enforce*). Copy these into the project you point
Claude Code at.

## Install
```bash
mkdir -p <your-project>/.claude/hooks
cp steer-large-reads.sh efficient-autoindex.sh <your-project>/.claude/hooks/
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

## Keeping the graph fresh as you edit
SessionStart indexing is a **snapshot at session start**. Files Claude edits
mid-session are NOT reflected until the next session/re-index, and the current
re-index is not incremental. For live freshness, add a PostToolUse re-index hook
once the single-file index endpoint lands (see the repo backlog:
"incremental codebase-graph updates").
