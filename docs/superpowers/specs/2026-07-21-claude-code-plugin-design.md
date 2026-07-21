# Claude Code Plugin — Design

**Date:** 2026-07-21
**Status:** Approved

## Goal

Package efficient's Claude Code integration (MCP server, four hooks, usage guidance) as an installable Claude Code plugin with a **monitor** that auto-starts the daemon — one-command install, ollama-style daemon residency. Replaces the manual `scripts/install-to-project.sh` + `examples/claude-hooks/` flow.

## Decisions (user-approved)

1. Monitor runs the stack via **docker compose** (daemon + MongoDB).
2. **Plugin replaces** the install script and example hooks — single source of truth; `examples/claude-hooks/` becomes a pointer README.
3. Distribution: this GitHub repo doubles as a **plugin marketplace** (`.claude-plugin/marketplace.json` at repo root, plugin under `plugin/`). Required regardless: project-scope plugins do not load monitors.
4. The plugin's own `docker-compose.yml` uses **git-URL build contexts** (`https://github.com/metahertz/efficient.git`) so the cached plugin copy is self-contained — no local checkout needed.

## Layout

```
.claude-plugin/marketplace.json          # {name: "efficient", plugins: [{name: "efficient", source: "./plugin", ...}]}
plugin/
├── .claude-plugin/plugin.json           # name, displayName, version 0.1.0, description, author
├── .mcp.json                            # "efficient": docker compose -f ${CLAUDE_PLUGIN_ROOT}/docker-compose.yml run --rm -T mcp
├── docker-compose.yml                   # mongodb + daemon + mcp; build contexts = git URL; hf_cache/mongo volumes; 127.0.0.1:7432; EFFICIENT_API_TOKEN passthrough
├── hooks/hooks.json                     # SessionStart → autoindex; PostToolUse(Write|Edit) → reindex; UserPromptSubmit → recall; PreToolUse(Read) → steer
├── scripts/
│   ├── efficient-autoindex.sh           # moved from examples/, unchanged logic
│   ├── reindex-on-edit.sh
│   ├── recall-memory.sh
│   ├── steer-large-reads.sh
│   └── daemon-monitor.sh                # compose up --wait daemon, then poll /health; print ONLY state transitions
├── monitors/monitors.json               # [{name: "efficient-daemon", command: "\"${CLAUDE_PLUGIN_ROOT}\"/scripts/daemon-monitor.sh", description}]
└── skills/efficient/SKILL.md            # when/how to use the 7 MCP tools (adapted from examples/claude-hooks/CLAUDE.md)
```

## Behavior notes

- Hook commands in hooks.json use `"${CLAUDE_PLUGIN_ROOT}"/scripts/<name>.sh`; scripts keep their current stdin/JSON contracts and `EFFICIENT_API_TOKEN` bearer support.
- Monitor stdout discipline: every line becomes a Claude notification, so `daemon-monitor.sh` prints one line on startup outcome and one per subsequent health transition (up→down, down→up); poll interval 30s. If Docker is unavailable it prints one line and exits 0.
- Monitor name uniqueness prevents duplicate daemons across plugin reloads; disabling the plugin mid-session does not stop a running monitor (documented Claude Code behavior).
- The daemon/MongoDB containers are `restart: unless-stopped` — sessions ending does not tear down the stack (residency is the point); `docker compose -f <plugin>/docker-compose.yml down` is the documented stop.
- The compose project name is fixed (`name: efficient`) so plugin-cache copies and repo checkout don't create parallel stacks.

## Removals / doc changes

- Delete `scripts/install-to-project.sh`.
- `examples/claude-hooks/`: delete scripts + settings.json + CLAUDE.md; README shrinks to "integration ships as a plugin now" with the two install commands and a manual-setup pointer at the plugin's files.
- Root README + `efficient-mcp-README.md`: install section becomes `/plugin marketplace add metahertz/efficient` + `/plugin install efficient@efficient`; keep the manual `claude mcp add` path documented for non-plugin users.

## Testing

- `tests/plugin/test_manifests.py` (non-integration): all four JSON files parse; hooks.json references only scripts that exist and are executable; monitors.json fields present; compose file parses via `docker compose -f ... config -q` when docker present else skipped.
- `tests/plugin/test_hook_scripts.py` (integration): drive the three daemon-calling hook scripts by stdin against `live_daemon` (as planned Layer B-style) — moved/adapted paths.
- Manual smoke (documented in plan, not CI): `/plugin marketplace add ./`, install, `/reload-plugins`, verify monitor starts stack and MCP tools list.

## Risks

- Git-URL build context requires network on first `up`; documented. Build takes minutes (torch); monitor's `--wait` uses a generous timeout and reports progress once.
- `docker compose run` for MCP joins the plugin compose project network; daemon must be up first — monitor handles this ordering.
