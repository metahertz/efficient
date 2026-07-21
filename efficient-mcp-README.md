# efficient — Claude Code MCP Server

The efficient MCP server exposes the token-saving daemon to Claude Code as native tools:
`optimize_context`, `index_codebase`, `lookup_symbol`, `find_references`, `retrieve_memory`,
`store_memory`, `reindex_file`.

It is a Python (FastMCP) server that runs inside the project's Docker container over
stdio and talks HTTP to the efficient daemon. No Node toolchain is required.

## Prerequisites

1. Build the images (installs CPU-only torch + all deps):

       docker compose build daemon dev

2. Start the daemon and MongoDB (the daemon must be running before Claude Code
   launches the MCP server — the server reaches it at EFFICIENT_DAEMON_URL):

       docker compose up -d daemon

   Confirm it is healthy:

       curl -s http://localhost:7432/health

## Register with Claude Code

The MCP server is launched on demand by Claude Code via
`docker compose run --rm -T mcp`. The `-T` disables pseudo-TTY allocation, which is
required for clean stdio JSON-RPC framing.

### Option A — `claude mcp add`

    claude mcp add efficient -- docker compose -f <absolute path to this repo>/docker-compose.yml run --rm -T mcp

For example, from the repo root:

    claude mcp add efficient -- docker compose -f $(pwd)/docker-compose.yml run --rm -T mcp

### Option B — `~/.claude.json` (or the project's `.mcp.json`) snippet

    {
      "mcpServers": {
        "efficient": {
          "command": "docker",
          "args": [
            "compose",
            "-f",
            "<absolute path to this repo>/docker-compose.yml",
            "run",
            "--rm",
            "-T",
            "mcp"
          ]
        }
      }
    }

Use the absolute path to this repo's `docker-compose.yml`. Because the MCP server
reads `EFFICIENT_DAEMON_URL` (baked into the `mcp` compose service as
`http://daemon:7432`) and joins the compose network, it reaches the running daemon
container directly.

## Security

- The daemon binds to `127.0.0.1` by default; set `EFFICIENT_HOST` to override if you
  need it reachable from elsewhere.
- Set `EFFICIENT_API_TOKEN` to require a bearer token on daemon requests (`/health`,
  `/metrics`, and `/dashboard*` stay exempt). `docker compose` passes the same
  `EFFICIENT_API_TOKEN` through to both the `daemon` and `mcp` services, so the MCP
  server authenticates automatically once it's set.
- `index_codebase` only indexes paths that fall under `modules.codebase_graph.repo_paths`
  (as configured in the daemon's Mongo-backed config) or `EFFICIENT_ALLOWED_INDEX_ROOTS`
  (a colon-separated list of additional allowed roots).

## Notes

- The daemon must be up first (`docker compose up -d daemon`); the MCP process is
  short-lived and exits when Claude Code closes the connection.
- If the daemon isn't already running when Claude Code launches the MCP server,
  `docker compose run` will attempt to start it via `depends_on`; always prefer
  `docker compose up -d daemon` first to avoid cold-start latency mid-session.
- stdout is the MCP protocol channel; the server logs only to stderr.
- Index a repo before using `lookup_symbol`: call `index_codebase(repo_id, path)`
  where `path` is a directory inside the mounted `/workspace`.
- `/codebase/index-file` (what the `reindex_file` tool and the project's hooks use)
  is mount-free — it accepts file contents directly over HTTP, so it needs no
  allowlisted root and works without a bind mount.
