# fullFinOps-AI — Claude Code MCP Server

The finops MCP server exposes the token-saving daemon to Claude Code as native tools:
`optimize_context`, `index_codebase`, `lookup_symbol`, `retrieve_memory`, `store_memory`.

It is a Python (FastMCP) server that runs inside the project's Docker container over
stdio and talks HTTP to the finops daemon. No Node toolchain is required.

## Prerequisites

1. Build the images (installs CPU-only torch + all deps):

       docker compose build daemon dev

2. Start the daemon and MongoDB (the daemon must be running before Claude Code
   launches the MCP server — the server reaches it at FINOPS_DAEMON_URL):

       docker compose up -d daemon

   Confirm it is healthy:

       curl -s http://localhost:7432/health

## Register with Claude Code

The MCP server is launched on demand by Claude Code via
`docker compose run --rm -T mcp`. The `-T` disables pseudo-TTY allocation, which is
required for clean stdio JSON-RPC framing.

### Option A — `claude mcp add`

    claude mcp add finops -- docker compose -f /Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/docker-compose.yml run --rm -T mcp

### Option B — `~/.claude.json` (or the project's `.mcp.json`) snippet

    {
      "mcpServers": {
        "finops": {
          "command": "docker",
          "args": [
            "compose",
            "-f",
            "/Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI/docker-compose.yml",
            "run",
            "--rm",
            "-T",
            "mcp"
          ]
        }
      }
    }

Use the absolute path to this repo's `docker-compose.yml`. Because the MCP server
reads `FINOPS_DAEMON_URL` (baked into the `mcp` compose service as
`http://daemon:7432`) and joins the compose network, it reaches the running daemon
container directly.

## Notes

- The daemon must be up first (`docker compose up -d daemon`); the MCP process is
  short-lived and exits when Claude Code closes the connection.
- stdout is the MCP protocol channel; the server logs only to stderr.
- Index a repo before using `lookup_symbol`: call `index_codebase(repo_id, path)`
  where `path` is a directory inside the mounted `/workspace`.
