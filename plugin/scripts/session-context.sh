#!/usr/bin/env bash
# SessionStart hook (sync, fast): inject a directive context note when the
# efficient codebase graph has data, so sessions actually use the MCP tools.
. "$(dirname "$0")/_repo_id.sh"
AUTH_ARGS=(-H "X-Efficient-Client: claude-code")
[ -n "${EFFICIENT_API_TOKEN:-}" ] && AUTH_ARGS+=(-H "Authorization: Bearer $EFFICIENT_API_TOKEN")
cat >/dev/null  # drain stdin
rid=$(compute_repo_id)
metrics=$(curl -s -m 2 "${AUTH_ARGS[@]}" http://localhost:7432/metrics 2>/dev/null) || exit 0
symbols=$(printf '%s' "$metrics" | jq -r '.store.codebase.symbols // 0' 2>/dev/null)
[ "${symbols:-0}" -gt 0 ] || exit 0
jq -n --arg r "$rid" '{
  hookSpecificOutput: {
    hookEventName: "SessionStart",
    additionalContext: ("The efficient MCP server is connected and its codebase graph is indexing this project under repo_id=\"\($r)\". ALWAYS prefer its tools over raw file access for code navigation: use lookup_symbol(query, repo_id=\"\($r)\") instead of reading whole files to find a definition, and find_references(repo_id=\"\($r)\", symbol) instead of grepping for callers/usages. Reading a large file or grepping a known identifier when these tools can answer wastes tokens and may be blocked by policy hooks.")
  }
}'
exit 0
