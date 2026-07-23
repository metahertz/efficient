#!/usr/bin/env bash
# PreToolUse hook (Grep): when the pattern is a plain identifier that the
# efficient codebase graph already knows, deny the grep and hand Claude the
# graph's answer instead. Requires jq; silently allows if the daemon is down.
. "$(dirname "$0")/_repo_id.sh"
AUTH_ARGS=(-H "X-Efficient-Client: claude-code")
[ -n "${EFFICIENT_API_TOKEN:-}" ] && AUTH_ARGS+=(-H "Authorization: Bearer $EFFICIENT_API_TOKEN")
input=$(cat)
pattern=$(printf '%s' "$input" | jq -r '.tool_input.pattern // empty')
[ -z "$pattern" ] && exit 0
# identifier-shaped only: leave regexes, paths, and phrases to native grep
printf '%s' "$pattern" | grep -Eq '^[A-Za-z_][A-Za-z0-9_]{2,}$' || exit 0
rid=$(compute_repo_id)

refs=$(jq -n --arg r "$rid" --arg s "$pattern" '{repo_id:$r, symbol:$s}' \
  | curl -s -m 5 -X POST http://localhost:7432/codebase/references \
      "${AUTH_ARGS[@]}" -H 'content-type: application/json' -d @- 2>/dev/null) || exit 0
n=$(printf '%s' "$refs" | jq '(.callers // [] | length) + (.callees // [] | length)' 2>/dev/null)
[ "${n:-0}" -gt 0 ] || exit 0

jq -n --arg s "$pattern" --arg n "$n" --arg r "$rid" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("The efficient codebase graph already knows `\($s)` (\($n) caller/callee edges). Use the MCP tool find_references(repo_id=\"\($r)\", symbol=\"\($s)\") for callers/callees, or lookup_symbol(query=\"\($s)\", repo_id=\"\($r)\") for its definition, instead of grep.")
  }
}'
exit 0
