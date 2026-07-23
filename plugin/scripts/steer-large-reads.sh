#!/usr/bin/env bash
# PreToolUse hook: steer large code-file reads to the efficient MCP tools.
# Copy to <your project>/.claude/hooks/ and chmod +x. Requires `jq`.
. "$(dirname "$0")/_repo_id.sh"
input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[ -z "$file" ] && exit 0
case "$file" in
  *.py|*.ts|*.tsx|*.js|*.jsx|*.go|*.rs|*.java|*.rb|*.c|*.h|*.cpp) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0
lines=$(wc -l < "$file" 2>/dev/null | tr -d ' ')
THRESHOLD="${EFFICIENT_READ_STEER_LINES:-150}"
if [ "${lines:-0}" -gt "$THRESHOLD" ]; then
  rid=$(compute_repo_id)
  jq -n --arg f "$file" --arg n "$lines" --arg r "$rid" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: ("\($f) is \($n) lines. Use the efficient MCP tool `lookup_symbol(query, repo_id=\"\($r)\")` to fetch only the relevant symbol, or `find_references` to trace usage, instead of reading the whole file.")
    }
  }'
fi
exit 0
