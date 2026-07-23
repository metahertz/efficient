#!/usr/bin/env bash
# PostToolUse hook: re-index a file into the efficient codebase graph after Claude edits it.
. "$(dirname "$0")/_repo_id.sh"
AUTH_ARGS=(-H "X-Efficient-Client: claude-code")
[ -n "${EFFICIENT_API_TOKEN:-}" ] && AUTH_ARGS+=(-H "Authorization: Bearer $EFFICIENT_API_TOKEN")
input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[ -z "$file" ] && exit 0
case "$file" in *.py|*.ts|*.tsx|*.js|*.jsx|*.mjs|*.cjs) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0
[ -n "$CLAUDE_PROJECT_DIR" ] || exit 0
rel="${file#"$CLAUDE_PROJECT_DIR"/}"
src=$(cat "$file")
rid=$(compute_repo_id)
jq -n --arg r "$rid" --arg f "$rel" --arg s "$src" \
  '{repo_id:$r, file_path:$f, source:$s}' \
  | curl -s -m 60 -X POST http://localhost:7432/codebase/index-file \
      "${AUTH_ARGS[@]}" \
      -H 'content-type: application/json' -d @- >/dev/null 2>&1 || true
exit 0
