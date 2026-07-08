#!/usr/bin/env bash
# PostToolUse hook: re-index a file into the efficient codebase graph after Claude edits it.
input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[ -z "$file" ] && exit 0
case "$file" in *.py) ;; *) exit 0 ;; esac
[ -f "$file" ] || exit 0
rel="${file#"$CLAUDE_PROJECT_DIR"/}"
src=$(cat "$file")
jq -n --arg r "project" --arg f "$rel" --arg s "$src" \
  '{repo_id:$r, file_path:$f, source:$s}' \
  | curl -s -m 60 -X POST http://localhost:7432/codebase/index-file \
      -H 'content-type: application/json' -d @- >/dev/null 2>&1 || true
exit 0
