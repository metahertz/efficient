#!/usr/bin/env bash
# SessionStart hook (async): sync Claude Code's native auto-memory files
# (~/.claude/projects/<project>/memory/*.md) into efficient's memory_files
# store under /memories/native/, making them vector-searchable via
# retrieve_memory and the recall hook. Unchanged files are no-ops daemon-side.
AUTH_ARGS=(-H "X-Efficient-Client: claude-code")
[ -n "${EFFICIENT_API_TOKEN:-}" ] && AUTH_ARGS+=(-H "Authorization: Bearer $EFFICIENT_API_TOKEN")
[ -n "$CLAUDE_PROJECT_DIR" ] || exit 0

slug=$(printf '%s' "$CLAUDE_PROJECT_DIR" | tr '/.' '--')
memdir="$HOME/.claude/projects/$slug/memory"
[ -d "$memdir" ] || exit 0

# wait briefly for the daemon (monitor may still be starting the stack)
up=""
for _ in $(seq 1 90); do
  if curl -sf -m 2 http://localhost:7432/health >/dev/null 2>&1; then up=1; break; fi
  sleep 10
done
[ -n "$up" ] || exit 0

find "$memdir" -maxdepth 1 -type f -name '*.md' | while read -r f; do
  name=$(basename "$f")
  content=$(head -c 100000 "$f")
  [ -n "$content" ] || continue
  jq -n --arg a project --arg p "/memories/native/$name" --arg t "$content" \
    '{agent_id:$a, command:"create", path:$p, file_text:$t}' \
    | curl -s -m 120 -X POST http://localhost:7432/memory/tool \
        "${AUTH_ARGS[@]}" -H 'content-type: application/json' -d @- >/dev/null 2>&1 || true
done
exit 0
