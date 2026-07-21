#!/usr/bin/env bash
# SessionStart hook: index the project's .py files into the efficient codebase graph.
# Mount-free — sends file contents to /codebase/index-file (no container bind mount needed).
# Copy to <your project>/.claude/hooks/ and chmod +x. Requires jq + the daemon running.
AUTH_ARGS=()
[ -n "${EFFICIENT_API_TOKEN:-}" ] && AUTH_ARGS=(-H "Authorization: Bearer $EFFICIENT_API_TOKEN")
[ -n "$CLAUDE_PROJECT_DIR" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
curl -s -m 2 http://localhost:7432/health >/dev/null 2>&1 || exit 0
find . -type f -name '*.py' \
  -not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/venv/*' -not -path '*/__pycache__/*' \
  | while read -r f; do
      rel="${f#./}"
      src=$(cat "$f")
      jq -n --arg r project --arg p "$rel" --arg s "$src" '{repo_id:$r, file_path:$p, source:$s}' \
        | curl -s -m 30 -X POST http://localhost:7432/codebase/index-file \
            "${AUTH_ARGS[@]}" \
            -H 'content-type: application/json' -d @- >/dev/null 2>&1 || true
    done
exit 0
