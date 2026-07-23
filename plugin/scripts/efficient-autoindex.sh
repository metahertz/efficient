#!/usr/bin/env bash
# SessionStart hook (async): index the project's source files (py/ts/tsx/js/
# jsx/mjs/cjs) into the efficient codebase graph. Mount-free — sends file
# contents to /codebase/index-file. Requires jq. Runs async, so it waits for
# the daemon rather than bailing: on a fresh plugin install the monitor may
# still be building images when the session starts.
. "$(dirname "$0")/_repo_id.sh"
AUTH_ARGS=(-H "X-Efficient-Client: claude-code")
[ -n "${EFFICIENT_API_TOKEN:-}" ] && AUTH_ARGS+=(-H "Authorization: Bearer $EFFICIENT_API_TOKEN")
[ -n "$CLAUDE_PROJECT_DIR" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0
rid=$(compute_repo_id)

# Wait up to 15 minutes for the daemon (first install builds docker images).
up=""
for _ in $(seq 1 90); do
  if curl -sf -m 2 http://localhost:7432/health >/dev/null 2>&1; then up=1; break; fi
  sleep 10
done
[ -n "$up" ] || exit 0

# Prime the embedding model with one long-timeout call: the daemon's first
# embed downloads the model, so the first index-file can take minutes. Doing
# it once here keeps the per-file timeout below sane.
jq -n --arg r "$rid" '{repo_id:$r, file_path:"__efficient_warmup__.py", source:"pass"}' \
  | curl -s -m 900 -X POST http://localhost:7432/codebase/index-file \
      "${AUTH_ARGS[@]}" \
      -H 'content-type: application/json' -d @- >/dev/null 2>&1 || true

find . -type f \
  \( -name '*.py' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' \
     -o -name '*.jsx' -o -name '*.mjs' -o -name '*.cjs' \) \
  -not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/venv/*' \
  -not -path '*/dist/*' -not -path '*/build/*' -not -path '*/__pycache__/*' \
  | while read -r f; do
      rel="${f#./}"
      src=$(cat "$f")
      jq -n --arg r "$rid" --arg p "$rel" --arg s "$src" '{repo_id:$r, file_path:$p, source:$s}' \
        | curl -s -m 60 -X POST http://localhost:7432/codebase/index-file \
            "${AUTH_ARGS[@]}" \
            -H 'content-type: application/json' -d @- >/dev/null 2>&1 || true
    done
exit 0
