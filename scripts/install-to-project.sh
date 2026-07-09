#!/usr/bin/env bash
# Install the efficient Claude Code integration (CLAUDE.md + hooks + MCP registration)
# into a target project directory — no manual copying/editing needed.
#
#   scripts/install-to-project.sh /path/to/your/project
#
# Idempotent. Requires jq. If the daemon is running it also does an initial index.
set -euo pipefail

EFFICIENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$EFFICIENT_DIR/examples/claude-hooks"

TARGET="${1:-}"
[ -n "$TARGET" ] || { echo "usage: $0 <project-dir>"; exit 1; }
[ -d "$TARGET" ] || { echo "error: '$TARGET' is not a directory"; exit 1; }
TARGET="$(cd "$TARGET" && pwd)"
command -v jq >/dev/null || { echo "error: jq is required"; exit 1; }

echo "Installing efficient integration into: $TARGET"

# 1. CLAUDE.md (append our section if a CLAUDE.md already exists)
if [ -f "$TARGET/CLAUDE.md" ]; then
  if grep -q "efficient-integration" "$TARGET/CLAUDE.md"; then
    echo "  CLAUDE.md: efficient section already present (skip)"
  else
    { echo; cat "$SRC/CLAUDE.md"; } >> "$TARGET/CLAUDE.md"
    echo "  CLAUDE.md: appended efficient section"
  fi
else
  cp "$SRC/CLAUDE.md" "$TARGET/CLAUDE.md"
  echo "  CLAUDE.md: created"
fi

# 2. hooks
mkdir -p "$TARGET/.claude/hooks"
cp "$SRC"/*.sh "$TARGET/.claude/hooks/"
chmod +x "$TARGET/.claude/hooks/"*.sh
echo "  .claude/hooks/: installed $(ls "$SRC"/*.sh | wc -l | tr -d ' ') scripts"

# 3. .claude/settings.json — merge hook event-arrays (append), idempotent
SETTINGS="$TARGET/.claude/settings.json"
[ -f "$SETTINGS" ] || echo '{}' > "$SETTINGS"
if grep -q "efficient-autoindex.sh" "$SETTINGS"; then
  echo "  settings.json: efficient hooks already present (skip)"
else
  jq -s '
    def merge_hooks($a; $b):
      reduce (($a + $b) | keys_unsorted[]) as $k ({}; .[$k] = (($a[$k] // []) + ($b[$k] // [])));
    .[0] * { hooks: merge_hooks((.[0].hooks // {}); (.[1].hooks // {})) }
  ' "$SETTINGS" "$SRC/settings.json" > "$SETTINGS.tmp" && mv "$SETTINGS.tmp" "$SETTINGS"
  echo "  settings.json: merged efficient hooks"
fi

# 4. .mcp.json — register the efficient MCP server (idempotent overwrite of that key)
MCPJSON="$TARGET/.mcp.json"
[ -f "$MCPJSON" ] || echo '{}' > "$MCPJSON"
jq --arg dir "$EFFICIENT_DIR" \
  '.mcpServers.efficient = {command:"docker", args:["compose","-f",($dir+"/docker-compose.yml"),"run","--rm","-T","mcp"]}' \
  "$MCPJSON" > "$MCPJSON.tmp" && mv "$MCPJSON.tmp" "$MCPJSON"
echo "  .mcp.json: registered efficient MCP server"

# 5. initial index (mount-free, content-based) if the daemon is up
if curl -s -m 2 http://localhost:7432/health >/dev/null 2>&1; then
  n=0
  while IFS= read -r f; do
    rel="${f#"$TARGET"/}"; src=$(cat "$f" 2>/dev/null || true)
    jq -n --arg r project --arg p "$rel" --arg s "$src" '{repo_id:$r, file_path:$p, source:$s}' \
      | curl -s -m 30 -X POST http://localhost:7432/codebase/index-file \
          -H 'content-type: application/json' -d @- >/dev/null 2>&1 || true
    n=$((n+1))
  done < <(find "$TARGET" -type f -name '*.py' \
             -not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/venv/*' -not -path '*/__pycache__/*')
  echo "  initial index: sent $n .py files to the codebase graph"
else
  echo "  daemon not running — start it and it will index on the first Claude Code session:"
  echo "      (cd $EFFICIENT_DIR && docker compose up -d daemon)"
fi

echo "Done. Ensure the daemon is running, then open Claude Code in $TARGET."
