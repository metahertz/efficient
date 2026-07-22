#!/usr/bin/env bash
# Plugin monitor: keep the efficient daemon running and report state changes.
# Every stdout line becomes a Claude Code notification — print transitions only.
# CLAUDE_PLUGIN_ROOT is substituted in the monitors.json command string but is
# NOT exported to the monitor process env — derive the root from this script's
# own location and use the env var only as a fallback.
PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$PLUGIN_ROOT/docker-compose.yml" ] || PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT}"
COMPOSE_FILE="${PLUGIN_ROOT}/docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "efficient: cannot locate plugin docker-compose.yml (looked in $PLUGIN_ROOT)"
  exit 0
fi
HEALTH_URL="http://localhost:7432/health"

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

if ! command -v docker >/dev/null 2>&1; then
  echo "efficient: docker not found — daemon not started (install Docker or run the daemon manually)"
  exit 0
fi

# Monitors run in the session working directory, so $PWD is the project.
index_project() {
  [ -x "$PLUGIN_ROOT/scripts/efficient-autoindex.sh" ] || return 0
  CLAUDE_PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$PWD}" \
    "$PLUGIN_ROOT/scripts/efficient-autoindex.sh" >/dev/null 2>&1 &
}

if curl -sf -m 2 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "efficient daemon running at localhost:7432"
else
  echo "efficient: starting daemon stack (first run builds images and downloads models — this can take minutes)"
  if compose up -d --wait daemon >/dev/null 2>&1 && curl -sf -m 5 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "efficient daemon up at localhost:7432 — indexing project in background"
    index_project
  else
    echo "efficient: daemon failed to start — try: docker compose -f \"$COMPOSE_FILE\" up daemon"
    exit 0
  fi
fi

state=up
while sleep 30; do
  if curl -sf -m 5 "$HEALTH_URL" >/dev/null 2>&1; then new=up; else new=down; fi
  if [ "$new" != "$state" ]; then
    if [ "$new" = "down" ]; then
      echo "efficient daemon DOWN (localhost:7432 unreachable)"
    else
      echo "efficient daemon recovered — re-indexing project in background"
      index_project
    fi
    state=$new
  fi
done
