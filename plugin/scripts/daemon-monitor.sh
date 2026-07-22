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

# Relay milestone events (notify=true) from the daemon's activity feed as
# notifications; baseline the sequence first so old events aren't replayed.
STATUS_URL="http://localhost:7432/status"
last_seq=$(curl -sf -m 5 "$STATUS_URL" 2>/dev/null | jq -r '.last_seq // 0' 2>/dev/null)
last_seq=${last_seq:-0}

state=up
while sleep 10; do
  if curl -sf -m 5 "$HEALTH_URL" >/dev/null 2>&1; then new=up; else new=down; fi
  if [ "$new" != "$state" ]; then
    if [ "$new" = "down" ]; then
      echo "efficient daemon DOWN (localhost:7432 unreachable)"
    else
      echo "efficient daemon recovered — re-indexing project in background"
      index_project
      last_seq=$(curl -sf -m 5 "$STATUS_URL" 2>/dev/null | jq -r '.last_seq // 0' 2>/dev/null)
      last_seq=${last_seq:-0}
    fi
    state=$new
  fi
  if [ "$new" = "up" ] && command -v jq >/dev/null 2>&1; then
    resp=$(curl -sf -m 5 "${STATUS_URL}?since=${last_seq}" 2>/dev/null) || continue
    printf '%s' "$resp" | jq -r '.events[] | select(.notify) | "efficient: " + .message' 2>/dev/null
    seq=$(printf '%s' "$resp" | jq -r '.last_seq // 0' 2>/dev/null)
    [ -n "$seq" ] && [ "$seq" != "0" ] && last_seq=$seq
  fi
done
