#!/usr/bin/env bash
# Plugin monitor: keep the efficient daemon running and report state changes.
# Every stdout line becomes a Claude Code notification — print transitions only.
COMPOSE_FILE="${CLAUDE_PLUGIN_ROOT}/docker-compose.yml"
HEALTH_URL="http://localhost:7432/health"

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

if ! command -v docker >/dev/null 2>&1; then
  echo "efficient: docker not found — daemon not started (install Docker or run the daemon manually)"
  exit 0
fi

if curl -sf -m 2 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "efficient daemon running at localhost:7432"
else
  echo "efficient: starting daemon stack (first run builds images and downloads models — this can take minutes)"
  if compose up -d --wait daemon >/dev/null 2>&1 && curl -sf -m 5 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "efficient daemon up at localhost:7432"
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
      echo "efficient daemon recovered"
    fi
    state=$new
  fi
done
