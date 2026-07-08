#!/usr/bin/env bash
# SessionStart hook: index the project into the efficient codebase graph.
# Requires the project mounted into the daemon container at /repo (see README).
# Copy to <your project>/.claude/hooks/ and chmod +x.
curl -s -m 300 -X POST http://localhost:7432/codebase/index \
  -H 'content-type: application/json' \
  -d '{"repo_id":"project","path":"/repo"}' >/dev/null 2>&1 || true
exit 0
