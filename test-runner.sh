#!/usr/bin/env bash
# Run the test suite against the dedicated mongodb-test instance (port 27018).
# Usage: ./test-runner.sh [--integration]
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d --wait mongodb-test
export EFFICIENT_TEST_MONGODB_URI="mongodb://localhost:27018/?directConnection=true"

if [ "${1:-}" = "--integration" ]; then
    exec venv/bin/python -m pytest -v
else
    exec venv/bin/python -m pytest -m "not integration" -q
fi
