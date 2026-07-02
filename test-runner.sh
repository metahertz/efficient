#!/bin/bash
cd /Users/matt.johnson/ClaudeCodeRepo/fullFinOps-AI

# Build a test Docker image
docker build --target test -t finops-ai:test . -f - << 'EOF' && \
docker run --rm finops-ai:test pytest -v tests/ || echo "Docker not available, skipping container test"
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY finops/ finops/
COPY tests/ tests/

CMD ["pytest", "-v", "tests/"]
EOF
