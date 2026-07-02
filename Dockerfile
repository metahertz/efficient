FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only pyproject.toml first so dep layer is cached independently of source changes
COPY pyproject.toml .
# Minimal stub so pip install -e . resolves without full source
RUN mkdir -p finops && touch finops/__init__.py
RUN pip install --no-cache-dir -e ".[dev]"

# Full source copy (overrides stub; used for daemon service builds)
COPY . .
RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 7432
CMD uvicorn finops.daemon.app:app --host 0.0.0.0 --port ${FINOPS_PORT:-7432}
