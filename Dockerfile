FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml .
RUN mkdir -p finops && touch finops/__init__.py
RUN pip install --no-cache-dir -e "."

COPY . .
RUN pip install --no-cache-dir -e "."

EXPOSE 7432
CMD uvicorn finops.daemon.app:app --host 0.0.0.0 --port ${FINOPS_PORT:-7432}

FROM base AS dev
RUN pip install --no-cache-dir -e ".[dev]"
