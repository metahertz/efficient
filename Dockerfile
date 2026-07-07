FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ENV PIP_DEFAULT_TIMEOUT=180 PIP_RETRIES=10

RUN --mount=type=cache,target=/root/.cache/pip pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
RUN mkdir -p finops && touch finops/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip pip install -e "."

COPY . .
RUN --mount=type=cache,target=/root/.cache/pip pip install -e "."

EXPOSE 7432
CMD uvicorn finops.daemon.app:app --host 0.0.0.0 --port ${FINOPS_PORT:-7432}

FROM base AS dev
RUN --mount=type=cache,target=/root/.cache/pip pip install -e ".[dev]"
