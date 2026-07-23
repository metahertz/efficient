import os
import secrets

from fastapi import HTTPException, Request

# Paths a browser or liveness probe hits without credentials.
# /v1 carries the caller's own Anthropic Authorization header (gateway
# passthrough) — the daemon bearer check must not intercept it.
_EXEMPT_PREFIXES = ("/health", "/metrics", "/dashboard", "/status", "/v1")


async def require_token(request: Request) -> None:
    expected = os.getenv("EFFICIENT_API_TOKEN", "")
    if not expected:
        return
    path = request.url.path
    if any(path == p or path.startswith(p + "/") for p in _EXEMPT_PREFIXES):
        return
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token (EFFICIENT_API_TOKEN)")
    provided = header[7:].strip()  # Remove "Bearer " prefix (7 chars)
    if not (provided and secrets.compare_digest(provided, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token (EFFICIENT_API_TOKEN)")
