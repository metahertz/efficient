import os
import secrets

from fastapi import HTTPException, Request

# Paths a browser or liveness probe hits without credentials.
_EXEMPT_PREFIXES = ("/health", "/metrics", "/dashboard")


async def require_token(request: Request) -> None:
    expected = os.getenv("FINOPS_API_TOKEN", "")
    if not expected:
        return
    path = request.url.path
    if any(path == p or path.startswith(p + "/") or path.startswith(p) for p in _EXEMPT_PREFIXES):
        return
    header = request.headers.get("authorization", "")
    provided = header.removeprefix("Bearer ").strip()
    if not (provided and secrets.compare_digest(provided, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token (FINOPS_API_TOKEN)")
