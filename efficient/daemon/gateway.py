"""Gateway mode v1 (read-mostly): forward Anthropic API traffic verbatim,
stream responses unbuffered, and record usage measurements. Never mutates
requests or responses — cache serving / compression are informed by these
measurements and land in a later version.
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from efficient.db.client import get_async_db
from efficient.db.collections import GATEWAY_LOG

router = APIRouter()

_HOP_HEADERS = {
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
    "accept-encoding",
}

_client: httpx.AsyncClient | None = None
_first_request_seen = False


def _upstream() -> str:
    return os.getenv("EFFICIENT_GATEWAY_UPSTREAM", "https://api.anthropic.com")


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None),
        )
    return _client


def _parse_usage_from_sse(data_lines: list[bytes]) -> dict:
    usage = {}
    for raw in data_lines:
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            continue
        if payload.get("type") == "message_start":
            u = payload.get("message", {}).get("usage", {}) or {}
            usage["input_tokens"] = u.get("input_tokens", 0)
            usage["cache_read_input_tokens"] = u.get("cache_read_input_tokens", 0)
            usage["cache_creation_input_tokens"] = u.get("cache_creation_input_tokens", 0)
        elif payload.get("type") == "message_delta":
            u = payload.get("usage", {}) or {}
            usage["output_tokens"] = u.get("output_tokens", 0)
    return usage


def _parse_usage_from_json(body: bytes) -> dict:
    try:
        u = json.loads(body).get("usage", {}) or {}
    except (ValueError, UnicodeDecodeError):
        return {}
    return {
        "input_tokens": u.get("input_tokens", 0),
        "output_tokens": u.get("output_tokens", 0),
        "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
    }


async def _record(doc: dict) -> None:
    global _first_request_seen
    try:
        from efficient import activity
        db = get_async_db()
        await db[GATEWAY_LOG].insert_one(doc)
        if not _first_request_seen:
            _first_request_seen = True
            activity.emit("gateway: proxying Claude Code model traffic", notify=True)
        activity.emit(
            f"gateway: {doc.get('model') or doc['path']} "
            f"in={doc['input_tokens']} out={doc['output_tokens']} "
            f"cache_read={doc['cache_read_input_tokens']}"
        )
    except Exception:
        # measurement must never break the wire path
        pass


@router.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def gateway(request: Request, path: str):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
    model = ""
    is_stream = False
    if body:
        try:
            parsed = json.loads(body)
            model = parsed.get("model", "")
            is_stream = bool(parsed.get("stream"))
        except (ValueError, UnicodeDecodeError):
            pass

    t0 = time.perf_counter()
    client = _get_client()
    upstream_request = client.build_request(
        request.method,
        f"{_upstream()}/v1/{path}",
        content=body or None,
        headers=headers,
        params=dict(request.query_params),
    )
    upstream = await client.send(upstream_request, stream=True)
    response_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _HOP_HEADERS
    }

    base_doc = {
        "created_at": datetime.now(timezone.utc),
        "path": path,
        "model": model,
        "stream": is_stream,
        "status": upstream.status_code,
        "request_bytes": len(body),
        "body_hash": hashlib.sha256(body).hexdigest() if body else "",
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }

    async def relay():
        # tee usage incrementally: SSE data-lines carrying "usage" (message_start
        # arrives first, message_delta last — a tail buffer would lose the former
        # on long streams); non-stream bodies are buffered whole, bounded.
        usage_lines: list[bytes] = []
        pending = b""
        json_body = b""
        try:
            async for chunk in upstream.aiter_raw():
                if path.startswith("messages"):
                    if is_stream:
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if (line.startswith(b"data:") and b'"usage"' in line
                                    and len(usage_lines) < 16):
                                usage_lines.append(line[5:].strip())
                    elif len(json_body) < 1_048_576:
                        json_body += chunk
                yield chunk
        finally:
            await upstream.aclose()
            if path.startswith("messages"):
                if is_stream:
                    usage = _parse_usage_from_sse(usage_lines)
                else:
                    usage = _parse_usage_from_json(json_body)
                doc = {**base_doc, **usage,
                       "latency_ms": (time.perf_counter() - t0) * 1000}
                await _record(doc)

    return StreamingResponse(
        relay(), status_code=upstream.status_code, headers=response_headers,
    )
