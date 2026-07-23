import asyncio
import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from httpx import ASGITransport

from efficient.db.collections import GATEWAY_LOG

SSE_BODY = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"usage":{"input_tokens":120,'
    '"cache_read_input_tokens":80,"cache_creation_input_tokens":10}}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","usage":{"output_tokens":42}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)


def _fake_upstream() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        if body.get("stream"):
            async def gen():
                for line in SSE_BODY.splitlines(keepends=True):
                    yield line.encode()
            return StreamingResponse(
                gen(), media_type="text/event-stream",
                headers={"x-seen-version": request.headers.get("anthropic-version", ""),
                         "x-seen-auth": request.headers.get("authorization", ""),
                         "x-seen-beta": request.headers.get("anthropic-beta", ""),
                         "x-seen-encoding": request.headers.get("accept-encoding", "")},
            )
        return JSONResponse({
            "id": "msg_1", "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 7, "output_tokens": 3},
        })

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "claude-fake"}]}

    return app


@pytest.fixture
async def client(efficient_db, monkeypatch):
    import efficient.daemon.gateway as gw
    fake = _fake_upstream()
    upstream_client = httpx.AsyncClient(
        transport=ASGITransport(app=fake), base_url="http://upstream",
        timeout=httpx.Timeout(10.0, read=None),
    )
    monkeypatch.setattr(gw, "_get_client", lambda: upstream_client)
    monkeypatch.setenv("EFFICIENT_GATEWAY_UPSTREAM", "http://upstream")
    from efficient.daemon.app import app
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    await upstream_client.aclose()


async def _wait_for_log(db, n, timeout=5.0):
    for _ in range(int(timeout * 10)):
        docs = [d async for d in db[GATEWAY_LOG].find({})]
        if len(docs) >= n:
            return docs
        await asyncio.sleep(0.1)
    return [d async for d in db[GATEWAY_LOG].find({})]


async def test_stream_passthrough_bytes_and_headers(client):
    r = await client.post("/v1/messages", json={"model": "claude-fake", "stream": True},
                          headers={"anthropic-version": "2023-06-01",
                                   "authorization": "Bearer sk-ant-xyz",
                                   "anthropic-beta": "context-1,tools-2"})
    assert r.status_code == 200
    assert r.text == SSE_BODY  # byte-for-byte, unbuffered semantics preserved
    assert r.headers["x-seen-version"] == "2023-06-01"
    assert r.headers["x-seen-auth"] == "Bearer sk-ant-xyz"
    assert r.headers["x-seen-beta"] == "context-1,tools-2"
    # the usage tee reads raw wire bytes — upstream must not compress
    assert r.headers["x-seen-encoding"] == "identity"


async def test_stream_usage_recorded(client, efficient_db):
    await client.post("/v1/messages", json={"model": "claude-fake", "stream": True})
    docs = await _wait_for_log(efficient_db, 1)
    assert len(docs) == 1
    d = docs[0]
    assert d["model"] == "claude-fake"
    assert d["input_tokens"] == 120
    assert d["output_tokens"] == 42
    assert d["cache_read_input_tokens"] == 80
    assert d["cache_creation_input_tokens"] == 10
    assert d["stream"] is True


async def test_non_stream_usage_recorded(client, efficient_db):
    r = await client.post("/v1/messages", json={"model": "claude-fake"})
    assert r.json()["content"][0]["text"] == "hello"
    docs = await _wait_for_log(efficient_db, 1)
    assert docs[0]["input_tokens"] == 7
    assert docs[0]["output_tokens"] == 3


async def test_duplicate_requests_counted_in_metrics(client, efficient_db):
    body = {"model": "claude-fake", "messages": [{"role": "user", "content": "same"}]}
    await client.post("/v1/messages", json=body)
    await client.post("/v1/messages", json=body)
    await _wait_for_log(efficient_db, 2)
    r = await client.get("/metrics")
    gw = r.json()["gateway"]
    assert gw["requests"] == 2
    assert gw["duplicate_requests"] == 1
    assert gw["input_tokens"] == 14


async def test_get_passthrough(client):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "claude-fake"


async def test_gateway_exempt_from_daemon_auth(client, monkeypatch):
    monkeypatch.setenv("EFFICIENT_API_TOKEN", "sekret")
    r = await client.post("/v1/messages", json={"model": "claude-fake"},
                          headers={"authorization": "Bearer sk-ant-not-our-token"})
    assert r.status_code == 200


async def test_guardian_fields_and_invalidation(client, efficient_db):
    from efficient.daemon import cache_guardian
    cache_guardian.reset()
    hdrs = {"x-claude-code-session-id": "sess-1"}
    body1 = {"model": "claude-fake", "tools": [{"name": "a"}],
             "messages": [{"role": "user", "content": "one"}]}
    await client.post("/v1/messages", json=body1, headers=hdrs)
    body2 = {**body1, "tools": [{"name": "a"}, {"name": "b"}],
             "messages": [{"role": "user", "content": "two"}]}
    await client.post("/v1/messages", json=body2, headers=hdrs)
    docs = await _wait_for_log(efficient_db, 2)
    docs.sort(key=lambda d: d["created_at"])
    assert docs[0]["session_id"] == "sess-1"
    assert docs[0]["invalidator"] is None
    assert docs[1]["invalidator"] == "tools_changed"
    assert "cache_read_ratio" in docs[1]

    r = await client.get("/metrics")
    gw = r.json()["gateway"]
    assert gw["invalidations"] == 1
    assert gw["sessions"] == 1
    assert 0 <= gw["cache_read_ratio"] <= 1


def test_cli_claude_sets_base_url(monkeypatch):
    import efficient.cli.main as cli_main
    captured = {}

    def fake_execvpe(file, args, env):
        captured.update(file=file, args=args, env=env)
        raise SystemExit(0)

    monkeypatch.setattr(cli_main.os, "execvpe", fake_execvpe)
    monkeypatch.setattr(cli_main.httpx, "get", lambda *a, **k: type("R", (), {"status_code": 200})())
    from click.testing import CliRunner
    result = CliRunner().invoke(cli_main.cli, ["claude", "--", "-p", "hi"])
    assert result.exit_code == 0
    assert captured["file"] == "claude"
    assert captured["env"]["ANTHROPIC_BASE_URL"].startswith("http")
    assert "-p" in captured["args"]
