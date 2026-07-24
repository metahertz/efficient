import json

import httpx
import pytest
from httpx import ASGITransport

FIXED = [0.2] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("efficient.modules.semantic_cache.embed_query", lambda t: FIXED)
    monkeypatch.setattr("efficient.modules.semantic_cache.embed_documents",
                        lambda ts: [FIXED] * len(ts))


@pytest.fixture
async def client(efficient_db, monkeypatch):
    calls = {"n": 0}

    async def fake_upstream(base_url, api_key, model, messages):
        calls["n"] += 1
        return "upstream answer", 30, 5

    monkeypatch.setattr("efficient.daemon.providers.call_openai_upstream", fake_upstream)
    from efficient.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c._calls = calls
        yield c


BODY = {"model": "gpt-x", "messages": [{"role": "user", "content": "what is a mutex?"}]}


async def test_miss_forwards_and_shapes_response(client):
    r = await client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "upstream answer"
    assert data["usage"]["total_tokens"] == 35
    assert data["efficient"]["cache_hit"] is False
    assert client._calls["n"] == 1


async def test_second_identical_call_is_cache_hit(client):
    await client.post("/v1/chat/completions", json=BODY)
    r = await client.post("/v1/chat/completions", json=BODY)
    data = r.json()
    assert data["efficient"]["cache_hit"] is True
    assert data["choices"][0]["message"]["content"] == "upstream answer"
    assert client._calls["n"] == 1  # upstream NOT called again


async def test_streaming_yields_sse_frames(client):
    r = await client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "chat.completion.chunk" in r.text
    assert "data: [DONE]" in r.text


async def test_route_is_shim_not_gateway(client, monkeypatch):
    # if this hit the gateway catch-all it would try the real Anthropic upstream;
    # the shim's fake upstream returning our sentinel proves routing order.
    r = await client.post("/v1/chat/completions", json=BODY)
    assert r.json()["choices"][0]["message"]["content"] == "upstream answer"


async def test_capability_attribution(client):
    await client.post("/v1/chat/completions", json=BODY)
    r = await client.get("/metrics")
    clients = {c["client"]: c for c in r.json()["clients"]}
    assert "openai_shim" in clients["openai-client"]["capabilities"]
