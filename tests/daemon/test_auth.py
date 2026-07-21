import httpx
import pytest
from fastapi import HTTPException
from httpx import ASGITransport


@pytest.fixture
async def client(finops_db):
    from finops.daemon.app import app
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_no_token_configured_allows_all(client, monkeypatch):
    monkeypatch.delenv("FINOPS_API_TOKEN", raising=False)
    r = await client.get("/config")
    assert r.status_code == 200


async def test_token_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config")
    assert r.status_code == 401


async def test_wrong_token_rejected(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


async def test_correct_token_accepted(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config", headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 200


async def test_health_and_metrics_exempt(client, monkeypatch):
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    assert (await client.get("/health")).status_code == 200
    assert (await client.get("/metrics")).status_code == 200


def test_lookalike_path_not_exempt(monkeypatch):
    """Test that /healthz (lookalike to /health) is NOT exempt."""
    from finops.daemon.auth import require_token
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")

    # Build a stub Request object
    class StubRequest:
        class StubURL:
            path = "/healthz"
        url = StubURL()
        headers = {}

    request = StubRequest()
    with pytest.raises(HTTPException) as exc_info:
        import asyncio
        asyncio.run(require_token(request))
    assert exc_info.value.status_code == 401


async def test_non_bearer_auth_rejected(client, monkeypatch):
    """Test that Authorization: sekret (no Bearer prefix) is rejected."""
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config", headers={"Authorization": "sekret"})
    assert r.status_code == 401


async def test_bearer_with_whitespace_token_rejected(client, monkeypatch):
    """Test that Authorization: Bearer (whitespace only) is rejected."""
    monkeypatch.setenv("FINOPS_API_TOKEN", "sekret")
    r = await client.get("/config", headers={"Authorization": "Bearer   "})
    assert r.status_code == 401
