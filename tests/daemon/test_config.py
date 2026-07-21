import pytest
from httpx import AsyncClient, ASGITransport
from efficient.daemon.app import app


async def test_get_config_returns_defaults(async_client):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "modules" in data
    assert "semantic_cache" in data["modules"]
    assert data["modules"]["semantic_cache"]["enabled"] is True
    assert data["embedding_model"] == "voyage-4-nano"


async def test_put_config_disables_module(async_client):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.put(
            "/config",
            json={"modules": {"semantic_cache": {"enabled": False}}}
        )
    assert r.status_code == 200
    data = r.json()
    assert data["modules"]["semantic_cache"]["enabled"] is False


async def test_config_has_no_id_field(async_client):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        r = await client.get("/config")
    assert "_id" not in r.json()


async def test_put_config_preserves_other_modules(async_client):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.put("/config", json={"modules": {"semantic_cache": {"enabled": False}}})
    data = r.json()
    assert data["modules"]["semantic_cache"]["enabled"] is False
    assert "agent_memory" in data["modules"]  # must not be deleted
    assert "codebase_graph" in data["modules"]  # must not be deleted


async def test_put_config_ignores_id_field(async_client):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.put("/config", json={"_id": "evil", "modules": {"semantic_cache": {"enabled": False}}})
    assert r.status_code == 200
