import pytest
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from finops.daemon.dashboard_routes import router


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(router)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_dashboard_index_served(client):
    r = await client.get("/dashboard")
    assert r.status_code == 200
    assert "fullFinOps-AI" in r.text


async def test_dashboard_assets_served(client):
    js = await client.get("/dashboard/app.js")
    assert js.status_code == 200
    assert js.headers["content-type"].startswith("application/javascript")

    css = await client.get("/dashboard/style.css")
    assert css.status_code == 200
    assert css.headers["content-type"].startswith("text/css")
