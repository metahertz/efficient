import httpx
import pytest

pytestmark = pytest.mark.integration


def test_lifespan_boots_and_health_reports_ok(live_daemon):
    r = httpx.get(f"{live_daemon}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_config_seeded_by_lifespan(live_daemon):
    r = httpx.get(f"{live_daemon}/config", timeout=5.0)
    assert r.status_code == 200
    assert "modules" in r.json()


def test_optimize_end_to_end(live_daemon):
    r = httpx.post(
        f"{live_daemon}/optimize",
        json={"prompt": "What is a mutex?", "context": "", "agent_id": "live-harness"},
        timeout=300.0,  # first call loads the embedding model
    )
    assert r.status_code == 200
    body = r.json()
    assert "optimized_context" in body
    assert "module_results" in body


def test_validation_enforced_on_live_daemon(live_daemon):
    r = httpx.post(f"{live_daemon}/codebase/query",
                   json={"repo_id": "r", "query": "q", "k": 10**9}, timeout=5.0)
    assert r.status_code == 422
