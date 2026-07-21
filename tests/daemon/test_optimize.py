import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport
from efficient.daemon.app import app
from efficient.modules._base import OptimizeRequest, ModuleResult
from efficient.daemon.strategies import COMPOSE_THEN_COMPRESS, Strategy
from efficient.db.collections import REQUEST_LOG


def _make_append_module(name, section_label):
    mod = MagicMock()
    mod.name = name
    async def proc(req):
        section = f"## {section_label}\nfrom-{name}"
        new_ctx = req.context + ("\n\n" if req.context else "") + section
        new_req = OptimizeRequest(prompt=req.prompt, context=new_ctx,
                                  agent_id=req.agent_id, framework=req.framework,
                                  corpus_id=req.corpus_id)
        return new_req, ModuleResult(module=name, tokens_in=1, tokens_out=1,
                                     tokens_saved=0, latency_ms=1.0, detail="append",
                                     tokens_added=3, baseline_tokens=10)
    mod.process = proc
    return mod


def _make_cache_hit_module():
    mod = MagicMock()
    mod.name = "semantic_cache"
    async def hit(req):
        new_req = OptimizeRequest(prompt=req.prompt, context="cached response",
                                  agent_id=req.agent_id, framework=req.framework,
                                  corpus_id=req.corpus_id)
        return new_req, ModuleResult(module="semantic_cache", tokens_in=500, tokens_out=0,
                                     tokens_saved=500, latency_ms=5.0, detail="exact hash hit",
                                     short_circuit=True, baseline_tokens=500)
    mod.process = hit
    return mod


def _make_passthrough_module(name):
    mod = MagicMock()
    mod.name = name
    mod.process = AsyncMock(side_effect=lambda req: (
        req, ModuleResult(module=name, tokens_in=10, tokens_out=10,
                          tokens_saved=0, latency_ms=1.0, detail="pass")))
    return mod


@pytest.fixture
async def client(efficient_db):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_optimize_returns_shape_with_strategy(client, efficient_db):
    from efficient.daemon.config import save_config
    await save_config(efficient_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": False},
        "agent_memory": {"enabled": False}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    resp = await client.post("/optimize", json={
        "prompt": "What is Python?", "context": "some context",
        "agent_id": "a1", "framework": "test",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "optimized_prompt" in data
    assert "optimized_context" in data
    assert "tokens_saved" in data
    assert "module_results" in data
    assert data["strategy"] == "compose_then_compress"
    assert data["module_results"] == []


async def test_optimize_preserves_prompt(client, efficient_db):
    from efficient.daemon.config import save_config
    await save_config(efficient_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": False},
        "agent_memory": {"enabled": False}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    resp = await client.post("/optimize", json={
        "prompt": "unique test prompt xyz", "context": "",
        "agent_id": "a1", "framework": "test",
    })
    assert resp.json()["optimized_prompt"] == "unique test prompt xyz"


async def test_pipeline_short_circuits_on_cache_hit(efficient_db):
    from efficient.daemon.router import ModulePipeline
    cache_mod = _make_cache_hit_module()
    other = _make_passthrough_module("context_compressor")
    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._strategy = COMPOSE_THEN_COMPRESS
    pipeline._modules = {"semantic_cache": cache_mod, "context_compressor": other}
    pipeline._enabled = {"semantic_cache": True, "context_compressor": True}
    req = OptimizeRequest(prompt="hi", context="ctx", agent_id="a", framework="f")
    result = await pipeline.run(req)
    assert result["cache_hit"] is True
    assert result["optimized_context"] == "cached response"
    assert result["tokens_saved"] == 500
    assert result["strategy"] == "compose_then_compress"
    other.process.assert_not_called()


async def test_pipeline_skips_disabled_modules(efficient_db):
    from efficient.daemon.router import ModulePipeline
    mod = _make_passthrough_module("context_compressor")
    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._strategy = COMPOSE_THEN_COMPRESS
    pipeline._modules = {"context_compressor": mod}
    pipeline._enabled = {"context_compressor": False}
    req = OptimizeRequest(prompt="hi", context="ctx", agent_id="a", framework="f")
    result = await pipeline.run(req)
    mod.process.assert_not_called()
    assert result["tokens_saved"] == 0


async def test_pipeline_composes_both_augmenters(efficient_db):
    from efficient.daemon.router import ModulePipeline
    graph = _make_append_module("codebase_graph", "Relevant Code")
    memory = _make_append_module("agent_memory", "Memory")
    strat = Strategy(name="two_aug", order=("codebase_graph", "agent_memory"),
                     composition="compose", cache_key="prompt+scope")
    pipeline = ModulePipeline.__new__(ModulePipeline)
    pipeline._strategy = strat
    pipeline._modules = {"codebase_graph": graph, "agent_memory": memory}
    pipeline._enabled = {"codebase_graph": True, "agent_memory": True}
    req = OptimizeRequest(prompt="hi", context="ORIG", agent_id="a", framework="f")
    result = await pipeline.run(req)
    ctx = result["optimized_context"]
    assert "ORIG" in ctx
    assert "## Relevant Code" in ctx
    assert "## Memory" in ctx
    assert result["cache_hit"] is False
    assert len(result["module_results"]) == 2


async def test_optimize_persists_module_events(client, efficient_db, monkeypatch):
    from efficient.db.collections import REQUEST_LOG

    class _FakePipeline:
        def __init__(self, *a, **kw):
            pass
        async def run(self, request):
            return {
                "optimized_prompt": request.prompt,
                "optimized_context": "ctx",
                "cache_hit": False,
                "strategy": "compose_then_compress",
                "tokens_saved": 100,
                "module_results": [
                    {"module": "codebase_graph", "tokens_in": 0, "tokens_out": 0,
                     "tokens_saved": 100, "tokens_added": 20, "baseline_tokens": 120,
                     "latency_ms": 1.0, "detail": "x"},
                ],
            }

    monkeypatch.setattr("efficient.daemon.router.ModulePipeline", _FakePipeline)
    resp = await client.post("/optimize", json={"prompt": "p", "context": "c", "agent_id": "a1", "framework": "test"})
    assert resp.status_code == 200
    count = await efficient_db[REQUEST_LOG].count_documents({"module": "codebase_graph"})
    assert count == 1


async def test_optimize_logs_module_events(client, efficient_db):
    from efficient.daemon.config import save_config
    # All modules disabled → module_results empty → record_module_events no-ops.
    await save_config(efficient_db, {"modules": {
        "codebase_graph": {"enabled": False}, "semantic_cache": {"enabled": False},
        "agent_memory": {"enabled": False}, "context_compressor": {"enabled": False},
        "hybrid_retrieval": {"enabled": False},
    }})
    resp = await client.post("/optimize", json={
        "prompt": "What is Python?", "context": "some context",
        "agent_id": "a1", "framework": "test",
    })
    assert resp.status_code == 200
    assert resp.json()["module_results"] == []
    # No module ran, so request_log stays empty (wiring path must not crash).
    assert await efficient_db[REQUEST_LOG].count_documents({}) == 0
