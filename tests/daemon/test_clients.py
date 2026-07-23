import pytest

from efficient.daemon import clients


@pytest.fixture(autouse=True)
def fresh():
    clients.reset()
    yield
    clients.reset()


class _Headers(dict):
    def get(self, k, default=None):
        return super().get(k.lower(), default)


def test_capability_mapping():
    assert clients.capability_for("/codebase/query") == "codebase_graph"
    assert clients.capability_for("/v1/messages") == "gateway"
    assert clients.capability_for("/memory/tool") == "memory_tool"
    assert clients.capability_for("/memory/retrieve") == "agent_memory"
    assert clients.capability_for("/optimize") == "optimize_pipeline"
    assert clients.capability_for("/complete") == "complete_proxy"
    assert clients.capability_for("/health") is None
    assert clients.capability_for("/metrics") is None


def test_resolve_client_prefers_header():
    assert clients.resolve_client("/codebase/query",
                                  _Headers({"x-efficient-client": "claude-code"})) == "claude-code"


def test_resolve_client_gateway_session():
    assert clients.resolve_client("/v1/messages",
                                  _Headers({"x-claude-code-session-id": "s1"})) == "claude-code"
    assert clients.resolve_client("/v1/messages", _Headers()) == "api-client"


def test_snapshot_groups_capabilities():
    clients.note("claude-code", "codebase_graph")
    clients.note("claude-code", "agent_memory")
    clients.note("agent-sdk", "memory_tool")
    snap = clients.snapshot()
    by = {c["client"]: c for c in snap}
    assert by["claude-code"]["capabilities"] == ["agent_memory", "codebase_graph"]
    assert by["agent-sdk"]["capabilities"] == ["memory_tool"]
    assert all("last_seen_s" in c for c in snap)


def test_snapshot_excludes_stale():
    import time
    clients.note("old-client", "gateway")
    clients._clients["old-client"]["gateway"] = time.time() - 10_000
    assert clients.snapshot(window_s=900) == []
