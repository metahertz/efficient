import pytest
from finops.mcp import daemon_client


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    calls = []

    def __init__(self, base_url=None, timeout=None):
        self.base_url = base_url
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, path, json=None):
        _FakeClient.calls.append({"base_url": self.base_url, "path": path, "json": json})
        return _FakeResponse({"echoed": path})


@pytest.fixture(autouse=True)
def patch_httpx(monkeypatch):
    _FakeClient.calls = []
    monkeypatch.setattr("finops.mcp.daemon_client.httpx.AsyncClient", _FakeClient)


def _last():
    return _FakeClient.calls[-1]


def test_base_url_default(monkeypatch):
    monkeypatch.delenv("FINOPS_DAEMON_URL", raising=False)
    assert daemon_client._base_url() == "http://daemon:7432"


def test_base_url_override(monkeypatch):
    monkeypatch.setenv("FINOPS_DAEMON_URL", "http://localhost:9999")
    assert daemon_client._base_url() == "http://localhost:9999"


async def test_optimize_posts_optimize(monkeypatch):
    monkeypatch.setenv("FINOPS_DAEMON_URL", "http://localhost:9999")
    out = await daemon_client.optimize("p", "c", agent_id="a1", corpus_id="corp", strategy="s1")
    call = _last()
    assert call["base_url"] == "http://localhost:9999"
    assert call["path"] == "/optimize"
    assert call["json"] == {"prompt": "p", "context": "c", "agent_id": "a1",
                            "corpus_id": "corp", "strategy": "s1", "framework": "claude-code-mcp"}
    assert out == {"echoed": "/optimize"}


async def test_optimize_defaults():
    out = await daemon_client.optimize("hello")
    call = _last()
    assert call["json"] == {"prompt": "hello", "context": "", "agent_id": "default",
                            "corpus_id": None, "strategy": None, "framework": "claude-code-mcp"}
    assert out == {"echoed": "/optimize"}


async def test_codebase_index_posts_index():
    await daemon_client.codebase_index("r1", "/workspace")
    call = _last()
    assert call["path"] == "/codebase/index"
    assert call["json"] == {"repo_id": "r1", "path": "/workspace"}


async def test_codebase_query_posts_query():
    await daemon_client.codebase_query("find add", "r1", k=3)
    call = _last()
    assert call["path"] == "/codebase/query"
    assert call["json"] == {"repo_id": "r1", "query": "find add", "k": 3}


async def test_memory_retrieve_posts_retrieve():
    await daemon_client.memory_retrieve("a1", "what did I say")
    call = _last()
    assert call["path"] == "/memory/retrieve"
    assert call["json"] == {"agent_id": "a1", "query": "what did I say"}


async def test_memory_store_posts_store():
    await daemon_client.memory_store("a1", "s1", "turn text", "response text")
    call = _last()
    assert call["path"] == "/memory/store"
    assert call["json"] == {"agent_id": "a1", "session_id": "s1",
                            "turn": "turn text", "response": "response text"}
