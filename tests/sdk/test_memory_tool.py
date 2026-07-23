from types import SimpleNamespace

import pytest

from efficient.sdk import EfficientMemoryTool


@pytest.fixture
def tool(monkeypatch):
    t = EfficientMemoryTool(daemon_url="http://fake:7432", agent_id="bot-1")
    calls = []

    def fake_post(command, **args):
        calls.append({"command": command, **args})
        return "ok-result"

    monkeypatch.setattr(t, "_post", fake_post)
    t._calls = calls
    return t


def test_tool_declares_memory_type():
    t = EfficientMemoryTool()
    d = t.to_dict()
    assert d["type"] == "memory_20250818"
    assert d["name"] == "memory"


def test_command_mapping(tool):
    tool.view(SimpleNamespace(path="/memories", view_range=None))
    tool.create(SimpleNamespace(path="/memories/a.md", file_text="x"))
    tool.str_replace(SimpleNamespace(path="/memories/a.md", old_str="x", new_str="y"))
    tool.insert(SimpleNamespace(path="/memories/a.md", insert_line=1, insert_text="z"))
    tool.delete(SimpleNamespace(path="/memories/a.md"))
    tool.rename(SimpleNamespace(old_path="/memories/a.md", new_path="/memories/b.md"))
    tool.clear_all_memory()
    commands = [c["command"] for c in tool._calls]
    assert commands == ["view", "create", "str_replace", "insert", "delete",
                        "rename", "clear_all"]
    assert tool._calls[1] == {"command": "create", "path": "/memories/a.md",
                              "file_text": "x"}
    assert tool._calls[5] == {"command": "rename", "old_path": "/memories/a.md",
                              "new_path": "/memories/b.md"}


def test_error_returned_as_text(monkeypatch):
    t = EfficientMemoryTool(daemon_url="http://fake:7432")

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"ok": False, "error": "not found: /memories/x"}

    import efficient.sdk.memory_tool as mt
    monkeypatch.setattr(mt.httpx, "post", lambda *a, **k: FakeResponse())
    out = t.view(SimpleNamespace(path="/memories/x", view_range=None))
    assert out == "Error: not found: /memories/x"


def test_bearer_header_when_token_set(monkeypatch):
    t = EfficientMemoryTool(daemon_url="http://fake:7432")
    monkeypatch.setenv("EFFICIENT_API_TOKEN", "sekret")
    captured = {}

    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {"ok": True, "result": "r"}

    import efficient.sdk.memory_tool as mt

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(mt.httpx, "post", fake_post)
    t.clear_all_memory()
    assert captured["headers"]["Authorization"] == "Bearer sekret"
