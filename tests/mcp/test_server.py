import pytest
from unittest.mock import AsyncMock
from finops.mcp import server


@pytest.fixture(autouse=True)
def patch_client(monkeypatch):
    m = {
        "optimize": AsyncMock(return_value={"tool": "optimize"}),
        "codebase_index": AsyncMock(return_value={"tool": "codebase_index"}),
        "codebase_query": AsyncMock(return_value={"tool": "codebase_query"}),
        "memory_retrieve": AsyncMock(return_value={"tool": "memory_retrieve"}),
        "memory_store": AsyncMock(return_value={"tool": "memory_store"}),
        "codebase_references": AsyncMock(return_value={"tool": "codebase_references"}),
        "codebase_index_file": AsyncMock(return_value={"tool": "codebase_index_file"}),
    }
    for name, mock in m.items():
        monkeypatch.setattr(server.daemon_client, name, mock)
    return m


async def test_optimize_context_delegates(patch_client):
    out = await server.optimize_context("p", "c", agent_id="a1", corpus_id="cp", strategy="s1")
    patch_client["optimize"].assert_awaited_once_with("p", "c", "a1", "cp", "s1")
    assert out == {"tool": "optimize"}


async def test_index_codebase_delegates(patch_client):
    out = await server.index_codebase("r1", "/workspace")
    patch_client["codebase_index"].assert_awaited_once_with("r1", "/workspace")
    assert out == {"tool": "codebase_index"}


async def test_lookup_symbol_delegates(patch_client):
    out = await server.lookup_symbol("find add", "r1", k=3)
    patch_client["codebase_query"].assert_awaited_once_with("find add", "r1", 3)
    assert out == {"tool": "codebase_query"}


async def test_retrieve_memory_delegates(patch_client):
    out = await server.retrieve_memory("a1", "what did I say")
    patch_client["memory_retrieve"].assert_awaited_once_with("a1", "what did I say")
    assert out == {"tool": "memory_retrieve"}


async def test_store_memory_delegates(patch_client):
    out = await server.store_memory("a1", "s1", "turn", "resp")
    patch_client["memory_store"].assert_awaited_once_with("a1", "s1", "turn", "resp")
    assert out == {"tool": "memory_store"}


async def test_find_references_delegates(patch_client):
    out = await server.find_references("r1", "helper")
    patch_client["codebase_references"].assert_awaited_once_with("r1", "helper")
    assert out == {"tool": "codebase_references"}


async def test_reindex_file_delegates(patch_client):
    out = await server.reindex_file("r1", "f.py", "src")
    patch_client["codebase_index_file"].assert_awaited_once_with("r1", "f.py", "src")
    assert out == {"tool": "codebase_index_file"}


async def test_all_seven_tools_registered():
    tools = await server.mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {"optimize_context", "index_codebase", "lookup_symbol",
                     "retrieve_memory", "store_memory", "find_references",
                     "reindex_file"}
