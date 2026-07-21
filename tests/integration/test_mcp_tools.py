import asyncio
import json

import pytest

from tests.integration.test_mcp_smoke import _run_session

pytestmark = pytest.mark.integration

SAMPLE_SOURCE = '''\
def greet(name):
    return helper(name)


def helper(name):
    return f"hello {name}"
'''


def _payload(result):
    payload = getattr(result, "structuredContent", None)
    if isinstance(payload, dict):
        return payload
    return json.loads(result.content[0].text)


async def test_memory_store_then_retrieve(live_daemon):
    async def cb(session):
        await session.call_tool("store_memory", {
            "agent_id": "mcp-harness", "session_id": "s1",
            "turn": "remember the port is 7432", "response": "noted: port 7432",
        })
        return await session.call_tool("retrieve_memory", {
            "agent_id": "mcp-harness", "query": "which port",
        })
    payload = _payload(await _run_session(live_daemon, cb))
    assert "working" in payload
    assert any("7432" in m.get("content", "") for m in payload["working"])


async def test_reindex_then_lookup_then_references(live_daemon):
    async def cb(session):
        await session.call_tool("reindex_file", {
            "repo_id": "mcp-harness", "file_path": "sample.py", "source": SAMPLE_SOURCE,
        })
        # lookup_symbol runs a vector search against a mongot search index; newly
        # indexed documents can lag a few seconds before mongot ingests them even
        # though the index itself is already queryable, so poll briefly here.
        lookup_payload = {}
        for _ in range(15):
            lookup = await session.call_tool("lookup_symbol", {
                "query": "greet", "repo_id": "mcp-harness",
            })
            lookup_payload = _payload(lookup)
            if any(r.get("symbol") == "greet" for r in lookup_payload.get("results", [])):
                break
            await asyncio.sleep(2)
        refs = await session.call_tool("find_references", {
            "repo_id": "mcp-harness", "symbol": "helper",
        })
        return lookup_payload, refs

    lookup_payload, refs = await _run_session(live_daemon, cb)
    assert any(r.get("symbol") == "greet" for r in lookup_payload.get("results", []))
    refs_payload = _payload(refs)
    assert "callers" in refs_payload
