import pytest
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

pytestmark = pytest.mark.integration


async def _run_session(callback):
    params = StdioServerParameters(command="python", args=["-m", "finops.mcp.server"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await callback(session)


async def test_mcp_lists_five_tools():
    async def cb(session):
        return await session.list_tools()
    result = await _run_session(cb)
    names = {t.name for t in result.tools}
    assert names == {"optimize_context", "index_codebase", "lookup_symbol",
                     "retrieve_memory", "store_memory"}


async def test_mcp_optimize_context_roundtrip():
    async def cb(session):
        return await session.call_tool("optimize_context", {
            "prompt": "What is Python?", "context": "some context", "agent_id": "smoke",
        })
    result = await _run_session(cb)
    payload = result.structuredContent if getattr(result, "structuredContent", None) else None
    if payload is None:
        import json
        payload = json.loads(result.content[0].text)
    assert "optimized_context" in payload
    assert "tokens_saved" in payload
