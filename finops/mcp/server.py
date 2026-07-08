import logging
import sys

from mcp.server.fastmcp import FastMCP

from finops.mcp import daemon_client

logging.basicConfig(level=logging.INFO, stream=sys.stderr)

mcp = FastMCP("efficient")


@mcp.tool()
async def optimize_context(prompt: str, context: str = "", agent_id: str = "default",
                           corpus_id: str | None = None, strategy: str | None = None) -> dict:
    """Route a prompt and its context through the finops optimization pipeline (semantic cache, codebase graph, retrieval, memory, compression). Returns the trimmed optimized_context, tokens_saved, cache_hit, and per-module detail."""
    return await daemon_client.optimize(prompt, context, agent_id, corpus_id, strategy)


@mcp.tool()
async def index_codebase(repo_id: str, path: str) -> dict:
    """Index a repository directory (.py files) into the codebase graph so its symbols can be looked up later. Returns counts of indexed files and symbols."""
    return await daemon_client.codebase_index(repo_id, path)


@mcp.tool()
async def lookup_symbol(query: str, repo_id: str, k: int = 5) -> dict:
    """Retrieve the most relevant code slices for a symbol name or natural-language description, instead of reading whole files. Returns up to k matching symbols with their source snippets."""
    return await daemon_client.codebase_query(query, repo_id, k)


@mcp.tool()
async def retrieve_memory(agent_id: str, query: str) -> dict:
    """Retrieve working, episodic, and semantic memory relevant to a query for an agent."""
    return await daemon_client.memory_retrieve(agent_id, query)


@mcp.tool()
async def store_memory(agent_id: str, session_id: str, turn: str, response: str) -> dict:
    """Store a conversation turn and extract durable facts into the agent's long-term memory."""
    return await daemon_client.memory_store(agent_id, session_id, turn, response)


@mcp.tool()
async def find_references(repo_id: str, symbol: str) -> dict:
    """Find the call/dependency edges for a symbol in an indexed repo: which symbols call it (callers) and which symbols it calls (callees). Use to trace impact and navigate the codebase graph."""
    return await daemon_client.codebase_references(repo_id, symbol)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
