import os
import httpx


def _base_url() -> str:
    return os.getenv("FINOPS_DAEMON_URL", "http://daemon:7432")


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60.0) as c:
        r = await c.post(path, json=payload)
        r.raise_for_status()
        return r.json()


async def optimize(prompt: str, context: str = "", agent_id: str = "default",
                   corpus_id: str | None = None, strategy: str | None = None) -> dict:
    return await _post("/optimize", {"prompt": prompt, "context": context,
                                     "agent_id": agent_id, "corpus_id": corpus_id,
                                     "strategy": strategy, "framework": "claude-code-mcp"})


async def codebase_index(repo_id: str, path: str) -> dict:
    return await _post("/codebase/index", {"repo_id": repo_id, "path": path})


async def codebase_query(query: str, repo_id: str, k: int = 5) -> dict:
    return await _post("/codebase/query", {"repo_id": repo_id, "query": query, "k": k})


async def codebase_index_file(repo_id: str, file_path: str, source: str) -> dict:
    return await _post("/codebase/index-file", {"repo_id": repo_id, "file_path": file_path, "source": source})


async def codebase_references(repo_id: str, symbol: str) -> dict:
    return await _post("/codebase/references", {"repo_id": repo_id, "symbol": symbol})


async def memory_retrieve(agent_id: str, query: str) -> dict:
    return await _post("/memory/retrieve", {"agent_id": agent_id, "query": query})


async def memory_store(agent_id: str, session_id: str, turn: str, response: str) -> dict:
    return await _post("/memory/store", {"agent_id": agent_id, "session_id": session_id,
                                         "turn": turn, "response": response})
