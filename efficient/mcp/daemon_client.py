import os
import httpx


def _base_url() -> str:
    return os.getenv("EFFICIENT_DAEMON_URL", "http://daemon:7432")


def _auth_headers() -> dict:
    headers = {"X-Efficient-Client": "claude-code"}
    token = os.getenv("EFFICIENT_API_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _post(path: str, payload: dict) -> dict:
    async with httpx.AsyncClient(base_url=_base_url(), timeout=60.0, headers=_auth_headers()) as c:
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


async def corpus_add_chunks(corpus_id: str, chunks: list[dict]) -> dict:
    return await _post("/corpus/add-chunks", {"corpus_id": corpus_id, "chunks": chunks})


async def corpus_remove_file(corpus_id: str, source_file: str) -> dict:
    return await _post("/corpus/remove-file", {"corpus_id": corpus_id, "source_file": source_file})


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
