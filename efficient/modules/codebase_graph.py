import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from motor.motor_asyncio import AsyncIOMotorDatabase

from efficient.modules._base import BaseModule, OptimizeRequest, ModuleResult
from efficient.modules.embeddings import embed_query, embed_documents
from efficient.db.collections import CODEBASE_NODES
from efficient.db.vector import vector_search

_PY_LANGUAGE = Language(tspython.language())
_PARSER = Parser(_PY_LANGUAGE)

_SYMBOL_TYPES = {
    "function_definition": "function",
    "async_function_definition": "function",
    "class_definition": "class",
}


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _extract_calls(node) -> list[str]:
    names = set()
    for n in _walk(node):
        if n.type == "call":
            fn = n.child_by_field_name("function")
            if fn is None:
                continue
            if fn.type == "identifier":
                names.add(fn.text.decode("utf-8"))
            elif fn.type == "attribute":
                attr = fn.child_by_field_name("attribute")
                if attr is not None:
                    names.add(attr.text.decode("utf-8"))
    return sorted(names)


def _extract_python_symbols(source: str, file_path: str, repo_id: str) -> list[dict]:
    tree = _PARSER.parse(source.encode())
    symbols = []
    for node in _walk(tree.root_node):
        symbol_type = _SYMBOL_TYPES.get(node.type)
        if not symbol_type:
            continue
        name_node = node.child_by_field_name("name")
        if not name_node:
            continue
        snippet = source[node.start_byte:node.end_byte]
        symbols.append({
            "repo_id": repo_id,
            "symbol": name_node.text.decode("utf-8"),
            "type": symbol_type,
            "file_path": file_path,
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "source_snippet": snippet,
            "language": "python",
            "references": _extract_calls(node),
        })
    return symbols


_EXTRACTORS = {
    ".py": _extract_python_symbols,
}


class CodebaseGraph(BaseModule):
    name = "codebase_graph"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._repo_paths: list[str] = config.get("repo_paths", [])
        # the hooks/MCP tools index under "project"; keep the pipeline on the
        # same id unless explicitly configured otherwise
        self._repo_id: str = config.get("repo_id") or (
            self._repo_paths[0] if self._repo_paths else "project"
        )

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()
        repo_id = self._repo_id
        results = await self.query(repo_id, request.prompt)
        if not results:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=(time.perf_counter() - t0) * 1000, detail="no symbols matched",
            )
        snippets = "\n\n".join(
            f"# {r['file_path']}:{r['line_start']}\n{r['source_snippet']}"
            for r in results
        )
        baseline_tokens = await self._repo_symbol_tokens(repo_id)
        tokens_added = _count_tokens(snippets)
        tokens_in = _count_tokens(request.context)
        section = "## Relevant Code\n" + snippets
        new_context = request.context + ("\n\n" if request.context else "") + section
        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=new_context,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=_count_tokens(new_context),
            tokens_saved=max(0, baseline_tokens - tokens_added),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"injected {len(results)} symbols (baseline={baseline_tokens} full-index tokens)",
            tokens_added=tokens_added,
            baseline_tokens=baseline_tokens,
        )

    async def clear_repo(self, repo_id: str) -> int:
        result = await self._db[CODEBASE_NODES].delete_many({"repo_id": repo_id})
        return result.deleted_count

    async def index_file(self, repo_id: str, file_path: str, source: str) -> int:
        ext = Path(file_path).suffix
        extractor = _EXTRACTORS.get(ext)
        if not extractor:
            return 0
        await self._db[CODEBASE_NODES].delete_many({"repo_id": repo_id, "file_path": file_path})
        symbols = extractor(source, file_path, repo_id)
        if not symbols:
            return 0
        snippets = [s["source_snippet"] for s in symbols]
        embeddings = await asyncio.to_thread(embed_documents, snippets)
        now = datetime.now(timezone.utc)
        file_tokens = _count_tokens(source)
        docs = [{**sym, "embedding": emb, "indexed_at": now, "file_tokens": file_tokens}
                for sym, emb in zip(symbols, embeddings)]
        await self._db[CODEBASE_NODES].insert_many(docs)
        return len(symbols)

    async def query(self, repo_id: str, query_text: str, k: int = 5) -> list[dict]:
        embedding = await asyncio.to_thread(embed_query, query_text)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "codebase_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": k * 4,
                    "limit": k,
                    "filter": {"repo_id": {"$eq": repo_id}},
                }
            },
            {"$project": {"embedding": 0}},
        ]
        results = []
        for doc in await vector_search(self._db[CODEBASE_NODES], pipeline):
            results.append(doc)
        return results

    def _node_view(self, doc: dict) -> dict:
        return {
            "symbol": doc.get("symbol"),
            "type": doc.get("type"),
            "file_path": doc.get("file_path"),
            "line_start": doc.get("line_start"),
            "line_end": doc.get("line_end"),
            "references": doc.get("references", []),
        }

    async def callees(self, repo_id: str, symbol: str) -> list[dict]:
        doc = await self._db[CODEBASE_NODES].find_one({"repo_id": repo_id, "symbol": symbol})
        if not doc:
            return []
        names = doc.get("references", [])
        if not names:
            return []
        results = []
        async for d in self._db[CODEBASE_NODES].find(
            {"repo_id": repo_id, "symbol": {"$in": list(names)}}, {"embedding": 0}
        ):
            results.append(self._node_view(d))
        return results

    async def callers(self, repo_id: str, symbol: str) -> list[dict]:
        results = []
        async for d in self._db[CODEBASE_NODES].find(
            {"repo_id": repo_id, "references": symbol}, {"embedding": 0}
        ):
            results.append(self._node_view(d))
        return results

    async def _repo_symbol_tokens(self, repo_id: str) -> int:
        total = 0
        async for doc in self._db[CODEBASE_NODES].find(
            {"repo_id": repo_id}, {"source_snippet": 1}
        ):
            total += _count_tokens(doc.get("source_snippet", ""))
        return total
