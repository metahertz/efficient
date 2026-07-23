import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
import tree_sitter_javascript as tsjavascript
from tree_sitter import Language, Parser
from motor.motor_asyncio import AsyncIOMotorDatabase

from efficient.modules._base import BaseModule, OptimizeRequest, ModuleResult
from efficient.modules.embeddings import embed_query, embed_documents
from efficient.db.collections import CODEBASE_NODES
from efficient.db.vector import vector_search


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


class _LangSpec:
    """Per-language extraction config: parser, symbol node types, and how to
    read a call's callee (bare-call type + callee field, member-access type +
    property field)."""

    def __init__(self, language, symbol_types, call_type, callee_field,
                 member_type, member_prop_field, arrow_types=()):
        self.parser = Parser(Language(language))
        self.symbol_types = symbol_types
        self.call_type = call_type
        self.callee_field = callee_field
        self.member_type = member_type
        self.member_prop_field = member_prop_field
        self.arrow_types = arrow_types  # value node types that make a var a "function"


_PY = _LangSpec(
    tspython.language(),
    {"function_definition": "function", "async_function_definition": "function",
     "class_definition": "class"},
    call_type="call", callee_field="function",
    member_type="attribute", member_prop_field="attribute",
)

_TS_SYMBOLS = {
    "function_declaration": "function",
    "class_declaration": "class",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
}
_TS = _LangSpec(
    tstypescript.language_typescript(), _TS_SYMBOLS,
    call_type="call_expression", callee_field="function",
    member_type="member_expression", member_prop_field="property",
    arrow_types=("arrow_function", "function_expression"),
)
_TSX = _LangSpec(
    tstypescript.language_tsx(), _TS_SYMBOLS,
    call_type="call_expression", callee_field="function",
    member_type="member_expression", member_prop_field="property",
    arrow_types=("arrow_function", "function_expression"),
)
_JS = _LangSpec(
    tsjavascript.language(),
    {"function_declaration": "function", "class_declaration": "class",
     "method_definition": "method"},
    call_type="call_expression", callee_field="function",
    member_type="member_expression", member_prop_field="property",
    arrow_types=("arrow_function", "function_expression"),
)

_LANGS = {
    ".py": ("python", _PY),
    ".ts": ("typescript", _TS),
    ".tsx": ("tsx", _TSX),
    ".js": ("javascript", _JS),
    ".jsx": ("javascript", _JS),
    ".mjs": ("javascript", _JS),
    ".cjs": ("javascript", _JS),
}


def _extract_calls(node, spec: _LangSpec) -> list[str]:
    names = set()
    for n in _walk(node):
        if n.type != spec.call_type:
            continue
        fn = n.child_by_field_name(spec.callee_field)
        if fn is None:
            continue
        if fn.type == "identifier":
            names.add(fn.text.decode("utf-8"))
        elif fn.type == spec.member_type:
            prop = fn.child_by_field_name(spec.member_prop_field)
            if prop is not None:
                names.add(prop.text.decode("utf-8"))
    return sorted(names)


def _symbol_name(node, spec: _LangSpec) -> tuple[str, str] | None:
    """Return (name, type) for a symbol-bearing node, or None."""
    stype = spec.symbol_types.get(node.type)
    if stype:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return name_node.text.decode("utf-8"), stype
        return None
    # arrow/function-expression assigned to a variable → treat as a function
    if node.type == "variable_declarator" and spec.arrow_types:
        value = node.child_by_field_name("value")
        if value is not None and value.type in spec.arrow_types:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                return name_node.text.decode("utf-8"), "function"
    return None


def _extract_symbols(source: str, file_path: str, repo_id: str) -> list[dict]:
    from pathlib import Path as _Path
    ext = _Path(file_path).suffix
    entry = _LANGS.get(ext)
    if entry is None:
        return []
    language_name, spec = entry
    tree = spec.parser.parse(source.encode())
    symbols = []
    for node in _walk(tree.root_node):
        resolved = _symbol_name(node, spec)
        if resolved is None:
            continue
        name, symbol_type = resolved
        symbols.append({
            "repo_id": repo_id,
            "symbol": name,
            "type": symbol_type,
            "file_path": file_path,
            "line_start": node.start_point[0] + 1,
            "line_end": node.end_point[0] + 1,
            "source_snippet": source[node.start_byte:node.end_byte],
            "language": language_name,
            "references": _extract_calls(node, spec),
        })
    return symbols


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
        if Path(file_path).suffix not in _LANGS:
            return 0
        await self._db[CODEBASE_NODES].delete_many({"repo_id": repo_id, "file_path": file_path})
        symbols = _extract_symbols(source, file_path, repo_id)
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
