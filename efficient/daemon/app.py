import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from efficient.daemon.auth import require_token
from efficient.daemon import schemas
from efficient.db.client import get_async_db, get_sync_db
from efficient.db.indexes import create_all_indexes
from efficient.daemon.config import load_config, save_config, validate_patch
from efficient.db.collections import CACHE_ENTRIES
from efficient.db.vector import vector_search
from efficient.daemon.strategies import get_strategy
from efficient.modules._base import OptimizeRequest

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    VERSION = _pkg_version("efficient")
except PackageNotFoundError:
    VERSION = "0.0.0-dev"


def _check_prerequisites(sync_db) -> None:
    info = sync_db.command("buildInfo")
    major = int(info["version"].split(".")[0])
    if major < 7:
        raise SystemExit(
            f"ERROR: MongoDB >= 7.0 required, found {info['version']}.\n"
            "Run: docker run -p 27017:27017 mongodb/mongodb-atlas-local:latest"
        )
    try:
        list(sync_db["config"].list_search_indexes())
    except Exception as exc:
        raise SystemExit(
            "ERROR: MongoDB Atlas Search (mongot) not available.\n"
            "Run: docker run -p 27017:27017 mongodb/mongodb-atlas-local:latest\n"
            f"Detail: {exc}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_db = get_sync_db()
    _check_prerequisites(sync_db)
    create_all_indexes(sync_db)
    db = get_async_db()
    await load_config(db)
    yield


app = FastAPI(title="efficient Daemon", lifespan=lifespan, dependencies=[Depends(require_token)])

from efficient.daemon.dashboard_routes import router as dashboard_router
app.include_router(dashboard_router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}


@app.get("/status")
async def get_status(since: int = 0):
    from efficient import activity
    return activity.snapshot(since)


@app.get("/config")
async def get_config():
    db = get_async_db()
    config = await load_config(db)
    config.pop("_id", None)
    return config


@app.put("/config")
async def put_config(patch: dict):
    patch.pop("_id", None)
    patch.pop("updated_at", None)
    try:
        validate_patch(patch)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    db = get_async_db()
    config = await save_config(db, patch)
    config.pop("_id", None)
    return config


@app.post("/optimize")
async def post_optimize(body: schemas.OptimizeBody):
    db = get_async_db()
    config = await load_config(db)
    strategy = get_strategy(body.strategy or config.get("strategy"))
    from efficient.daemon.router import ModulePipeline
    pipeline = ModulePipeline(db, config.get("modules", {}), strategy)
    request = OptimizeRequest(
        prompt=body.prompt,
        context=body.context,
        agent_id=body.agent_id,
        framework=body.framework,
        corpus_id=body.corpus_id,
    )
    result = await pipeline.run(request)
    from efficient.daemon.metrics import record_module_events
    await record_module_events(db, result["module_results"])
    return result


@app.post("/complete")
async def post_complete(body: schemas.CompleteBody):
    db = get_async_db()
    config = await load_config(db)
    strategy = get_strategy(body.strategy or config.get("strategy"))
    from efficient.daemon.router import ModulePipeline
    pipeline = ModulePipeline(db, config.get("modules", {}), strategy)
    request = OptimizeRequest(
        prompt=body.prompt,
        context=body.context,
        agent_id=body.agent_id,
        framework=body.framework,
        corpus_id=body.corpus_id,
    )
    optimized = await pipeline.run(request)
    from efficient.daemon.metrics import record_module_events
    await record_module_events(db, optimized["module_results"])

    if optimized["cache_hit"]:
        return {
            "response": optimized["optimized_context"],
            "tokens_saved": optimized["tokens_saved"],
            "cache_hit": True,
            "input_tokens": 0,
            "output_tokens": 0,
            "module_results": optimized["module_results"],
        }

    from efficient.daemon.providers import call_llm
    response_text, input_tokens, output_tokens = await call_llm(
        provider=body.provider,
        model=body.model,
        prompt=optimized["optimized_prompt"],
        context=optimized["optimized_context"],
    )

    cache_cfg = {**config.get("modules", {}).get("semantic_cache", {}), "cache_key": strategy.cache_key}
    from efficient.modules.semantic_cache import SemanticCache
    cache = SemanticCache(db, cache_cfg)
    await cache.store(
        prompt=request.prompt,
        response=response_text,
        framework=request.framework,
        model=body.model,
        tokens_saved=input_tokens + output_tokens,
        agent_id=request.agent_id,
        corpus_id=request.corpus_id or "",
    )

    from efficient.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, config.get("modules", {}).get("agent_memory", {}))
    await memory.store_turn(
        request.agent_id, body.session_id, request.prompt, response_text
    )

    return {
        "response": response_text,
        "tokens_saved": optimized["tokens_saved"],
        "cache_hit": False,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "module_results": optimized["module_results"],
    }


@app.get("/cache/lookup")
async def cache_lookup(prompt_hash: str, embedding: list[float] | None = None):
    db = get_async_db()
    config = await load_config(db)
    cache_cfg = config.get("modules", {}).get("semantic_cache", {})
    entry = await db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
    if entry:
        await db[CACHE_ENTRIES].update_one({"_id": entry["_id"]}, {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}})
        return {"hit": True, "response": entry["response"], "similarity_score": 1.0}
    if embedding:
        threshold = cache_cfg.get("similarity_threshold", 0.80)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "cache_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 1,
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"_score": {"$gte": threshold}}},
        ]
        for doc in await vector_search(db[CACHE_ENTRIES], pipeline):
            await db[CACHE_ENTRIES].update_one({"_id": doc["_id"]}, {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}})
            return {"hit": True, "response": doc["response"], "similarity_score": doc["_score"]}
    return {"hit": False, "response": None, "similarity_score": 0.0}


@app.post("/cache/store")
async def cache_store(body: schemas.CacheStoreBody):
    db = get_async_db()
    config = await load_config(db)
    cache_cfg = config.get("modules", {}).get("semantic_cache", {})
    strategy = get_strategy(config.get("strategy"))
    cache_cfg = {**cache_cfg, "cache_key": strategy.cache_key}
    from efficient.modules.semantic_cache import SemanticCache
    cache = SemanticCache(db, cache_cfg)
    await cache.store(
        prompt=body.prompt,
        response=body.response,
        framework=body.framework,
        model=body.model,
        tokens_saved=body.tokens_saved,
        agent_id=body.agent_id,
        corpus_id=body.corpus_id,
    )
    return {"stored": True}


@app.post("/memory/retrieve")
async def memory_retrieve(body: schemas.MemoryRetrieveBody):
    agent_id = body.agent_id
    query = body.query
    db = get_async_db()
    config = await load_config(db)
    mem_cfg = config.get("modules", {}).get("agent_memory", {})
    from efficient.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, mem_cfg)
    working = await memory._get_working_memory(agent_id)
    episodic = await memory._get_episodic_memory(agent_id, query)
    semantic = await memory._get_semantic_memory(agent_id, query)
    return {"working": working, "episodic": episodic, "semantic": semantic}


@app.post("/memory/store")
async def memory_store(body: schemas.MemoryStoreBody):
    agent_id = body.agent_id
    session_id = body.session_id
    turn = body.turn
    response = body.response
    db = get_async_db()
    config = await load_config(db)
    mem_cfg = config.get("modules", {}).get("agent_memory", {})
    from efficient.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, mem_cfg)
    await memory.store_turn(agent_id, session_id, turn, response)
    from efficient import activity
    activity.emit(f"stored memory turn (agent={agent_id})")
    return {"stored": True}


def _allowed_index_roots(cg_cfg: dict) -> list[Path]:
    roots = [Path(p).resolve() for p in cg_cfg.get("repo_paths", [])]
    env = os.getenv("EFFICIENT_ALLOWED_INDEX_ROOTS", "")
    roots += [Path(p).resolve() for p in env.split(":") if p]
    return roots


@app.post("/codebase/index")
async def codebase_index(body: schemas.CodebaseIndexBody):
    db = get_async_db()
    config = await load_config(db)
    cg_cfg = config.get("modules", {}).get("codebase_graph", {})
    root = Path(body.path).resolve() if body.path else None
    if root is None or not root.is_dir():
        return {"repo_id": body.repo_id, "indexed_files": 0, "indexed_symbols": 0}
    allowed = _allowed_index_roots(cg_cfg)
    if not any(root == r or root.is_relative_to(r) for r in allowed):
        raise HTTPException(
            status_code=403,
            detail="path not under an allowed index root "
                   "(configure modules.codebase_graph.repo_paths or EFFICIENT_ALLOWED_INDEX_ROOTS)",
        )
    from efficient.modules.codebase_graph import CodebaseGraph
    from efficient import activity
    graph = CodebaseGraph(db, cg_cfg)
    files = 0
    symbols = 0
    with activity.activity(f"indexing repo {body.repo_id} from {root}", notify=True):
        await graph.clear_repo(body.repo_id)
        for py in root.rglob("*.py"):
            try:
                source = py.read_text(encoding="utf-8")
            except Exception:
                continue
            n = await graph.index_file(body.repo_id, str(py.relative_to(root)), source)
            if n:
                files += 1
                symbols += n
    activity.emit(f"repo {body.repo_id}: {files} files, {symbols} symbols indexed")
    return {"repo_id": body.repo_id, "indexed_files": files, "indexed_symbols": symbols}


@app.post("/codebase/query")
async def codebase_query(body: schemas.CodebaseQueryBody):
    repo_id = body.repo_id
    query = body.query
    k = body.k
    db = get_async_db()
    config = await load_config(db)
    cg_cfg = config.get("modules", {}).get("codebase_graph", {})
    from efficient.modules.codebase_graph import CodebaseGraph
    graph = CodebaseGraph(db, cg_cfg)
    import time as _time
    t0 = _time.perf_counter()
    results = await graph.query(repo_id, query, k)
    out = [{
        "symbol": r.get("symbol"), "type": r.get("type"),
        "file_path": r.get("file_path"), "line_start": r.get("line_start"),
        "line_end": r.get("line_end"), "source_snippet": r.get("source_snippet"),
    } for r in results]
    if results:
        # Honest savings: the counterfactual to a symbol lookup is reading the
        # whole file(s) the symbols came from (what the read-steer hook avoids).
        from efficient.modules.codebase_graph import _count_tokens
        file_tokens = {}
        for r in results:
            file_tokens[r.get("file_path")] = max(
                file_tokens.get(r.get("file_path"), 0), int(r.get("file_tokens", 0))
            )
        baseline = sum(file_tokens.values())
        returned = sum(_count_tokens(r.get("source_snippet", "")) for r in results)
        from efficient.daemon.metrics import record_module_events
        await record_module_events(db, [{
            "module": "codebase_graph",
            "tokens_saved": max(0, baseline - returned),
            "tokens_added": returned,
            "baseline_tokens": baseline,
            "latency_ms": (_time.perf_counter() - t0) * 1000,
        }])
    return {"repo_id": repo_id, "results": out}


@app.post("/codebase/index-file")
async def codebase_index_file(body: schemas.CodebaseIndexFileBody):
    repo_id = body.repo_id
    file_path = body.file_path
    source = body.source
    db = get_async_db()
    config = await load_config(db)
    cg_cfg = config.get("modules", {}).get("codebase_graph", {})
    from efficient.modules.codebase_graph import CodebaseGraph
    graph = CodebaseGraph(db, cg_cfg)
    n = await graph.index_file(repo_id, file_path, source)
    from efficient import activity
    activity.note_indexed(file_path)
    return {"repo_id": repo_id, "file_path": file_path, "indexed_symbols": n}


@app.post("/codebase/references")
async def codebase_references(body: schemas.CodebaseReferencesBody):
    repo_id = body.repo_id
    symbol = body.symbol
    db = get_async_db()
    config = await load_config(db)
    cg_cfg = config.get("modules", {}).get("codebase_graph", {})
    from efficient.modules.codebase_graph import CodebaseGraph
    graph = CodebaseGraph(db, cg_cfg)
    return {
        "repo_id": repo_id,
        "symbol": symbol,
        "callers": await graph.callers(repo_id, symbol),
        "callees": await graph.callees(repo_id, symbol),
    }


@app.get("/metrics")
async def get_metrics():
    from efficient.daemon.metrics import aggregate_metrics
    db = get_async_db()
    return await aggregate_metrics(db)
