from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from finops.db.client import get_async_db, get_sync_db
from finops.db.indexes import create_all_indexes
from finops.daemon.config import load_config, save_config
from finops.db.collections import CACHE_ENTRIES
from finops.daemon.strategies import get_strategy
from finops.modules._base import OptimizeRequest

VERSION = "0.1.0"


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


app = FastAPI(title="fullFinOps-AI Daemon", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}


@app.get("/config")
async def get_config():
    db = get_async_db()
    config = await load_config(db)
    config.pop("_id", None)
    return config


@app.put("/config")
async def put_config(patch: dict):
    patch.pop("_id", None)
    db = get_async_db()
    config = await save_config(db, patch)
    config.pop("_id", None)
    return config


@app.post("/optimize")
async def post_optimize(body: dict):
    db = get_async_db()
    config = await load_config(db)
    strategy = get_strategy(body.get("strategy") or config.get("strategy"))
    from finops.daemon.router import ModulePipeline
    pipeline = ModulePipeline(db, config.get("modules", {}), strategy)
    request = OptimizeRequest(
        prompt=body.get("prompt", ""),
        context=body.get("context", ""),
        agent_id=body.get("agent_id", "default"),
        framework=body.get("framework", "unknown"),
        corpus_id=body.get("corpus_id"),
    )
    return await pipeline.run(request)


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
        threshold = cache_cfg.get("similarity_threshold", 0.92)
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
        async for doc in db[CACHE_ENTRIES].aggregate(pipeline):
            await db[CACHE_ENTRIES].update_one({"_id": doc["_id"]}, {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}})
            return {"hit": True, "response": doc["response"], "similarity_score": doc["_score"]}
    return {"hit": False, "response": None, "similarity_score": 0.0}


@app.post("/cache/store")
async def cache_store(body: dict):
    db = get_async_db()
    config = await load_config(db)
    cache_cfg = config.get("modules", {}).get("semantic_cache", {})
    strategy = get_strategy(config.get("strategy"))
    cache_cfg = {**cache_cfg, "cache_key": strategy.cache_key}
    from finops.modules.semantic_cache import SemanticCache
    cache = SemanticCache(db, cache_cfg)
    await cache.store(
        prompt=body.get("prompt", ""),
        response=body.get("response", ""),
        framework=body.get("framework", "unknown"),
        model=body.get("model", ""),
        tokens_saved=int(body.get("tokens_saved", 0)),
        agent_id=body.get("agent_id", ""),
        corpus_id=body.get("corpus_id", ""),
    )
    return {"stored": True}


@app.post("/memory/retrieve")
async def memory_retrieve(body: dict):
    agent_id = body.get("agent_id", "default")
    query = body.get("query", "")
    db = get_async_db()
    config = await load_config(db)
    mem_cfg = config.get("modules", {}).get("agent_memory", {})
    from finops.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, mem_cfg)
    working = await memory._get_working_memory(agent_id)
    episodic = await memory._get_episodic_memory(agent_id, query)
    semantic = await memory._get_semantic_memory(agent_id, query)
    return {"working": working, "episodic": episodic, "semantic": semantic}


@app.post("/memory/store")
async def memory_store(body: dict):
    agent_id = body.get("agent_id", "default")
    session_id = body.get("session_id", "default")
    turn = body.get("turn", "")
    response = body.get("response", "")
    db = get_async_db()
    config = await load_config(db)
    mem_cfg = config.get("modules", {}).get("agent_memory", {})
    from finops.modules.agent_memory import AgentMemory
    memory = AgentMemory(db, mem_cfg)
    await memory.store_turn(agent_id, session_id, turn, response)
    return {"stored": True}
