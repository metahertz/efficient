from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from efficient.db.collections import (
    CACHE_ENTRIES, CODEBASE_NODES, COMPRESSION_STATS, CORPUS_CHUNKS,
    EPISODIC_MEMORY, GATEWAY_LOG, MEMORY_FILES, REQUEST_LOG, SEMANTIC_MEMORY,
    WORKING_MEMORY,
)

_AUGMENTER_MODULES = ("codebase_graph", "hybrid_retrieval", "agent_memory")


async def record_module_events(db, module_results: list[dict]) -> None:
    if not module_results:
        return
    now = datetime.now(timezone.utc)
    docs = [{
        "module": r.get("module"),
        "tokens_saved": r.get("tokens_saved", 0),
        "tokens_added": r.get("tokens_added", 0),
        "baseline_tokens": r.get("baseline_tokens", 0),
        "latency_ms": r.get("latency_ms", 0.0),
        "created_at": now,
    } for r in module_results]
    await db[REQUEST_LOG].insert_many(docs)


async def aggregate_metrics(db: AsyncIOMotorDatabase) -> dict:
    cache_pipeline = [
        {
            "$group": {
                "_id": None,
                "total_tokens": {"$sum": {"$multiply": ["$tokens_saved", "$hit_count"]}},
                "total_entries": {"$sum": 1},
                "hit_entries": {"$sum": {"$cond": [{"$gt": ["$hit_count", 0]}, 1, 0]}},
            }
        }
    ]
    cache_tokens = 0
    cache_hit_rate = 0.0
    cache_events = 0
    async for doc in db[CACHE_ENTRIES].aggregate(cache_pipeline):
        cache_tokens = doc["total_tokens"]
        total = doc["total_entries"]
        cache_hit_rate = doc["hit_entries"] / total if total > 0 else 0.0
        cache_events = int(doc["hit_entries"])

    comp_pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_ratio": {"$avg": "$ratio"},
                "total_saved": {"$sum": {"$subtract": ["$original_tokens", "$compressed_tokens"]}},
                "events": {"$sum": 1},
            }
        }
    ]
    comp_ratio = 0.0
    comp_tokens = 0
    comp_events = 0
    async for doc in db[COMPRESSION_STATS].aggregate(comp_pipeline):
        comp_ratio = round(doc["avg_ratio"], 2)
        comp_tokens = int(doc["total_saved"])
        comp_events = int(doc["events"])

    per_module = []
    if cache_tokens > 0 or cache_events > 0:
        per_module.append({"module": "semantic_cache", "tokens_saved": cache_tokens, "events": cache_events})
    if comp_tokens > 0 or comp_events > 0:
        per_module.append({"module": "context_compressor", "tokens_saved": comp_tokens, "events": comp_events})

    total_tokens_saved = cache_tokens + comp_tokens

    # Augmenter modules persist per-event data in request_log ONLY, so aggregate
    # them from there. cache/compression are excluded from this pipeline, so no
    # double-counting with the dedicated collections above.
    aug_pipeline = [
        {"$match": {"module": {"$in": list(_AUGMENTER_MODULES)}}},
        {"$group": {"_id": "$module", "tokens_saved": {"$sum": "$tokens_saved"}, "events": {"$sum": 1}}},
    ]
    async for doc in db[REQUEST_LOG].aggregate(aug_pipeline):
        tokens_saved = int(doc["tokens_saved"])
        per_module.append({"module": doc["_id"], "tokens_saved": tokens_saved, "events": int(doc["events"])})
        total_tokens_saved += tokens_saved

    return {
        "total_tokens_saved": total_tokens_saved,
        "cache_hit_rate":     round(cache_hit_rate, 4),
        "compression_ratio":  comp_ratio,
        "per_module":         per_module,
        "store":              await _store_stats(db),
        "gateway":            await _gateway_stats(db),
    }


async def _gateway_stats(db: AsyncIOMotorDatabase) -> dict:
    stats = {"requests": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_creation_tokens": 0,
             "duplicate_requests": 0, "cache_read_ratio": 0.0,
             "invalidations": 0, "sessions": 0}
    pipeline = [{
        "$group": {
            "_id": None,
            "requests": {"$sum": 1},
            "input_tokens": {"$sum": "$input_tokens"},
            "output_tokens": {"$sum": "$output_tokens"},
            "cache_read_tokens": {"$sum": "$cache_read_input_tokens"},
            "cache_creation_tokens": {"$sum": "$cache_creation_input_tokens"},
            "invalidations": {"$sum": {"$cond": [{"$ifNull": ["$invalidator", False]}, 1, 0]}},
            "sessions": {"$addToSet": "$session_id"},
        }
    }]
    async for doc in db[GATEWAY_LOG].aggregate(pipeline):
        stats.update({k: int(doc[k]) for k in
                      ("requests", "input_tokens", "output_tokens",
                       "cache_read_tokens", "cache_creation_tokens",
                       "invalidations")})
        stats["sessions"] = len([s for s in doc["sessions"] if s])
        total_prompt = (stats["input_tokens"] + stats["cache_read_tokens"]
                        + stats["cache_creation_tokens"])
        if total_prompt:
            stats["cache_read_ratio"] = round(stats["cache_read_tokens"] / total_prompt, 4)
    dup_pipeline = [
        {"$match": {"body_hash": {"$ne": ""}}},
        {"$group": {"_id": "$body_hash", "n": {"$sum": 1}}},
        {"$group": {"_id": None, "dups": {"$sum": {"$subtract": ["$n", 1]}}}},
    ]
    async for doc in db[GATEWAY_LOG].aggregate(dup_pipeline):
        stats["duplicate_requests"] = int(doc["dups"])
    return stats


async def _store_stats(db: AsyncIOMotorDatabase) -> dict:
    """What's seeded in the backing store (as opposed to usage savings)."""
    codebase = {"symbols": 0, "files": 0, "repos": 0}
    pipeline = [{
        "$group": {
            "_id": None,
            "symbols": {"$sum": 1},
            "files": {"$addToSet": {"repo": "$repo_id", "file": "$file_path"}},
            "repos": {"$addToSet": "$repo_id"},
        }
    }]
    async for doc in db[CODEBASE_NODES].aggregate(pipeline):
        codebase = {
            "symbols": int(doc["symbols"]),
            "files": len(doc["files"]),
            "repos": len(doc["repos"]),
        }
    return {
        "codebase": codebase,
        "cache_entries": await db[CACHE_ENTRIES].count_documents({}),
        "memory": {
            "working_sessions": await db[WORKING_MEMORY].count_documents({}),
            "episodic": await db[EPISODIC_MEMORY].count_documents({}),
            "semantic_facts": await db[SEMANTIC_MEMORY].count_documents({}),
            "memory_files": await db[MEMORY_FILES].count_documents({}),
        },
        "corpus_chunks": await db[CORPUS_CHUNKS].count_documents({}),
    }
