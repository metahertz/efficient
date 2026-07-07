from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from finops.db.collections import CACHE_ENTRIES, COMPRESSION_STATS, REQUEST_LOG

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
    }
