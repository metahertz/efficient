from motor.motor_asyncio import AsyncIOMotorDatabase
from finops.db.collections import CACHE_ENTRIES, COMPRESSION_STATS


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

    return {
        "total_tokens_saved": cache_tokens + comp_tokens,
        "cache_hit_rate":     round(cache_hit_rate, 4),
        "compression_ratio":  comp_ratio,
        "per_module":         per_module,
    }
