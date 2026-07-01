from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from finops.db.collections import CONFIG

DEFAULT_CONFIG: dict = {
    "_id": "global",
    "modules": {
        "codebase_graph":    {"enabled": True,  "repo_paths": []},
        "semantic_cache":    {"enabled": True,  "similarity_threshold": 0.92, "ttl_hours": 168},
        "agent_memory":      {"enabled": True,  "working_memory_turns": 20,
                              "episodic_ttl_days": 30, "semantic_ttl_days": 90},
        "context_compressor":{"enabled": True,  "token_threshold": 8000, "target_ratio": 4.0},
        "hybrid_retrieval":  {"enabled": False, "top_k": 5, "rrf_k": 60},
        "benchmark_runner":  {"enabled": True,  "judge_model": "claude-sonnet-4-6"},
    },
    "embedding_model":       "voyage-4-nano",
    "embedding_dimensions":  1024,
    "cost_per_input_token":  0.000003,
    "cost_per_output_token": 0.000015,
    "updated_at":            None,
}


async def load_config(db: AsyncIOMotorDatabase) -> dict:
    doc = await db[CONFIG].find_one({"_id": "global"})
    if doc is None:
        initial = {**DEFAULT_CONFIG, "updated_at": datetime.now(timezone.utc)}
        await db[CONFIG].insert_one(initial)
        return dict(initial)
    return dict(doc)


async def save_config(db: AsyncIOMotorDatabase, patch: dict) -> dict:
    await db[CONFIG].update_one(
        {"_id": "global"},
        {"$set": {**patch, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return await load_config(db)
