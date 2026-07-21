from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
from finops.db.collections import CONFIG
from finops.db.indexes import EMBEDDING_DIMENSIONS

DEFAULT_CONFIG: dict = {
    "_id": "global",
    "modules": {
        "codebase_graph":    {"enabled": True,  "repo_paths": []},
        "semantic_cache":    {"enabled": True,  "similarity_threshold": 0.80, "ttl_hours": 168},
        "agent_memory":      {"enabled": True,  "working_memory_turns": 20,
                              "episodic_ttl_days": 30, "semantic_ttl_days": 90,
                              "fact_extraction": {"provider": "anthropic", "base_url": "", "model": ""}},
        "context_compressor":{"enabled": True,  "token_threshold": 8000, "target_ratio": 4.0},
        "hybrid_retrieval":  {"enabled": False, "top_k": 5, "rrf_k": 60},
    },
    "strategy":              "compose_then_compress",
    "embedding_model":       "voyage-4-nano",
    "embedding_dimensions":  EMBEDDING_DIMENSIONS,
    "cost_per_input_token":  0.000003,
    "cost_per_output_token": 0.000015,
    "updated_at":            None,
}


_PATCHABLE_KEYS = {
    "modules", "strategy", "embedding_model", "embedding_dimensions",
    "cost_per_input_token", "cost_per_output_token",
}


def _check_keys(obj: dict) -> None:
    for k, v in obj.items():
        if not isinstance(k, str) or k.startswith("$") or "." in k:
            raise ValueError(f"invalid config key: {k!r}")
        if isinstance(v, dict):
            _check_keys(v)


def validate_patch(patch: dict) -> None:
    unknown = set(patch) - _PATCHABLE_KEYS
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    _check_keys(patch)


async def load_config(db: AsyncIOMotorDatabase) -> dict:
    doc = await db[CONFIG].find_one({"_id": "global"})
    if doc is None:
        initial = {**DEFAULT_CONFIG, "updated_at": datetime.now(timezone.utc)}
        await db[CONFIG].insert_one(initial)
        return dict(initial)
    return dict(doc)


def _flatten(obj: dict, prefix: str = "") -> dict:
    """Flatten nested dict to dot-notation keys for MongoDB $set."""
    result = {}
    for k, v in obj.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten(v, full_key))
        else:
            result[full_key] = v
    return result


async def save_config(db: AsyncIOMotorDatabase, patch: dict) -> dict:
    # Ensure the document is seeded with defaults before patching.
    await load_config(db)
    flat = _flatten(patch)
    flat["updated_at"] = datetime.now(timezone.utc)
    await db[CONFIG].update_one(
        {"_id": "global"},
        {"$set": flat},
    )
    return await load_config(db)
