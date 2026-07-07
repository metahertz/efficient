from pymongo import ASCENDING, TEXT
from pymongo.database import Database
from finops.db.collections import (
    CODEBASE_NODES, CACHE_ENTRIES, WORKING_MEMORY,
    EPISODIC_MEMORY, SEMANTIC_MEMORY, CORPUS_CHUNKS,
    COMPRESSION_STATS, BENCHMARK_RUNS,
)

EMBEDDING_DIMENSIONS = 1024
VECTOR_SIMILARITY    = "cosine"


_ALL_COLLECTIONS = (
    CODEBASE_NODES, CACHE_ENTRIES, WORKING_MEMORY, EPISODIC_MEMORY,
    SEMANTIC_MEMORY, COMPRESSION_STATS, CORPUS_CHUNKS, BENCHMARK_RUNS,
)


def _ensure_collections(db: Database) -> None:
    existing = set(db.list_collection_names())
    for name in _ALL_COLLECTIONS:
        if name not in existing:
            db.create_collection(name)


def _search_index_exists(collection, name: str) -> bool:
    return any(idx["name"] == name for idx in collection.list_search_indexes())


def _create_vector_index(
    collection, name: str, field: str = "embedding",
    filter_paths: list[str] | None = None,
) -> None:
    if _search_index_exists(collection, name):
        return
    fields = [{
        "type": "vector",
        "path": field,
        "numDimensions": EMBEDDING_DIMENSIONS,
        "similarity": VECTOR_SIMILARITY,
    }]
    for path in (filter_paths or []):
        fields.append({"type": "filter", "path": path})
    collection.create_search_index({
        "name": name,
        "type": "vectorSearch",
        "definition": {"fields": fields},
    })


def create_all_indexes(db: Database) -> None:
    _ensure_collections(db)

    col = db[CODEBASE_NODES]
    col.create_index([("repo_id", ASCENDING), ("symbol", ASCENDING)])
    col.create_index([("repo_id", ASCENDING), ("file_path", ASCENDING)])
    _create_vector_index(col, "codebase_vector_index", filter_paths=["repo_id"])

    col = db[CACHE_ENTRIES]
    col.create_index([("prompt_hash", ASCENDING)], unique=True)
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "cache_vector_index")

    col = db[WORKING_MEMORY]
    col.create_index([("agent_id", ASCENDING), ("session_id", ASCENDING)])

    col = db[EPISODIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "episodic_vector_index", filter_paths=["agent_id"])

    col = db[SEMANTIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "semantic_vector_index", filter_paths=["agent_id"])

    col = db[COMPRESSION_STATS]
    col.create_index([("created_at", ASCENDING)])

    col = db[CORPUS_CHUNKS]
    col.create_index([("corpus_id", ASCENDING)])
    col.create_index([("bm25_tokens", TEXT)])
    _create_vector_index(col, "corpus_vector_index", filter_paths=["corpus_id"])

    col = db[BENCHMARK_RUNS]
    col.create_index([("started_at", ASCENDING)])
