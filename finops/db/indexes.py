from pymongo import ASCENDING, TEXT
from pymongo.database import Database
from finops.db.collections import (
    CODEBASE_NODES, CACHE_ENTRIES, WORKING_MEMORY,
    EPISODIC_MEMORY, SEMANTIC_MEMORY, CORPUS_CHUNKS,
    COMPRESSION_STATS, BENCHMARK_RUNS,
)

EMBEDDING_DIMENSIONS = 1024
VECTOR_SIMILARITY    = "cosine"


def _search_index_exists(collection, name: str) -> bool:
    return any(idx["name"] == name for idx in collection.list_search_indexes())


def _create_vector_index(collection, name: str, field: str = "embedding") -> None:
    if _search_index_exists(collection, name):
        return
    collection.create_search_index({
        "name": name,
        "type": "vectorSearch",
        "definition": {
            "fields": [{
                "type": "vector",
                "path": field,
                "numDimensions": EMBEDDING_DIMENSIONS,
                "similarity": VECTOR_SIMILARITY,
            }]
        },
    })


def create_all_indexes(db: Database) -> None:
    # codebase_nodes
    col = db[CODEBASE_NODES]
    col.create_index([("repo_id", ASCENDING), ("symbol", ASCENDING)])
    col.create_index([("repo_id", ASCENDING), ("file_path", ASCENDING)])
    _create_vector_index(col, "codebase_vector_index")

    # cache_entries
    col = db[CACHE_ENTRIES]
    col.create_index([("prompt_hash", ASCENDING)], unique=True)
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "cache_vector_index")

    # working_memory
    col = db[WORKING_MEMORY]
    col.create_index([("agent_id", ASCENDING), ("session_id", ASCENDING)])

    # episodic_memory
    col = db[EPISODIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "episodic_vector_index")

    # semantic_memory
    col = db[SEMANTIC_MEMORY]
    col.create_index([("agent_id", ASCENDING)])
    col.create_index("expires_at", expireAfterSeconds=0)
    _create_vector_index(col, "semantic_vector_index")

    # compression_stats
    col = db[COMPRESSION_STATS]
    col.create_index([("created_at", ASCENDING)])

    # corpus_chunks
    col = db[CORPUS_CHUNKS]
    col.create_index([("corpus_id", ASCENDING)])
    col.create_index([("bm25_tokens", TEXT)])
    _create_vector_index(col, "corpus_vector_index")

    # benchmark_runs
    col = db[BENCHMARK_RUNS]
    col.create_index([("started_at", ASCENDING)])
