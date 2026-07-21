import pytest
from pymongo.errors import DuplicateKeyError
from efficient.db.indexes import create_all_indexes
from efficient.db.collections import CACHE_ENTRIES, EPISODIC_MEMORY, CORPUS_CHUNKS


def test_create_indexes_is_idempotent(sync_db):
    create_all_indexes(sync_db)
    create_all_indexes(sync_db)  # must not raise


def test_unique_index_on_cache_prompt_hash(sync_db):
    create_all_indexes(sync_db)
    col = sync_db[CACHE_ENTRIES]
    col.insert_one({"prompt_hash": "abc123"})
    with pytest.raises(DuplicateKeyError):
        col.insert_one({"prompt_hash": "abc123"})


def test_ttl_index_on_cache_expires_at(sync_db):
    create_all_indexes(sync_db)
    indexes = {i["name"]: i for i in sync_db[CACHE_ENTRIES].list_indexes()}
    assert "expires_at_1" in indexes
    assert indexes["expires_at_1"].get("expireAfterSeconds") == 0


def test_text_index_on_corpus_chunks(sync_db):
    create_all_indexes(sync_db)
    indexes = {i["name"]: i for i in sync_db[CORPUS_CHUNKS].list_indexes()}
    assert any("text" in name for name in indexes)
