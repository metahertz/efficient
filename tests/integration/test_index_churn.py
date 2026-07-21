import pytest

from finops.db.collections import CACHE_ENTRIES
from finops.db.indexes import create_all_indexes
from tests.conftest import wait_for_queryable

pytestmark = pytest.mark.integration


def test_search_index_recreated_after_collection_drop(sync_db):
    """Regression: mongot briefly lists a dropped collection's search indexes;
    create_all_indexes must not skip creation based on that stale listing."""
    create_all_indexes(sync_db)
    wait_for_queryable(sync_db[CACHE_ENTRIES], "cache_vector_index")

    for name in sync_db.list_collection_names():
        sync_db[name].drop()

    create_all_indexes(sync_db)
    wait_for_queryable(sync_db[CACHE_ENTRIES], "cache_vector_index")
