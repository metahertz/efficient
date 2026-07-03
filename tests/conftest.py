import os
import time
import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from finops.db.client import reset_clients

MONGO_URI = os.getenv("FINOPS_TEST_MONGODB_URI", "mongodb://localhost:27017/?directConnection=true")
TEST_DB   = "finops_test"


@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    os.environ["FINOPS_MONGODB_URI"] = MONGO_URI
    os.environ["FINOPS_DB_NAME"]     = TEST_DB


@pytest.fixture(scope="session")
def sync_client():
    client = MongoClient(MONGO_URI, directConnection=True, serverSelectionTimeoutMS=5000)
    yield client
    client.drop_database(TEST_DB)
    client.close()


@pytest.fixture
def sync_db(sync_client):
    db = sync_client[TEST_DB]
    yield db
    for name in db.list_collection_names():
        db[name].drop()


@pytest.fixture
async def async_client():
    reset_clients()
    os.environ["FINOPS_DB_NAME"] = TEST_DB
    client = AsyncIOMotorClient(MONGO_URI)
    yield client
    await client.drop_database(TEST_DB)
    client.close()
    reset_clients()


@pytest.fixture
async def finops_db(async_client, sync_db):
    from finops.db.indexes import create_all_indexes
    create_all_indexes(sync_db)
    yield async_client[os.environ["FINOPS_DB_NAME"]]


def wait_for_queryable(collection, index_name, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for idx in collection.list_search_indexes():
            if idx["name"] == index_name and idx.get("queryable") is True:
                return
        time.sleep(1)
    raise TimeoutError(f"index {index_name} not queryable within {timeout}s")
