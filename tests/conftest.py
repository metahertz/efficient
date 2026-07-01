import os
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
