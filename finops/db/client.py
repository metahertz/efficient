from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database
import os

_async_client: AsyncIOMotorClient | None = None
_sync_client:  MongoClient | None = None


def get_async_client() -> AsyncIOMotorClient:
    global _async_client
    if _async_client is None:
        _async_client = AsyncIOMotorClient(
            os.getenv("FINOPS_MONGODB_URI", "mongodb://localhost:27017")
        )
    return _async_client


def get_sync_client() -> MongoClient:
    global _sync_client
    if _sync_client is None:
        _sync_client = MongoClient(
            os.getenv("FINOPS_MONGODB_URI", "mongodb://localhost:27017")
        )
    return _sync_client


def get_async_db(client: AsyncIOMotorClient | None = None) -> AsyncIOMotorDatabase:
    if client is None:
        client = get_async_client()
    return client[os.getenv("FINOPS_DB_NAME", "finops")]


def get_sync_db(client: MongoClient | None = None) -> Database:
    if client is None:
        client = get_sync_client()
    return client[os.getenv("FINOPS_DB_NAME", "finops")]


def reset_clients() -> None:
    """Reset singletons. Call in tests only."""
    global _async_client, _sync_client
    _async_client = None
    _sync_client  = None
