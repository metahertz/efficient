import os
import pytest
from efficient.db.client import get_sync_db, get_async_db, reset_clients


def test_sync_db_uses_env_db_name(sync_client, monkeypatch):
    monkeypatch.setenv("EFFICIENT_DB_NAME", "efficient_test")
    reset_clients()
    db = get_sync_db()
    assert db.name == "efficient_test"
    reset_clients()


async def test_async_db_name_matches_env(monkeypatch):
    monkeypatch.setenv("EFFICIENT_DB_NAME", "efficient_test")
    reset_clients()
    db = get_async_db()
    assert db.name == "efficient_test"
    reset_clients()
