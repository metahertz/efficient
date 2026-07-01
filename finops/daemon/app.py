from contextlib import asynccontextmanager
from fastapi import FastAPI
from finops.db.client import get_async_db, get_sync_db
from finops.db.indexes import create_all_indexes
from finops.daemon.config import load_config, save_config

VERSION = "0.1.0"


def _check_prerequisites(sync_db) -> None:
    info = sync_db.command("buildInfo")
    major = int(info["version"].split(".")[0])
    if major < 7:
        raise SystemExit(
            f"ERROR: MongoDB >= 7.0 required, found {info['version']}.\n"
            "Run: docker run -p 27017:27017 mongodb/mongodb-atlas-local:latest"
        )
    try:
        list(sync_db["config"].list_search_indexes())
    except Exception as exc:
        raise SystemExit(
            "ERROR: MongoDB Atlas Search (mongot) not available.\n"
            "Run: docker run -p 27017:27017 mongodb/mongodb-atlas-local:latest\n"
            f"Detail: {exc}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    sync_db = get_sync_db()
    _check_prerequisites(sync_db)
    create_all_indexes(sync_db)
    db = get_async_db()
    await load_config(db)
    yield


app = FastAPI(title="fullFinOps-AI Daemon", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "version": VERSION}


@app.get("/config")
async def get_config():
    db = get_async_db()
    config = await load_config(db)
    config.pop("_id", None)
    return config


@app.put("/config")
async def put_config(patch: dict):
    db = get_async_db()
    config = await save_config(db, patch)
    config.pop("_id", None)
    return config
