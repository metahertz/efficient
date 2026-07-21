import os
import socket
import subprocess
import sys
import time

import httpx
import pytest
from pymongo import MongoClient

MONGO_URI = os.getenv("FINOPS_TEST_MONGODB_URI", "mongodb://localhost:27018/?directConnection=true")
LIVE_DB = "finops_live_test"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def live_daemon():
    port = _free_port()
    env = {
        **os.environ,
        "FINOPS_MONGODB_URI": MONGO_URI,
        "FINOPS_DB_NAME": LIVE_DB,
    }
    env.pop("FINOPS_API_TOKEN", None)  # keep the harness auth-free
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "finops.daemon.app:app",
         "--host", "127.0.0.1", "--port", str(port)],
        env=env,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120  # lifespan creates search indexes; first run is slow
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"daemon exited early with code {proc.returncode}")
            try:
                if httpx.get(f"{url}/health", timeout=1.0).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > deadline:
                raise TimeoutError("live daemon did not become healthy within 120s")
            time.sleep(0.5)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        MongoClient(MONGO_URI, directConnection=True).drop_database(LIVE_DB)
