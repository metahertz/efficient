"""Host-side directory watcher: ingests text files from configured directories
into retrieval corpora over the daemon HTTP API (mount-free, like the codebase
and memory hooks). Config: ~/.efficient/watch.json.
"""
import json
import os
from pathlib import Path

import httpx

from efficient.ingest.chunker import chunk_text

_DEFAULT_EXTS = [".md", ".txt", ".rst", ".mdx", ".markdown"]
_MAX_BYTES = 1_000_000
CONFIG_PATH = Path.home() / ".efficient" / "watch.json"


def _daemon_url() -> str:
    return os.getenv("EFFICIENT_DAEMON_URL", "http://localhost:7432").rstrip("/")


def _headers() -> dict:
    token = os.getenv("EFFICIENT_API_TOKEN", "")
    h = {"X-Efficient-Client": "watcher"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def load_watches(path: Path = CONFIG_PATH) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    watches = []
    for w in data.get("watches", []):
        root = Path(os.path.expanduser(w["path"])).resolve()
        watches.append({
            "root": root,
            "corpus_id": w.get("corpus_id", root.name),
            "extensions": w.get("extensions", _DEFAULT_EXTS),
        })
    return watches


def _iter_files(root: Path, extensions: list[str]):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions and "/." not in str(p):
            yield p


def ingest_file(client: httpx.Client, root: Path, corpus_id: str, file: Path) -> int:
    try:
        text = file.read_text(encoding="utf-8")[:_MAX_BYTES]
    except (OSError, UnicodeDecodeError):
        return 0
    rel = str(file.relative_to(root))
    chunks = chunk_text(text)
    if not chunks:
        remove_file(client, corpus_id, rel)
        return 0
    payload = {"corpus_id": corpus_id, "chunks": [
        {"text": c, "source_file": rel, "chunk_index": i}
        for i, c in enumerate(chunks)]}
    r = client.post(f"{_daemon_url()}/corpus/add-chunks", json=payload,
                    headers=_headers(), timeout=120.0)
    r.raise_for_status()
    # drop any stale trailing chunks from a previous, longer version
    remove_file(client, corpus_id, rel, from_index=len(chunks))
    return len(chunks)


def remove_file(client: httpx.Client, corpus_id: str, rel: str,
                from_index: int | None = None) -> None:
    # from_index unused server-side in v1 (full remove); kept for API symmetry.
    if from_index is not None:
        return
    client.post(f"{_daemon_url()}/corpus/remove-file",
                json={"corpus_id": corpus_id, "source_file": rel},
                headers=_headers(), timeout=30.0)


def sync_once(watches: list[dict] | None = None) -> dict:
    watches = watches if watches is not None else load_watches()
    summary = {}
    with httpx.Client() as client:
        for w in watches:
            root, corpus_id = w["root"], w["corpus_id"]
            if not root.is_dir():
                continue
            files = added = 0
            for f in _iter_files(root, w["extensions"]):
                n = ingest_file(client, root, corpus_id, f)
                if n:
                    files += 1
                    added += n
            summary[corpus_id] = {"files": files, "chunks": added}
    return summary


async def watch_forever(watches: list[dict] | None = None):
    from watchfiles import awatch, Change
    watches = watches if watches is not None else load_watches()
    roots = {str(w["root"]): w for w in watches if w["root"].is_dir()}
    if not roots:
        return
    sync_once(list(roots.values()))
    with httpx.Client() as client:
        async for changes in awatch(*roots):
            for change, path in changes:
                p = Path(path)
                w = next((v for k, v in roots.items() if str(p).startswith(k)), None)
                if w is None or p.suffix.lower() not in w["extensions"]:
                    continue
                rel = str(p.relative_to(w["root"]))
                if change == Change.deleted:
                    remove_file(client, w["corpus_id"], rel)
                else:
                    ingest_file(client, w["root"], w["corpus_id"], p)
