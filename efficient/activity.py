"""In-process activity feed: recent events + in-flight operations.

The daemon is a single process, so a module-level registry is sufficient.
Exposed over GET /status for the dashboard (full feed) and the Claude Code
plugin monitor (notify-flagged milestones only).
"""
import itertools
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone

_lock = threading.Lock()
_counter = itertools.count(1)
_events: deque = deque(maxlen=200)
_active: dict[int, dict] = {}

# Per-file index calls are coalesced into one batch so a 500-file index
# produces two events (start + summary), not 500.
_BATCH_IDLE_S = 20.0
_batch = {"open": False, "count": 0, "last_file": "", "last_ts": 0.0}


def emit(message: str, *, level: str = "info", notify: bool = False,
         kind: str = "event") -> int:
    with _lock:
        return _emit_locked(message, level=level, notify=notify, kind=kind)


def _emit_locked(message: str, *, level: str = "info", notify: bool = False,
                 kind: str = "event") -> int:
    seq = next(_counter)
    _events.append({
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "level": level,
        "notify": notify,
        "kind": kind,
    })
    return seq


@contextmanager
def activity(message: str, *, notify: bool = False):
    t0 = time.time()
    with _lock:
        aid = next(_counter)
        _active[aid] = {"id": aid, "message": message, "started_at": t0}
        _emit_locked(f"{message}…", notify=notify, kind="start")
    try:
        yield
        emit(f"{message} done ({time.time() - t0:.0f}s)", notify=notify, kind="done")
    except Exception:
        emit(f"{message} failed", level="error", notify=True, kind="error")
        raise
    finally:
        with _lock:
            _active.pop(aid, None)


def note_indexed(file_path: str) -> None:
    with _lock:
        if not _batch["open"]:
            _batch.update(open=True, count=0, last_file="", last_ts=0.0)
            _emit_locked("indexing codebase…", notify=True, kind="start")
        _batch["count"] += 1
        _batch["last_file"] = file_path
        _batch["last_ts"] = time.time()


def _close_idle_batch_locked() -> None:
    if _batch["open"] and time.time() - _batch["last_ts"] > _BATCH_IDLE_S:
        _emit_locked(f"indexed {_batch['count']} files", notify=True, kind="done")
        _batch["open"] = False


def snapshot(since: int = 0) -> dict:
    now = time.time()
    with _lock:
        _close_idle_batch_locked()
        events = [e for e in _events if e["seq"] > since]
        active = [
            {**a, "elapsed_s": round(now - a["started_at"], 1)}
            for a in _active.values()
        ]
        if _batch["open"]:
            active.append({
                "id": 0,
                "message": f"indexing codebase ({_batch['count']} files, last: {_batch['last_file']})",
                "elapsed_s": round(now - _batch["last_ts"], 1),
            })
        last_seq = _events[-1]["seq"] if _events else 0
    return {"active": active, "events": events, "last_seq": last_seq}


def reset() -> None:
    """Clear all state. Tests only."""
    global _counter
    with _lock:
        _counter = itertools.count(1)
        _events.clear()
        _active.clear()
        _batch.update(open=False, count=0, last_file="", last_ts=0.0)
