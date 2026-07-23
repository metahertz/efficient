"""Client/capability registry: tracks which clients are active and which
capabilities each exercises, for the dashboard's Connected Clients view.
In-process, last-seen based; a client is 'active' if seen within the window.
"""
import threading
import time

_ACTIVE_WINDOW_S = 900  # 15 minutes

_lock = threading.Lock()
# client -> {capability -> last_seen_epoch}
_clients: dict[str, dict[str, float]] = {}

# path prefix -> capability label (first match wins; order matters)
_CAPABILITY_BY_PATH = [
    ("/v1", "gateway"),
    ("/codebase", "codebase_graph"),
    ("/memory/tool", "memory_tool"),
    ("/memory", "agent_memory"),
    ("/optimize", "optimize_pipeline"),
    ("/complete", "complete_proxy"),
    ("/cache", "semantic_cache"),
]

_IGNORE = ("/health", "/metrics", "/status", "/dashboard")


def capability_for(path: str) -> str | None:
    if any(path == p or path.startswith(p) for p in _IGNORE):
        return None
    for prefix, cap in _CAPABILITY_BY_PATH:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix):
            return cap
    return None


def resolve_client(path: str, headers) -> str:
    explicit = headers.get("x-efficient-client")
    if explicit:
        return explicit
    if path.startswith("/v1"):
        return "claude-code" if headers.get("x-claude-code-session-id") else "api-client"
    return "unknown"


def note(client: str, capability: str) -> None:
    with _lock:
        _clients.setdefault(client, {})[capability] = time.time()


def snapshot(window_s: int = _ACTIVE_WINDOW_S) -> list[dict]:
    now = time.time()
    out = []
    with _lock:
        for client, caps in _clients.items():
            active_caps = sorted(c for c, ts in caps.items() if now - ts <= window_s)
            if not active_caps:
                continue
            last_seen = max(caps.values())
            out.append({
                "client": client,
                "capabilities": active_caps,
                "last_seen_s": round(now - last_seen, 1),
            })
    out.sort(key=lambda c: c["last_seen_s"])
    return out


def reset() -> None:
    with _lock:
        _clients.clear()
