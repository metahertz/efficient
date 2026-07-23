"""Cache guardian: per-session prompt-cache health analysis for the gateway.

Detects the silent invalidators that make Anthropic's prompt cache go cold on
agentic traffic (tool-set changes, system-prompt churn, oversized turns) and
scores every request's cache utilization. Measurement only — never touches
the wire path.
"""
import hashlib
import json
import threading
from collections import OrderedDict

_MAX_SESSIONS = 50
_COLD_MIN_TOKENS = 2048
_LOOKBACK_BLOCKS = 20

_lock = threading.Lock()
_sessions: "OrderedDict[str, dict]" = OrderedDict()


def _hash_field(value) -> str:
    if value is None:
        return ""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()


def _common_prefix_ratio(prev: bytes, cur: bytes) -> float:
    if not prev:
        return 0.0
    n = min(len(prev), len(cur))
    i = 0
    while i < n and prev[i] == cur[i]:
        i += 1
    return round(i / len(prev), 4)


def _count_blocks(messages) -> int:
    total = 0
    for m in messages or []:
        content = m.get("content")
        total += len(content) if isinstance(content, list) else 1
    return total


def analyze(session_id: str, body: bytes, parsed: dict, usage: dict) -> dict:
    """Return per-request cache-health fields + at most one alertable finding.

    usage keys: input_tokens, cache_read_input_tokens, cache_creation_input_tokens.
    """
    tools_hash = _hash_field(parsed.get("tools"))
    system_hash = _hash_field(parsed.get("system"))
    blocks = _count_blocks(parsed.get("messages"))
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
    input_tokens = usage.get("input_tokens", 0) or 0
    total = input_tokens + cache_read + cache_creation
    ratio = round(cache_read / total, 4) if total else 0.0

    with _lock:
        prev = _sessions.pop(session_id, None)
        _sessions[session_id] = {
            "body": body,
            "tools_hash": tools_hash,
            "system_hash": system_hash,
            "blocks": blocks,
            "cache_read": cache_read,
        }
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)

    invalidator = None
    message = None
    prefix_overlap = 0.0
    if prev is not None:
        prefix_overlap = _common_prefix_ratio(prev["body"], body)
        blocks_delta = blocks - prev["blocks"]
        if tools_hash != prev["tools_hash"]:
            invalidator = "tools_changed"
            message = "prompt cache invalidated: tool set changed"
        elif system_hash != prev["system_hash"]:
            invalidator = "system_changed"
            message = "prompt cache invalidated: system prompt changed"
        elif prev["cache_read"] > 0 and cache_read == 0 and total > _COLD_MIN_TOKENS:
            invalidator = "cache_cold"
            message = f"prompt cache went cold ({total} tokens re-written)"
        elif blocks_delta > _LOOKBACK_BLOCKS:
            invalidator = "lookback_risk"
            message = (f"turn added {blocks_delta} content blocks — may exceed "
                       "Anthropic's cache lookback window")

    return {
        "session_id": session_id,
        "cache_read_ratio": ratio,
        "prefix_overlap": prefix_overlap,
        "content_blocks": blocks,
        "invalidator": invalidator,
        "alert": message,
    }


def reset() -> None:
    """Clear session state. Tests only."""
    with _lock:
        _sessions.clear()
