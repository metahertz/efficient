# Cache Guardian (gateway v2) — Design

**Date:** 2026-07-23
**Status:** Approved

## Goal

Make the gateway the guardian of Anthropic's prompt cache for agentic traffic: measure per-session cache health, detect silent invalidators in-band, and alert when the cache goes cold — instead of adding a second (evidence-rejected) response cache. Pure measurement/diagnostics; the wire path stays verbatim.

## Detection model

Sessions are keyed by the `x-claude-code-session-id` request header (per the LLM Gateway Protocol); absent that, a stable fallback key (`"default"`). Per-session state (in-process LRU, cap 50 sessions): previous request body, tools hash, system hash, previous usage.

Per request, `efficient/daemon/cache_guardian.py` computes:

- `cache_read_ratio` = cache_read / (input + cache_read + cache_creation) — the health number.
- `prefix_overlap` = common-prefix bytes with the session's previous body / previous body length.
- `tools_hash`, `system_hash` (sha256 of the JSON-serialized `tools` / `system` fields).
- `content_blocks` = total content blocks across messages; `blocks_delta` vs previous turn.

Findings (emitted as activity events, notify=true, and stored on the gateway_log doc as `invalidator`):

| Finding | Trigger | Message |
|---|---|---|
| tools_changed | tools hash differs from previous turn | "prompt cache invalidated: tool set changed" |
| system_changed | system hash differs | "prompt cache invalidated: system prompt changed" |
| cache_cold | previous turn had cache_read>0, this turn cache_read==0 with total prompt > 2048 tokens | "prompt cache went cold (N tokens re-written)" |
| lookback_risk | blocks_delta > 20 | "turn added >20 content blocks — may exceed Anthropic's cache lookback window" |

Only one finding per request is alerted (priority: tools > system > cold > lookback); all computed fields are stored regardless. `prefix_overlap` is stored as a contributing signal, not alerted.

## Surfacing

- gateway_log docs gain: `session_id`, `cache_read_ratio`, `prefix_overlap`, `content_blocks`, `invalidator` (nullable string).
- `/metrics.gateway` gains: `cache_read_ratio` (token-weighted overall), `invalidations` (count of docs with invalidator), `sessions` (distinct session_ids).
- Dashboard Gateway panel gains rows: prompt-cache read ratio, invalidations detected, sessions.
- Monitor: invalidator events arrive via the existing notify relay.

## Non-goals

- No request mutation (no breakpoint injection yet — data first).
- No cross-restart session state (in-process only; gateway_log persists the per-request fields).

## Testing

Unit: guardian state machine (tools/system change, cold detection, prefix overlap, LRU cap). Integration: fake-upstream gateway tests asserting doc fields + `/metrics` additions + activity events.
