# Gateway Mode (v1, read-mostly) — Design

**Date:** 2026-07-23
**Status:** Approved (user: "build a gateway mode for testing and improvement — emphasis on easy UX")

## Goal

Put the daemon on Claude Code's model-call path via the documented LLM Gateway Protocol (`ANTHROPIC_BASE_URL`), so the traffic modules finally see real traffic. **v1 never mutates requests or responses** — it forwards verbatim, streams SSE unbuffered, and measures: token usage, Anthropic prompt-cache utilization, duplicate requests (exact-cache potential), request sizes (compression potential). Serving cached/compressed traffic is v2, informed by v1's measurements.

## UX

- **`efficient claude [args…]`** — new CLI subcommand: health-checks the daemon, sets `ANTHROPIC_BASE_URL=<daemon url>`, and `exec`s `claude` with all args passed through. One command, zero config.
- Dashboard gains a **Gateway** panel: requests proxied, input/output tokens, prompt-cache read/created tokens, duplicate requests; when empty it shows the `efficient claude` hint.
- Activity feed: one notify event on first gateway request of a session; per-request events dashboard-only.

## Mechanics

- `efficient/daemon/gateway.py`, APIRouter mounted on the app: `api_route("/v1/{path:path}", methods=["GET","POST"])` — generic passthrough (covers `/v1/messages`, `count_tokens`, `models`).
- Forward all headers except hop-by-hop (`host`, `content-length`, `connection`, `transfer-encoding`, `accept-encoding`, …) — this preserves `authorization`/`x-api-key`, `anthropic-version`, `anthropic-beta`, custom headers verbatim. Body forwarded byte-for-byte (no JSON reparse on the wire path).
- Upstream: `EFFICIENT_GATEWAY_UPSTREAM` (default `https://api.anthropic.com`). Shared `httpx.AsyncClient` with `read=None` timeout for long streams.
- Response: `StreamingResponse` yielding `aiter_raw()` chunks — never buffered. Status + non-hop headers mirrored.
- **Auth exemption**: `/v1` added to the daemon's bearer-exempt prefixes — the `Authorization` header on gateway traffic belongs to Anthropic, not to us. (Gateway endpoints are still useless without valid Anthropic credentials.)

## Measurement (per request, fire-and-forget after stream close)

New collection `gateway_log` (TTL 30 days): `created_at`, `path`, `model`, `stream`, `status`, `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `latency_ms`, `request_bytes`, `body_hash` (sha256 — exact-duplicate detection).

Usage extraction: non-stream → response JSON `.usage`; stream → tee only SSE `data:` lines containing `"usage"` (bounded), parse `message_start` (input/cache fields) and `message_delta` (output_tokens). Parse failures degrade to zeros — never break the wire path.

`/metrics` gains `gateway`: `{requests, input_tokens, output_tokens, cache_read_tokens, cache_creation_tokens, duplicate_requests}` (duplicates = requests sharing a `body_hash` beyond the first).

## Non-goals (v1)

- No cache serving, no compression, no request rewriting of any kind.
- No `count_tokens` emulation (passthrough only; Claude Code estimates locally if upstream lacks it).
- No model discovery config (`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY` passthrough works via the generic route).

## Testing

`tests/daemon/test_gateway.py` with a fake upstream ASGI app (SSE + JSON endpoints), gateway client patched to `ASGITransport`: stream passthrough byte-integrity, header forwarding (including auth + anthropic-beta), usage recording (stream and non-stream), duplicate detection, `/v1` bearer-auth exemption. CLI: `efficient claude` env injection with `execvpe` mocked. Layer C-style live test against real Anthropic is manual (documented, needs API key).

## Risks

- Streaming stall if any middleware buffers — verified by byte-integrity test and a live `efficient claude` smoke.
- The gateway must not reorder the `system` array or strip beta headers — guaranteed by never parsing/rebuilding the request on the wire path.
