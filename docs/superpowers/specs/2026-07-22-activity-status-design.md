# Activity/Status Feed — Design

**Date:** 2026-07-22
**Status:** Approved

## Goal

Make daemon activity visible: model downloads, codebase-graph builds, memory/cache events — as a full feed on the dashboard and as milestone notifications in the Claude Code monitor console.

## Components

1. **`efficient/activity.py`** — in-process, thread-safe registry (daemon is single-process):
   - `emit(message, level="info", notify=False, kind="event") -> seq` appends to a ring buffer (maxlen 200) with monotonic `seq` + UTC timestamp.
   - `activity(message, notify=False)` context manager — tracks in-flight ops with elapsed time; emits `…`/`done (Ns)`/`failed` events.
   - `note_indexed(file_path)` — coalesces per-file `/codebase/index-file` calls into one batch: first call emits "indexing codebase…" (notify); `snapshot()` lazily closes the batch after ~20s idle, emitting "indexed N files" (notify). While open, the batch appears as an in-flight activity with count + last file.
   - `snapshot(since=0) -> {"active": [...], "events": [...], "last_seq": N}`.
   - `reset()` for tests.

2. **Daemon** — `GET /status?since=<seq>` returns `snapshot(since)`; `/status` added to auth-exempt prefixes (read-only, like `/health` and `/metrics`).

3. **Instrumentation** (approved list):
   - embeddings `_get_model` and compressor `_get_compressor`: `activity("loading embedding model (voyage-4-nano)", notify=True)` / `activity("loading compressor model (LLMLingua-2)", notify=True)` around construction (inside the init lock).
   - `/codebase/index-file`: `note_indexed(file_path)`.
   - `/codebase/index`: `activity(f"indexing {repo_id} from {path}")` around the loop, summary event with file/symbol counts.
   - `/memory/store` + `/complete` memory write: `emit("stored memory turn (agent=…)")`, notify=False.
   - semantic cache hits: `emit("cache hit (exact|semantic sim=…)")`, notify=False.

4. **Dashboard** — Activity panel: poll `/status?since=<last>` every 2s; render in-flight activities with elapsed seconds and a scrolling log of recent events (newest first, capped ~50 rows client-side).

5. **Monitor** — `daemon-monitor.sh` polls `/status?since=<last_seq>` every 5s (jq to filter `.events[] | select(.notify)`), printing only notify-flagged messages. Health-transition logic unchanged. Requires jq (hooks already do).

## Noise policy

- notify=true (reaches Claude): model load start/done, index batch start/summary, `/codebase/index` runs, failures.
- notify=false (dashboard only): per-batch progress, cache hits, memory stores.

## Non-goals

- No download percentages (HF gives no cheap progress callbacks) — start/done with elapsed time only.
- No persistence: the feed is in-memory; restart clears it (dashboard shows fresh feed).
- Docker build progress: pre-daemon, already covered by the monitor's own lines.

## Testing

- `tests/daemon/test_status.py`: emit/snapshot/since semantics, batch coalescing (injectable idle threshold), endpoint response shape, `/status` reachable with `EFFICIENT_API_TOKEN` set and no header.
- Existing suites must stay green (instrumented paths run under tests).
