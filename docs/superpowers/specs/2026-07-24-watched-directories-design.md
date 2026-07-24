# Watched Directories → RAG Corpus — Design

**Date:** 2026-07-24
**Status:** Approved

## Goal

Let users drop notes/external docs into configured host directories and have efficient auto-ingest them into a retrieval corpus (the `hybrid_retrieval` RAG store), keeping the corpus in sync as files are added/changed/removed.

## Approach

Host-side watcher (mirrors the mount-free codebase/memory hooks — sends file *contents* over HTTP, no container bind mounts). New `efficient watch` CLI command:

- **Config** `~/.efficient/watch.json`: `{"watches": [{"path": "~/notes", "corpus_id": "notes", "extensions": [".md",".txt",".rst",".mdx"]}]}`. `path` tilde-expanded; `extensions` optional (defaults to the text set).
- **Chunking** `efficient/ingest/chunker.py` `chunk_text(text, target_chars=1200) -> list[str]`: split on blank lines into paragraphs, greedily pack into ~target_chars chunks, never splitting mid-paragraph unless a single paragraph exceeds target (then hard-split). Deterministic.
- **Ingest** per file: read (cap 1 MB), chunk, `POST /corpus/add-chunks` with `corpus_id`, `source_file`=path relative to the watch root, one chunk per `chunk_index`. `add_chunks`' upsert key `(corpus_id, source_file, chunk_index)` means re-ingest replaces cleanly; if a re-ingested file has fewer chunks than before, stale trailing chunks are deleted (`DELETE /corpus/file`).
- **Modes**: `efficient watch --once` (one sync pass over all configured dirs, then exit — for cron/manual) and `efficient watch` (continuous via `watchfiles.awatch`, already a dependency; on add/modify → ingest, on delete → remove the file's chunks).
- **Deletion**: `POST /corpus/remove-file {corpus_id, source_file}` → delete chunks for that file.

## Endpoints (daemon)

- `POST /corpus/remove-file {corpus_id, source_file}` → `{corpus_id, source_file, removed}`.
- (reuse existing `POST /corpus/add-chunks`.)

## Non-goals (v1)

- No PDF/binary extraction (text formats only).
- No plugin-monitor auto-start of the watcher (a later convenience); v1 is the explicit `efficient watch` command.
- Watcher runs on the host; it is not the daemon's responsibility.

## Testing

- `chunker`: paragraph packing, oversized-paragraph hard split, empty input, determinism.
- `remove-file` endpoint: deletes only the named file's chunks.
- watch sync (`--once`): a tmp dir with 2 files → correct add-chunks payloads (httpx mocked); deletion path calls remove-file. Continuous loop is smoke-tested lightly.

## Plan

1. Chunker + tests.
2. `/corpus/remove-file` endpoint + `daemon_client.corpus_remove_file` + test.
3. Watch config loader + one-shot sync (`efficient watch --once`) + tests (mock httpx).
4. Continuous mode via `watchfiles.awatch` + docs.
