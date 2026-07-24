# Corpus Ingest + OpenAI-Compatible Shim — Design

**Date:** 2026-07-24
**Status:** Approved

Unlocks the idle modules for non-Claude-Code clients: `hybrid_retrieval` (needs a seeded corpus) and `semantic_cache`/`context_compressor`/`complete_proxy` (need completion traffic).

## Part 1 — Corpus ingest

- `POST /corpus/add-chunks` `{corpus_id, chunks:[{text, source_file, chunk_index, metadata?}]}` → `HybridRetrieval.add_chunks` → `{corpus_id, added}`. Enables `optimize_context(corpus_id=…)` retrieval.
- `daemon_client.corpus_add_chunks(corpus_id, chunks)` + MCP tool `add_corpus(corpus_id, chunks)`.
- Schema `CorpusAddChunksBody`; chunk_index defaults handled server-side if omitted (enumerate).

## Part 2 — OpenAI-compatible shim

`POST /v1/chat/completions` — OpenAI Chat Completions shape in and out, so any OpenAI-base-URL client (aider, Continue, Cline, Open WebUI, LibreChat, LM Studio-fronted apps) can point at the daemon and get semantic caching for free.

**Flow (mirrors `/complete`, reusing `SemanticCache`):**
1. Serialize `messages` → cache key; run `SemanticCache.process`. **Hit** → synthesize a ChatCompletion from the cached text, no upstream call (the measurable win). **Miss** → forward.
2. Forward the original `messages` **verbatim** to an OpenAI-compatible upstream (`EFFICIENT_OPENAI_UPSTREAM`, default `https://api.openai.com/v1`), using the caller's `Authorization` bearer key (fallback `OPENAI_API_KEY`). Store the response in the cache.
3. Response: standard ChatCompletion JSON with an added `efficient` block `{cache_hit, tokens_saved}`.

**Streaming:** if `stream:true`, buffer the upstream (non-stream) call and re-emit as `chat.completion.chunk` SSE frames + `[DONE]`. v1 does not stream token-by-token — documented limitation; keeps streaming clients working.

**Compression:** off by default (rewriting chat messages risks the same prefix-cache hazards as the gateway). Cache only in v1.

**Routing:** register the shim router **before** the gateway's `/v1/{path:path}` catch-all so the fixed path wins; add `call_openai_upstream` to `providers.py`.

**Attribution:** `clients.capability_for` maps `/v1/chat/completions` → `openai_shim` (checked before the generic `/v1`→gateway). Client label from `X-Efficient-Client` or `openai-client`.

## Non-goals

- No compression/token-by-token streaming in v1.
- No `/v1/completions` (legacy) or embeddings passthrough.

## Testing

- Corpus: add-chunks endpoint stores/embeds; MCP tool delegates; retrieval finds a seeded chunk.
- Shim: cache miss forwards to fake upstream + shapes response; second identical call is a cache hit with no upstream call; streaming request yields SSE chunks + `[DONE]`; `/v1/chat/completions` routes to the shim not the gateway; capability attribution = openai_shim.
