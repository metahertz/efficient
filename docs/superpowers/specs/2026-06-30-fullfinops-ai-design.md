# fullFinOps-AI Design Spec

**Goal:** A developer toolkit that saves as many AI tokens as possible by combining codebase graph indexing, semantic caching, agent memory, context compression, and hybrid retrieval — all backed by MongoDB, exposed as drop-in plugins for every major AI framework.

**Architecture:** Daemon-first. A local FastAPI daemon (port 7432) owns all MongoDB connections, module logic, and metrics collection. Every plugin (LangChain, LlamaIndex, AutoGen, CrewAI, raw Anthropic/OpenAI, Claude Code MCP) is a thin HTTP client to the daemon. On/Off toggles and benchmark harness live in the daemon.

**Tech Stack:** Python 3.11+ (FastAPI, Click, langchain-mongodb, tree-sitter, llmlingua, httpx), TypeScript (MCP SDK), MongoDB 7.0+ (documents + native vector search), vanilla HTML/JS dashboard.

---

## Global Constraints

- MongoDB version ≥ 7.0 required with `mongot` process for vector search. Two supported local setups: **MongoDB Atlas** (cloud) or **MongoDB Atlas Local** (Docker: `mongodb/mongodb-atlas-local`). Community edition without `mongot` is not supported. Daemon exits with a clear error if vector search is unavailable.
- Python ≥ 3.11
- All vector embeddings: `text-embedding-3-small` (1536 dimensions, OpenAI) — configurable
- Default daemon port: 7432 — configurable via env var `FINOPS_PORT`
- Monetary savings estimates use configurable per-model input/output token costs in the `config` collection (defaults: `$0.000003/input_token`, `$0.000015/output_token` matching claude-sonnet-4-6 pricing)
- No Mem0 dependency — memory stack built entirely on `langchain-mongodb`
- Dashboard has zero npm dependencies — plain HTML/JS/CSS only

---

## Architecture

```
Plugins (Python)                                TypeScript
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐
│LangChain │ │LlamaIndex│ │  AutoGen │ │  CrewAI  │ │ Raw API  │ │ Claude Code │
│  tool    │ │  tool    │ │middleware│ │  tool    │ │middleware│ │  MCP server │
└────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬──────┘
     └────────────┴────────────┴────────────┴────────────┴──────────────┘
                                HTTP REST (localhost:7432)
                       ┌──────────────────────────────────┐
                       │         FastAPI Daemon            │
                       │  Module router · Metrics engine  │
                       │  On/Off toggles · Benchmark API  │
                       └───────────────┬──────────────────┘
         ┌───────────┬─────────────────┼──────────────┬──────────────┐
    ┌────┴────┐ ┌────┴────┐ ┌─────────┴──┐ ┌─────────┴──┐ ┌────────┴────┐
    │Codebase │ │Semantic │ │   Agent    │ │  Context   │ │  Hybrid     │
    │ Graph   │ │  Cache  │ │   Memory   │ │ Compressor │ │  Retrieval  │
    └────┬────┘ └────┬────┘ └─────┬──────┘ └──────┬─────┘ └────┬────────┘
         └───────────┴────────────┴───────────────┴────────────┘
                                MongoDB 7.0+
                       (documents + native vector index)

                  Web dashboard (http://localhost:7432/dashboard)
                  CLI (finops start/stop/status/savings/toggle/bench/index)
```

**Request flow:** plugin intercepts outgoing prompt → `POST /optimize` to daemon → daemon routes through enabled modules in pipeline order → returns optimized context + per-module token counts → plugin forwards to LLM → response + savings recorded to MongoDB.

**On/Off toggle:** per-module `enabled` flag in `config` collection. Daemon checks on each request. Flip via `finops toggle <module>` or dashboard — no plugin restart needed.

---

## Daemon API

All plugins speak this contract. Plugin code = one `POST /optimize` call + response unpack.

### Core Endpoints

```
POST /optimize
  Request:  { prompt, context, agent_id, framework, corpus_id? }
  Response: { optimized_prompt, optimized_context, tokens_saved, module_results[] }

GET  /cache/lookup
  Request:  { prompt_hash, embedding }
  Response: { hit: bool, response?, similarity_score }

POST /memory/retrieve
  Request:  { agent_id, query, turn }
  Response: { working: [...], episodic: [...], semantic: [...] }

POST /memory/store
  Request:  { agent_id, turn, response }
  Response: { stored: bool }

POST /benchmark/run
  Request:  { benchmark, model_config, modules_enabled[] }
  Response: { run_id }   — async; poll or WebSocket for results

GET  /benchmark/results/{run_id}
  Response: { status, tokens_saved, quality_delta, latency_delta, per_category[] }

GET  /metrics
  Response: { total_tokens_saved, cache_hit_rate, compression_ratio, per_module[] }

GET  /config
PUT  /config
  Body: { modules: { semantic_cache: { enabled, similarity_threshold }, ... } }

GET  /health
WS   /stream   — live token savings counter for dashboard
```

### Module Result Shape

Every module appends one entry to `module_results[]`:
```json
{
  "module": "semantic_cache",
  "tokens_in": 4200,
  "tokens_out": 0,
  "tokens_saved": 4200,
  "latency_ms": 12,
  "detail": "cache hit (similarity=0.97)"
}
```

---

## Modules

All modules implement `BaseModule(ABC)` with `process(request) -> ModuleResult` and `metrics() -> dict`.

### Module 1: Codebase Graph (`finops/modules/codebase_graph.py`)

Parses a repo using Tree-Sitter into an AST node graph stored in MongoDB. On each request, given a file path or symbol name, returns only the relevant definitions + call sites — not the whole file.

- **Indexing:** triggered by `finops index <repo_path>` or daemon startup if configured. Walks repo files, parses with Tree-Sitter (66 languages), extracts symbols → stores in `codebase_nodes` with embeddings.
- **Query:** symbol name or natural language description → vector similarity search → returns minimal code slice.
- **Token saving mechanism:** replaces "read entire file" with "here is the symbol you need."

### Module 2: Semantic Cache (`finops/modules/semantic_cache.py`)

Two-layer cache:
1. **Exact match:** SHA-256 hash of prompt → `cache_entries` lookup. O(1).
2. **Semantic match:** embedding similarity via MongoDB vector index, cosine threshold configurable (default 0.92). Returns cached response if above threshold.

Also manages Anthropic/OpenAI **prefix cache breakpoints** — inserts `cache_control` markers at optimal positions in long prompts to maximise native API cache hits.

- Cache TTL: configurable (default 168h / 7 days). MongoDB TTL index on `expires_at`.
- On cache miss: forwards request, stores response + embedding.

### Module 3: Agent Memory (`finops/modules/agent_memory.py`)

Three-tier memory stack built entirely on `langchain-mongodb`. No Mem0 dependency.

**Tier 1 — Working Memory** (`MongoDBChatMessageHistory`):
- Raw conversation turns, last N (default 20) always injected.
- Per session, per agent.

**Tier 2 — Episodic Memory** (`MongoDBAtlasVectorSearch` over `episodic_memory`):
- Compressed summaries of older turns, retrieved by relevance to current query.
- TTL: configurable (default 30 days).

**Tier 3 — Semantic Memory** (`MongoDBAtlasVectorSearch` over `semantic_memory`):
- Discrete extracted facts ("user prefers Python", "project uses FastAPI").
- Retrieved by cosine similarity to current query.
- TTL: configurable (default 90 days).

**Fact extraction (gap we implement, ~100 lines):**
```python
def extract_and_store_facts(turn, response, agent_id):
    facts = llm.invoke(FACT_EXTRACTION_PROMPT.format(turn=turn, response=response))
    for fact in facts:
        existing = vector_store.similarity_search(fact, k=1, score_threshold=0.95)
        if existing:
            update_fact(existing[0], fact)   # deduplicate
        else:
            vector_store.add_texts([fact], metadatas=[{"agent_id": agent_id}])
```

**Forgetting policies:** MongoDB TTL indexes on `episodic_memory.expires_at` and `semantic_memory.expires_at`. Configurable per agent via `PUT /config`.

### Module 4: Context Compressor (`finops/modules/context_compressor.py`)

Wraps LLMLingua-2. Runs only when context exceeds token threshold (default 8,000 tokens) to avoid overhead on short prompts. Compression ratio configurable (default 4x target).

- Saves `compression_stats` to MongoDB for dashboard analytics.
- Does not use MongoDB for operation — compression runs as a local model call.

### Module 5: Hybrid Retrieval (`finops/modules/hybrid_retrieval.py`)

Combines BM25 sparse search + MongoDB vector search (dense) with RRF (Reciprocal Rank Fusion) score merging. Used when the agent has a document corpus.

- **BM25:** pre-tokenized `bm25_tokens` field + MongoDB text index.
- **Dense:** `MongoDBAtlasVectorSearch.similarity_search()` on `corpus_chunks`.
- **RRF fusion:** `score = Σ 1/(k + rank_i)` where k=60 (default).
- Returns top-K ranked chunks (default 5) instead of full documents.

### Module 6: Benchmark Runner (`finops/modules/benchmark_runner.py`)

Runs target LLM twice per sample — baseline (modules OFF) and optimised (modules ON). Records token counts, latency, and LLM-judge quality score for both.

**Supported benchmarks:**
- **HELMET** — 7 categories (recall, RAG, ICL, re-rank, QA, summarization, citation), 0–128K tokens
- **RULER** — retrieval, multi-hop, aggregation, long QA, up to 256K tokens
- **LongBench** — long-context QA, summarization, few-shot
- **Custom** — user-supplied JSONL: `{"prompt": "...", "expected_output": "..."}`

**Dataset loading:** HELMET, RULER, LongBench pulled from HuggingFace on first run, cached to `~/.finops/datasets/`.

**Quality scoring:** LLM-as-judge using `claude-sonnet-4-6` (configurable). Separate call comparing expected vs actual output on 0–1 scale. Can be disabled to reduce benchmark cost.

---

## Plugin Layer

### Shared Python Base (`finops/plugins/_base.py`)

```python
class FinOpsPlugin:
    def __init__(self, daemon_url="http://localhost:7432", agent_id=None):
        self.daemon_url = daemon_url
        self.agent_id   = agent_id or str(uuid4())

    def optimize(self, prompt, context, framework, corpus_id=None):
        r = httpx.post(f"{self.daemon_url}/optimize",
                       json={"prompt": prompt, "context": context,
                             "agent_id": self.agent_id, "framework": framework,
                             "corpus_id": corpus_id})
        return r.json()
```

### Per-Framework Adapters

| Plugin | File | Integration point |
|---|---|---|
| LangChain | `langchain_plugin.py` | Subclasses `BaseTool`; drop into `AgentExecutor(tools=[FinOpsTool()])` |
| LlamaIndex | `llamaindex_plugin.py` | Subclasses LlamaIndex `BaseTool` |
| AutoGen | `autogen_plugin.py` | Wraps `ConversableAgent.generate_reply` hook |
| CrewAI | `crewai_plugin.py` | Subclasses `BaseTool`; registered in `Agent(tools=[...])` |
| Raw Anthropic | `anthropic_plugin.py` | Drop-in: `from finops.plugins.anthropic_plugin import AnthropicClient as Anthropic` |
| Raw OpenAI | `openai_plugin.py` | Drop-in: `from finops.plugins.openai_plugin import OpenAIClient as OpenAI` |
| Claude Code MCP | `finops-mcp/src/index.ts` | TypeScript MCP server; exposes `optimize_context`, `lookup_symbol`, `retrieve_memory` tools |

**LangChain memory integration:** LangChain plugin uses `MongoDBChatMessageHistory` directly (native, zero daemon overhead for working memory). Other plugins route all memory through `/memory/*` daemon endpoints.

### Installation

```bash
pip install finops-ai                     # Python core + all Python plugins
npm install -g @finops-ai/mcp             # Claude Code MCP server
```

Claude Code MCP config (`~/.claude/settings.json`):
```json
{ "mcpServers": { "finops": { "command": "finops-mcp", "args": [] } } }
```

---

## Dashboard + CLI

### CLI (`finops` command)

```bash
finops start                              # start daemon (background, port 7432)
finops stop                               # stop daemon
finops status                             # daemon health + per-module on/off state
finops savings                            # token savings summary (24h / 7d / all)
finops toggle <module>                    # enable/disable a module
finops bench run <name> [--model] [--modules]
finops bench results                      # last benchmark run results
finops index <repo_path>                  # trigger codebase graph indexing
```

### Web Dashboard (`http://localhost:7432/dashboard`)

Plain HTML/JS/CSS, zero npm dependencies, ~400 lines. Four panels:

```
┌─────────────────────────────────────────────────────────┐
│  fullFinOps-AI   ● daemon running    [7d] [24h] [all]   │
├──────────────────────┬──────────────────────────────────┤
│  TOKENS SAVED        │  PER-MODULE BREAKDOWN            │
│  2,847,293  total    │  ■ codebase_graph    42%  [ON]   │
│  $14.23 saved        │  ■ semantic_cache    31%  [ON]   │
│  ───────────         │  ■ agent_memory      18%  [ON]   │
│  Live: ▁▃▅▇▅▃▁      │  ■ compressor         7%  [ON]   │
│  (WebSocket)         │  ■ hybrid_retrieval   2%  [OFF]  │
├──────────────────────┴──────────────────────────────────┤
│  BENCHMARK RESULTS (last run: HELMET, claude-sonnet-4)  │
│  Baseline: 48,200 tokens  Optimised: 14,800 tokens      │
│  Quality delta: +0.3%     Latency delta: +120ms         │
│  [Run new benchmark ▼]                                  │
├─────────────────────────────────────────────────────────┤
│  MEMORY EXPLORER   agent: default                       │
│  Working (12 turns)  Episodic (847 facts)  [Flush]      │
│  Search memories: [___________________] [Search]        │
└─────────────────────────────────────────────────────────┘
```

Data via REST + WebSocket `/stream` for live token counter.

---

## MongoDB Schema

Database: `finops`. All `embedding` fields use cosine similarity vector index.

```javascript
codebase_nodes:    { _id, repo_id, symbol, type, file_path, line_start, line_end,
                     source_snippet, language, embedding[1536], references[], indexed_at }
                   // Indexes: {repo_id,symbol}, {repo_id,file_path}, vector(embedding)

cache_entries:     { _id, prompt_hash, embedding[1536], prompt_preview, response,
                     framework, model, tokens_saved, hit_count, created_at,
                     last_hit_at, expires_at }
                   // Indexes: {prompt_hash} unique, vector(embedding), TTL(expires_at)

working_memory:    { _id, agent_id, session_id,
                     messages[{role,content,timestamp}], created_at, updated_at }
                   // Indexes: {agent_id, session_id}

episodic_memory:   { _id, agent_id, content, embedding[1536],
                     source_turns[ObjectId], created_at, expires_at }
                   // Indexes: {agent_id}, vector(embedding), TTL(expires_at)

semantic_memory:   { _id, agent_id, fact, embedding[1536],
                     confidence, source_session, created_at, updated_at, expires_at }
                   // Indexes: {agent_id}, vector(embedding), TTL(expires_at)

compression_stats: { _id, request_id, framework, model,
                     original_tokens, compressed_tokens, ratio, latency_ms, created_at }
                   // Indexes: {created_at}

corpus_chunks:     { _id, corpus_id, source_file, chunk_index, text,
                     embedding[1536], bm25_tokens[string], metadata{}, created_at }
                   // Indexes: {corpus_id}, vector(embedding), text(bm25_tokens)

benchmark_runs:    { _id:run_id, benchmark, model_config, modules_enabled[],
                     status, results[{sample_id, category, baseline{}, optimised{},
                     tokens_saved, quality_delta, latency_delta}],
                     summary{}, started_at, completed_at }
                   // Indexes: {started_at}

config:            { _id:"global", modules:{ codebase_graph:{enabled,repo_paths[]},
                     semantic_cache:{enabled,similarity_threshold,ttl_hours},
                     agent_memory:{enabled,working_memory_turns,episodic_ttl_days,
                       semantic_ttl_days},
                     context_compressor:{enabled,token_threshold,target_ratio},
                     hybrid_retrieval:{enabled,top_k,rrf_k},
                     benchmark_runner:{enabled,judge_model} },
                     embedding_model, cost_per_token, updated_at }
```

Vector indexes created at daemon startup via `pymongo` if not present. Daemon checks MongoDB ≥ 7.0 on startup.

---

## Repository Structure

```
fullFinOps-AI/
├── pyproject.toml                  # Python package: finops-ai
├── package.json                    # TypeScript MCP workspace root
│
├── finops/
│   ├── daemon/
│   │   ├── app.py                  # FastAPI app, route registration
│   │   ├── router.py               # module pipeline orchestrator
│   │   ├── metrics.py              # token savings aggregation
│   │   └── websocket.py            # live updates stream
│   │
│   ├── modules/
│   │   ├── _base.py                # BaseModule(ABC)
│   │   ├── codebase_graph.py
│   │   ├── semantic_cache.py
│   │   ├── agent_memory.py         # langchain-mongodb + fact extraction
│   │   ├── context_compressor.py   # LLMLingua-2 wrapper
│   │   ├── hybrid_retrieval.py     # BM25 + vector + RRF
│   │   └── benchmark_runner.py     # HELMET/RULER/LongBench harness
│   │
│   ├── plugins/
│   │   ├── _base.py
│   │   ├── langchain_plugin.py
│   │   ├── llamaindex_plugin.py
│   │   ├── autogen_plugin.py
│   │   ├── crewai_plugin.py
│   │   ├── anthropic_plugin.py
│   │   └── openai_plugin.py
│   │
│   ├── db/
│   │   ├── client.py               # MongoDB connection singleton
│   │   ├── indexes.py              # vector + text index creation on startup
│   │   └── collections.py          # collection name constants
│   │
│   └── cli/
│       └── main.py                 # Click CLI
│
├── finops-mcp/
│   ├── package.json
│   ├── tsconfig.json
│   └── src/
│       ├── index.ts                # MCP server entrypoint
│       ├── tools/
│       │   ├── optimize_context.ts
│       │   ├── lookup_symbol.ts
│       │   └── retrieve_memory.ts
│       └── daemon_client.ts        # fetch wrapper for daemon REST API
│
├── dashboard/
│   ├── index.html
│   ├── app.js                      # vanilla JS, ~400 lines
│   └── style.css
│
├── tests/
│   ├── modules/
│   ├── plugins/
│   ├── daemon/
│   └── fixtures/
│
└── docs/
    └── superpowers/
        └── specs/
            └── 2026-06-30-fullfinops-ai-design.md
```
