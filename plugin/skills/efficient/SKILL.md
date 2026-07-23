---
name: efficient
description: Use the efficient MCP tools (lookup_symbol, find_references, retrieve_memory, store_memory, optimize_context, index_codebase, reindex_file) instead of brute-force file reading, grep-based impact analysis, or re-deriving prior decisions — cuts tokens and keeps context tight.
---

# Using the `efficient` MCP tools (token optimization)

The `efficient` MCP server is connected and its daemon is auto-started by this
plugin's monitor. ALWAYS use these tools instead of brute-force file reading
and re-derivation: `lookup_symbol` before reading a file to find a definition,
`find_references` before grepping an identifier. Policy hooks may deny large
`Read`s and identifier `Grep`s that these tools can answer. Use a stable `repo_id` and `agent_id` of `"project"` (the
plugin's hooks use the same), so the codebase graph and memory accrue
consistently.

## Codebase graph — read less, target more
- **`lookup_symbol(query, repo_id="project", k=5)`** — before opening a whole
  file to find a function/class, fetch just the relevant symbol slice.
- **`find_references(repo_id="project", symbol)`** — see who calls a symbol
  (callers) and what it calls (callees) before changing it. Use for impact
  analysis instead of grep-reading the tree.
- **`index_codebase(repo_id="project", path)`** — index a repo once. Normally
  handled automatically by the SessionStart hook; call it manually only if the
  graph looks empty.
- **`reindex_file(repo_id="project", file_path, source)`** — re-index a file
  after changing it outside the normal Edit/Write flow (the PostToolUse hook
  covers normal edits).

## Memory — recall and persist across sessions
- **`retrieve_memory(agent_id="project", query)`** — recall prior facts and
  decisions relevant to the task. Relevant memory is also auto-injected by the
  UserPromptSubmit hook; call this explicitly for a targeted lookup.
- **`store_memory(agent_id="project", session_id, turn, response)`** — persist
  durable facts and decisions worth remembering ("we chose X because Y",
  conventions, gotchas). Memory only accrues if you store it.

## Context — compress before you send a lot
- **`optimize_context(prompt, context, agent_id="project")`** — for large
  pasted context, route it through the pipeline first and use the returned
  `optimized_context` (it also reports `tokens_saved`).

## Notes
- Keep `repo_id`/`agent_id` = `"project"` to match the hooks.
- The daemon is managed by the plugin monitor; if tools error with connection
  failures, check `docker compose -f <plugin>/docker-compose.yml ps`.
- No API key is required for these tools (embeddings/graph/cache/retrieval/
  memory run locally).
