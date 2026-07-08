# Using the `efficient` MCP tools (token optimization)

The `efficient` MCP server is connected. Prefer its tools over brute-force
file reading and re-derivation — they cut tokens and keep context tight.
Use a stable `repo_id` and `agent_id` of `"project"` (the hooks use the same),
so the codebase graph and memory accrue consistently.

## Codebase graph — read less, target more
- **`lookup_symbol(query, repo_id="project", k=5)`** — before opening a whole
  file to find a function/class, fetch just the relevant symbol slice.
- **`find_references(repo_id="project", symbol)`** — to see who calls a symbol
  (callers) and what it calls (callees) before changing it. Use for impact
  analysis instead of grep-reading the tree.
- **`index_codebase(repo_id="project", path="/repo")`** — index a repo once.
  Normally handled automatically by the SessionStart hook; call it manually only
  if the graph looks empty.
- **`reindex_file(repo_id="project", file_path, source)`** — after you edit a
  file, re-index it so `lookup_symbol`/`find_references` stay accurate. Normally
  handled automatically by the PostToolUse hook; call it manually if you changed
  a file outside the normal Edit/Write flow.

## Memory — recall and persist across sessions
- **`retrieve_memory(agent_id="project", query)`** — recall prior facts and
  decisions relevant to the task. Relevant memory is also auto-injected by the
  UserPromptSubmit hook, but call this explicitly for a targeted lookup.
- **`store_memory(agent_id="project", session_id, turn, response)`** — persist
  durable facts and decisions worth remembering ("we chose X because Y",
  conventions, gotchas). Memory only accrues if you store it, so store the
  things future sessions should know.

## Context — compress before you send a lot
- **`optimize_context(prompt, context, agent_id="project")`** — for large pasted
  context, route it through the pipeline first and use the returned
  `optimized_context` (it also reports `tokens_saved`).

## Notes
- Keep `repo_id`/`agent_id` = `"project"` to match the hooks.
- The daemon must be running (`docker compose up -d daemon`). No API key is
  required for these tools (embeddings/graph/cache/retrieval/memory are local).
