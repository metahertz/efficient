# TypeScript Support + Per-Repo Scoping — Design

**Date:** 2026-07-23
**Status:** Approved

Fixes two bugs found in live use: (1) the codebase graph only extracts `.py`, so TS/JS repos index zero symbols; (2) hooks hardcode `repo_id="project"`, so every checkout shares one store — a cross-project data leak (a TS repo returned symbols from an unrelated Python project).

## Part 1 — Multi-language extraction

Generalize `codebase_graph.py` from one Python function to a per-language spec table:

```
_LANGS = {
  ".py":  Python grammar,
  ".ts":  TypeScript grammar,
  ".tsx": TSX grammar,
  ".js" / ".jsx" / ".mjs" / ".cjs": JavaScript grammar,
}
```

Each spec: compiled `Parser`, `symbol_types` (node type → "function"|"class"|"method"|"interface"|"type"), a name resolver, and a call-node spec (call node type + callee field + member/attribute node type + property field). One generic `_extract(source, file_path, ext)` walks the tree using the spec — replaces `_extract_python_symbols`; the per-symbol doc shape is unchanged (adds correct `language`).

TS/JS symbol nodes: `function_declaration`, `class_declaration`, `method_definition`, `interface_declaration`, `type_alias_declaration`, and `variable_declarator` whose value is an `arrow_function`/`function_expression` (→ "function", name from the declarator). Calls: `call_expression` → `function` field; bare `identifier` or `member_expression` (property field).

New deps: `tree-sitter-typescript>=0.23`, `tree-sitter-javascript>=0.23`.

## Part 2 — Per-repo scoping

Root cause: `repo_id="project"` is a global constant in the hooks. Fix by deriving a stable, collision-resistant id per checkout in a shared helper `plugin/scripts/_repo_id.sh` (sourced by every hook):

```
repo_id = ${EFFICIENT_REPO_ID:-<basename>-<first 8 of sha256(abs git-toplevel path)>}
```

- Toplevel via `git -C "$CLAUDE_PROJECT_DIR" rev-parse --show-toplevel`, fallback to `$CLAUDE_PROJECT_DIR`.
- basename keeps it human-readable in the dashboard; the path-hash suffix guarantees uniqueness across same-named dirs.
- All five affected scripts (`efficient-autoindex.sh`, `reindex-on-edit.sh`, `steer-large-reads.sh`, `steer-grep.sh`, `session-context.sh`) use `$(compute_repo_id)` instead of the literal `project`. `recall-memory.sh`/`sync-native-memory.sh` use agent_id (memory scope, separate concern) — leave as `project` for now but note it.

Steering hooks that query the graph (`steer-grep`, `steer-large-reads` reason text) must reference the same derived id so their suggestions point at the right store.

## Non-goals

- No migration of the existing polluted `project` store — a re-index under the new id supersedes it; document that `codebase_graph` data keyed `project` is stale and can be dropped.
- Language coverage beyond py/ts/tsx/js/jsx — Go/Rust/etc. later via the same table.

## Testing

- `tests/modules/test_codebase_graph_ts.py`: extract functions/classes/methods/interfaces/types + arrow-fn consts from a TS sample and a TSX sample; call edges (`find_references`) across TS symbols; `.js` handled.
- `tests/plugin/test_repo_id.sh` driven from pytest: `compute_repo_id` is stable across calls, differs for different paths, honors `EFFICIENT_REPO_ID`.
- Manifest test: hooks no longer contain the literal `"project"` for repo scoping (grep guard).
