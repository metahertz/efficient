#!/usr/bin/env bash
# Shared helper: derive a stable, collision-resistant repo_id for the current
# project so different checkouts never share a codebase-graph store.
# repo_id = ${EFFICIENT_REPO_ID:-<basename>-<8 hex of sha256(abs git toplevel)>}
compute_repo_id() {
  if [ -n "${EFFICIENT_REPO_ID:-}" ]; then
    printf '%s' "$EFFICIENT_REPO_ID"
    return
  fi
  local dir root
  dir="${CLAUDE_PROJECT_DIR:-$PWD}"
  root=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null) || root="$dir"
  local hash
  hash=$(printf '%s' "$root" | shasum -a 256 2>/dev/null | cut -c1-8)
  [ -n "$hash" ] || hash=$(printf '%s' "$root" | cksum | cut -d' ' -f1)
  printf '%s-%s' "$(basename "$root")" "$hash"
}
