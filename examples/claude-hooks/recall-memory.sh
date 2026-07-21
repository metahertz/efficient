#!/usr/bin/env bash
# UserPromptSubmit hook: auto-recall relevant efficient memory and inject it as context.
# Copy to <your project>/.claude/hooks/ and chmod +x. Requires jq + the daemon running.
AUTH_ARGS=()
[ -n "${FINOPS_API_TOKEN:-}" ] && AUTH_ARGS=(-H "Authorization: Bearer $FINOPS_API_TOKEN")
input=$(cat)
q=$(printf '%s' "$input" | jq -r '.prompt // empty')
[ -z "$q" ] && exit 0
resp=$(curl -s -m 10 -X POST http://localhost:7432/memory/retrieve \
  "${AUTH_ARGS[@]}" \
  -H 'content-type: application/json' \
  -d "$(jq -n --arg a project --arg q "$q" '{agent_id:$a, query:$q}')" 2>/dev/null) || exit 0
mem=$(printf '%s' "$resp" | jq -r '((.semantic // []) + (.episodic // [])) | if length > 0 then "Relevant memory (from efficient):\n- " + join("\n- ") else empty end' 2>/dev/null)
[ -z "$mem" ] && exit 0
jq -n --arg c "$mem" '{hookSpecificOutput: {hookEventName: "UserPromptSubmit", additionalContext: $c}}'
exit 0
