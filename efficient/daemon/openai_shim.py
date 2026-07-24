"""OpenAI-compatible chat-completions shim: lets any OpenAI-base-URL client
(aider, Continue, Cline, Open WebUI, ...) route through the daemon and get
semantic caching. Cache hit → served locally with no upstream call. Miss →
forwarded verbatim to an OpenAI-compatible upstream, response cached.

v1 caches only (no compression — rewriting chat messages risks the same
prefix-cache hazards as the gateway) and buffers streaming responses into
SSE frames rather than streaming token-by-token.
"""
import json
import os
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from efficient.db.client import get_async_db
from efficient.daemon.config import load_config
from efficient.daemon.strategies import get_strategy
from efficient.modules._base import OptimizeRequest

router = APIRouter()


def _upstream() -> str:
    return os.getenv("EFFICIENT_OPENAI_UPSTREAM", "https://api.openai.com/v1")


def _api_key(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.startswith("Bearer "):
        return header[7:].strip()
    return os.getenv("OPENAI_API_KEY", "")


def _completion(model: str, text: str, ptok: int, ctok: int,
                cache_hit: bool, saved: int) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                  "total_tokens": ptok + ctok},
        "efficient": {"cache_hit": cache_hit, "tokens_saved": saved},
    }


def _sse(completion: dict) -> StreamingResponse:
    chunk_id = completion["id"]
    model = completion["model"]
    text = completion["choices"][0]["message"]["content"]

    def frame(delta, finish):
        return "data: " + json.dumps({
            "id": chunk_id, "object": "chat.completion.chunk",
            "created": completion["created"], "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    async def gen():
        yield frame({"role": "assistant", "content": text}, None)
        yield frame({}, "stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "")
    stream = bool(body.get("stream"))
    db = get_async_db()
    config = await load_config(db)
    strategy = get_strategy(config.get("strategy"))
    cache_cfg = {**config.get("modules", {}).get("semantic_cache", {}),
                 "cache_key": strategy.cache_key}

    key = json.dumps(messages, sort_keys=True, default=str)
    from efficient.modules.semantic_cache import SemanticCache
    cache = SemanticCache(db, cache_cfg)
    req = OptimizeRequest(prompt=key, context="", agent_id=body.get("user", "default"),
                          framework="openai-shim", corpus_id=None)
    cached_req, result = await cache.process(req)
    import dataclasses
    from efficient.daemon.metrics import record_module_events
    await record_module_events(db, [dataclasses.asdict(result)])

    if result.short_circuit:
        completion = _completion(model, cached_req.context, 0, 0,
                                 cache_hit=True, saved=result.tokens_saved)
        return _sse(completion) if stream else JSONResponse(completion)

    from efficient.daemon.providers import call_openai_upstream
    try:
        text, ptok, ctok = await call_openai_upstream(
            _upstream(), _api_key(request), model, messages)
    except Exception as exc:
        return JSONResponse({"error": {"message": str(exc), "type": "upstream_error"}},
                            status_code=502)

    await cache.store(prompt=key, response=text, framework="openai-shim",
                      model=model, tokens_saved=ptok + ctok, agent_id="default",
                      corpus_id="")
    completion = _completion(model, text, ptok, ctok, cache_hit=False, saved=0)
    return _sse(completion) if stream else JSONResponse(completion)
