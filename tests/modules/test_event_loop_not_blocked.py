import asyncio
import time

from efficient.modules._base import OptimizeRequest
from efficient.modules.semantic_cache import SemanticCache


async def test_slow_embedding_does_not_block_event_loop(efficient_db, monkeypatch):
    def slow_embed_query(text):
        time.sleep(0.5)  # simulates model inference on CPU
        return [0.1] * 1024

    import efficient.modules.semantic_cache as sc
    monkeypatch.setattr(sc, "embed_query", slow_embed_query)

    cache = SemanticCache(efficient_db, {})
    request = OptimizeRequest(prompt="p", context="", agent_id="a", framework="t", corpus_id=None)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.02)
            ticks += 1

    await asyncio.gather(cache.process(request), heartbeat())
    # If embed_query ran on the event loop, the heartbeat stalls for 0.5s
    # and completes far fewer iterations during the overlap.
    assert ticks >= 15
