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


async def _run_with_heartbeat(coro):
    ticks = 0

    async def heartbeat():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.02)
            ticks += 1

    await asyncio.gather(coro, heartbeat())
    return ticks


async def test_codebase_index_does_not_block_event_loop(efficient_db, monkeypatch):
    def slow_embed_documents(texts):
        time.sleep(0.5)  # simulates first-call model download/load
        return [[0.1] * 1024 for _ in texts]

    import efficient.modules.codebase_graph as cg
    monkeypatch.setattr(cg, "embed_documents", slow_embed_documents)

    from efficient.modules.codebase_graph import CodebaseGraph
    graph = CodebaseGraph(efficient_db, {})
    ticks = await _run_with_heartbeat(
        graph.index_file("r", "a.py", "def f():\n    return 1\n")
    )
    assert ticks >= 15


async def test_hybrid_add_chunks_does_not_block_event_loop(efficient_db, monkeypatch):
    def slow_embed_documents(texts):
        time.sleep(0.5)
        return [[0.1] * 1024 for _ in texts]

    import efficient.modules.hybrid_retrieval as hr
    monkeypatch.setattr(hr, "embed_documents", slow_embed_documents)

    from efficient.modules.hybrid_retrieval import HybridRetrieval
    retrieval = HybridRetrieval(efficient_db, {})
    ticks = await _run_with_heartbeat(
        retrieval.add_chunks("c", [{"text": "t", "source_file": "d", "chunk_index": 0, "metadata": {}}])
    )
    assert ticks >= 15
