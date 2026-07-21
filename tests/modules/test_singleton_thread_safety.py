"""Regression tests: concurrent cold-start calls must not double-construct
the lazily-initialized embedding model / prompt compressor singletons."""

import threading
from concurrent.futures import ThreadPoolExecutor

import finops.modules.context_compressor as context_compressor
import finops.modules.embeddings as embeddings


def test_get_model_thread_safe_single_construction(monkeypatch):
    counter = {"n": 0}
    lock = threading.Lock()

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            with lock:
                counter["n"] += 1
            import time
            time.sleep(0.1)

    monkeypatch.setattr(embeddings, "SentenceTransformer", FakeSentenceTransformer)
    embeddings.reset_model()

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: embeddings._get_model(), range(8)))

        assert counter["n"] == 1
        first = results[0]
        assert all(r is first for r in results)
    finally:
        embeddings.reset_model()


def test_get_compressor_thread_safe_single_construction(monkeypatch):
    counter = {"n": 0}
    lock = threading.Lock()

    class FakePromptCompressor:
        def __init__(self, *args, **kwargs):
            with lock:
                counter["n"] += 1
            import time
            time.sleep(0.1)

    monkeypatch.setattr(context_compressor, "PromptCompressor", FakePromptCompressor)
    context_compressor._compressor = None

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: context_compressor._get_compressor(), range(8)))

        assert counter["n"] == 1
        first = results[0]
        assert all(r is first for r in results)
    finally:
        context_compressor._compressor = None
