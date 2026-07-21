import pytest
from efficient.modules.hybrid_retrieval import HybridRetrieval
from efficient.modules._base import OptimizeRequest
from efficient.db.collections import CORPUS_CHUNKS

FIXED_EMBEDDING = [0.1] * 1024


@pytest.fixture(autouse=True)
def mock_embed(monkeypatch):
    monkeypatch.setattr("efficient.modules.hybrid_retrieval.embed_query", lambda t: FIXED_EMBEDDING)
    monkeypatch.setattr("efficient.modules.hybrid_retrieval.embed_documents", lambda ts: [FIXED_EMBEDDING] * len(ts))


@pytest.fixture
def config():
    return {"top_k": 3, "rrf_k": 60}


@pytest.fixture
async def retrieval(efficient_db, config):
    return HybridRetrieval(efficient_db, config)


@pytest.fixture
def req_no_corpus():
    return OptimizeRequest(prompt="hi", context="ctx", agent_id="a1", framework="test")


async def test_process_no_op_when_no_corpus_id(retrieval, req_no_corpus):
    new_req, result = await retrieval.process(req_no_corpus)
    assert new_req is req_no_corpus
    assert result.tokens_saved == 0
    assert "no corpus" in result.detail


async def test_add_chunks_stores_in_mongo(retrieval, efficient_db):
    chunks = [
        {"text": "MongoDB is a document database", "source_file": "doc.txt", "chunk_index": 0, "metadata": {}},
        {"text": "Python is a programming language", "source_file": "doc.txt", "chunk_index": 1, "metadata": {}},
    ]
    count = await retrieval.add_chunks("corp1", chunks)
    assert count == 2
    stored = await efficient_db[CORPUS_CHUNKS].count_documents({"corpus_id": "corp1"})
    assert stored == 2
    doc = await efficient_db[CORPUS_CHUNKS].find_one({"corpus_id": "corp1", "chunk_index": 0})
    assert doc["bm25_tokens"]  # non-empty tokenization


async def test_rrf_fusion_returns_ranked_results():
    from efficient.modules.hybrid_retrieval import _rrf_fusion
    results_a = [{"_id": "a"}, {"_id": "b"}, {"_id": "c"}]
    results_b = [{"_id": "b"}, {"_id": "c"}, {"_id": "a"}]
    fused = _rrf_fusion(results_a, results_b, k=60)
    ids = [r["_id"] for r in fused]
    # b: 1/61+1/62, a: 1/61+1/63, c: 1/63+1/62 — deterministic order
    assert ids == ["b", "a", "c"]


async def test_process_no_op_when_no_chunks_found(efficient_db):
    module = HybridRetrieval(efficient_db, {"top_k": 3, "rrf_k": 60})
    request = OptimizeRequest(
        prompt="find me something",
        context="ctx",
        agent_id="a1",
        framework="test",
        corpus_id="empty-corpus",
    )
    new_req, result = await module.process(request)
    assert new_req is request
    assert new_req.context == "ctx"
    assert result.tokens_saved == 0
    assert "no chunks" in result.detail
