import pytest
from unittest.mock import MagicMock
from efficient.modules.context_compressor import ContextCompressor, _count_tokens
from efficient.modules._base import OptimizeRequest
from efficient.db.collections import COMPRESSION_STATS


@pytest.fixture(autouse=True)
def mock_compressor(monkeypatch):
    fake = MagicMock()
    def fake_compress(context_list, rate, force_tokens, **kw):
        text = context_list[0]
        words = text.split()
        compressed = " ".join(words[: max(1, len(words) // 4)])
        return {"compressed_prompt": compressed}
    fake.compress_prompt.side_effect = fake_compress
    monkeypatch.setattr("efficient.modules.context_compressor._get_compressor", lambda: fake)


@pytest.fixture
def config():
    return {"token_threshold": 10, "target_ratio": 4.0}


@pytest.fixture
async def compressor(efficient_db, config):
    return ContextCompressor(efficient_db, config)


@pytest.fixture
def short_req():
    return OptimizeRequest(prompt="hi", context="short", agent_id="a1", framework="test")


@pytest.fixture
def long_req():
    long_ctx = " ".join(["word"] * 200)
    return OptimizeRequest(prompt="hi", context=long_ctx, agent_id="a1", framework="test")


async def test_bypass_when_context_below_threshold(compressor, efficient_db, short_req):
    new_req, result = await compressor.process(short_req)
    assert new_req.context == "short"
    assert result.tokens_saved == 0
    assert result.tokens_added == 0
    assert "bypass" in result.detail
    assert await efficient_db[COMPRESSION_STATS].count_documents({}) == 0


async def test_compresses_when_above_threshold(compressor, long_req):
    before = len(long_req.context) // 4
    new_req, result = await compressor.process(long_req)
    assert len(new_req.context) < len(long_req.context)
    assert _count_tokens(new_req.context) < _count_tokens(long_req.context)
    assert result.tokens_saved > 0
    assert result.tokens_added == 0
    assert result.baseline_tokens == before


async def test_saves_compression_stats(compressor, efficient_db, long_req):
    await compressor.process(long_req)
    count = await efficient_db[COMPRESSION_STATS].count_documents({})
    assert count == 1
    doc = await efficient_db[COMPRESSION_STATS].find_one({})
    assert doc["original_tokens"] > 0
    assert doc["compressed_tokens"] > 0
    assert doc["ratio"] > 1.0
