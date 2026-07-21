import numpy as np
from unittest.mock import MagicMock
import efficient.modules.embeddings as emb


def _fake_model():
    m = MagicMock()
    m.encode_document.side_effect = lambda texts, **kw: [np.full(1024, 0.2) for _ in texts]
    m.encode_query.side_effect = lambda texts, **kw: [np.full(1024, 0.3) for _ in texts]
    return m


def test_embed_documents_returns_list_of_1024_floats(monkeypatch):
    monkeypatch.setattr(emb, "_get_model", _fake_model)
    out = emb.embed_documents(["a", "b"])
    assert len(out) == 2
    assert len(out[0]) == 1024
    assert isinstance(out[0][0], float)


def test_embed_query_returns_1024_floats(monkeypatch):
    monkeypatch.setattr(emb, "_get_model", _fake_model)
    out = emb.embed_query("hello")
    assert len(out) == 1024
    assert isinstance(out[0], float)


def test_embed_query_and_documents_use_distinct_encoders(monkeypatch):
    m = _fake_model()
    monkeypatch.setattr(emb, "_get_model", lambda: m)
    emb.embed_documents(["doc"])
    emb.embed_query("qry")
    m.encode_document.assert_called_once()
    m.encode_query.assert_called_once()


def test_reset_model_clears_singleton(monkeypatch):
    monkeypatch.setattr(emb, "_model", object())
    emb.reset_model()
    assert emb._model is None


def test_embed_documents_empty_input_returns_empty():
    assert emb.embed_documents([]) == []
