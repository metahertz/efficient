import pytest
from pymongo.errors import OperationFailure
from efficient.db.vector import vector_search


class _FakeCollection:
    def __init__(self, docs=None, exc=None):
        self._docs = docs or []
        self._exc = exc

    async def aggregate(self, pipeline):
        if self._exc is not None:
            raise self._exc
        for doc in self._docs:
            yield doc


async def test_vector_search_yields_docs():
    col = _FakeCollection(docs=[{"_id": 1}, {"_id": 2}])
    results = await vector_search(col, [{"$vectorSearch": {}}])
    assert results == [{"_id": 1}, {"_id": 2}]


async def test_vector_search_returns_empty_when_not_queryable():
    exc = OperationFailure("cannot query vector index while in state INITIAL_SYNC")
    col = _FakeCollection(exc=exc)
    results = await vector_search(col, [{"$vectorSearch": {}}])
    assert results == []


async def test_vector_search_reraises_other_operation_failure():
    exc = OperationFailure("some other error")
    col = _FakeCollection(exc=exc)
    with pytest.raises(OperationFailure):
        await vector_search(col, [{"$vectorSearch": {}}])
