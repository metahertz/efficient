from pymongo.errors import OperationFailure

_NOT_QUERYABLE = ("not queryable", "cannot query vector index", "initial_sync", "does not exist")


def _is_not_queryable(exc: OperationFailure) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _NOT_QUERYABLE)


async def vector_search(collection, pipeline: list[dict]) -> list[dict]:
    results = []
    try:
        async for doc in collection.aggregate(pipeline):
            results.append(doc)
    except OperationFailure as exc:
        if _is_not_queryable(exc):
            return []
        raise
    return results
