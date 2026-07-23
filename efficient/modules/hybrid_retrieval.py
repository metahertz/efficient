import asyncio
import time
from datetime import datetime, timezone

from motor.motor_asyncio import AsyncIOMotorDatabase

from efficient.modules._base import BaseModule, OptimizeRequest, ModuleResult
from efficient.modules.embeddings import embed_query, embed_documents
from efficient.db.collections import CORPUS_CHUNKS
from efficient.db.vector import vector_search


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _rrf_fusion(dense_results: list[dict], sparse_results: list[dict], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    id_to_doc: dict[str, dict] = {}
    for rank, doc in enumerate(dense_results):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_doc[doc_id] = doc
    for rank, doc in enumerate(sparse_results):
        doc_id = str(doc["_id"])
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_doc[doc_id] = doc
    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [id_to_doc[doc_id] for doc_id in ranked_ids]


class HybridRetrieval(BaseModule):
    name = "hybrid_retrieval"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._top_k = config.get("top_k", 5)
        self._rrf_k = config.get("rrf_k", 60)

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        if not request.corpus_id:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=0.0, detail="no corpus_id provided",
            )
        t0 = time.perf_counter()
        dense = await self._dense_search(request.corpus_id, request.prompt)
        sparse = await self._sparse_search(request.corpus_id, request.prompt)
        fused = _rrf_fusion(dense, sparse, k=self._rrf_k)[: self._top_k]
        if not fused:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=(time.perf_counter() - t0) * 1000, detail="no chunks found",
            )
        retrieved = "\n\n".join(doc["text"] for doc in fused)
        baseline_tokens = await self._corpus_tokens(request.corpus_id)
        tokens_added = _count_tokens(retrieved)
        tokens_in = _count_tokens(request.context)
        section = "## Retrieved Docs\n" + retrieved
        new_context = request.context + ("\n\n" if request.context else "") + section
        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=new_context,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=tokens_in,
            tokens_out=_count_tokens(new_context),
            tokens_saved=max(0, baseline_tokens - tokens_added),
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail=f"retrieved {len(fused)} chunks (baseline={baseline_tokens} full-corpus tokens)",
            tokens_added=tokens_added,
            baseline_tokens=baseline_tokens,
        )

    async def add_chunks(self, corpus_id: str, chunks: list[dict]) -> int:
        texts = [c["text"] for c in chunks]
        embeddings = await asyncio.to_thread(embed_documents, texts)
        now = datetime.now(timezone.utc)
        for chunk, emb in zip(chunks, embeddings):
            tokens = chunk["text"].lower().split()
            await self._db[CORPUS_CHUNKS].update_one(
                {"corpus_id": corpus_id, "chunk_index": chunk["chunk_index"],
                 "source_file": chunk["source_file"]},
                {"$set": {
                    "corpus_id": corpus_id,
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "embedding": emb,
                    "bm25_tokens": tokens,
                    "metadata": chunk.get("metadata", {}),
                    "created_at": now,
                }},
                upsert=True,
            )
        return len(chunks)

    async def _dense_search(self, corpus_id: str, query: str) -> list[dict]:
        embedding = await asyncio.to_thread(embed_query, query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "corpus_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": self._top_k * 4,
                    "limit": self._top_k,
                    "filter": {"corpus_id": {"$eq": corpus_id}},
                }
            },
            {"$project": {"embedding": 0}},
        ]
        results = []
        for doc in await vector_search(self._db[CORPUS_CHUNKS], pipeline):
            results.append(doc)
        return results

    async def _sparse_search(self, corpus_id: str, query: str) -> list[dict]:
        cursor = (
            self._db[CORPUS_CHUNKS]
            .find(
                {"corpus_id": corpus_id, "$text": {"$search": query}},
                {"score": {"$meta": "textScore"}, "embedding": 0},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(self._top_k)
        )
        results = []
        async for doc in cursor:
            results.append(doc)
        return results

    async def _corpus_tokens(self, corpus_id: str) -> int:
        total = 0
        async for doc in self._db[CORPUS_CHUNKS].find(
            {"corpus_id": corpus_id}, {"text": 1}
        ):
            total += _count_tokens(doc.get("text", ""))
        return total
