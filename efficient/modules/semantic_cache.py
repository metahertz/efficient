import asyncio
import hashlib
import time
from datetime import datetime, timezone, timedelta

from motor.motor_asyncio import AsyncIOMotorDatabase

from efficient.modules._base import BaseModule, OptimizeRequest, ModuleResult
from efficient.modules.embeddings import embed_query, embed_documents
from efficient.db.collections import CACHE_ENTRIES
from efficient.db.vector import vector_search


class SemanticCache(BaseModule):
    name = "semantic_cache"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._threshold = config.get("similarity_threshold", 0.80)
        self._ttl_hours = config.get("ttl_hours", 168)
        self._cache_key = config.get("cache_key", "prompt+scope")

    def is_enabled(self) -> bool:
        return True

    def _key_material(self, request: OptimizeRequest) -> str:
        if self._cache_key == "prompt":
            return request.prompt
        return f"{request.prompt}|agent={request.agent_id}|corpus={request.corpus_id or ''}"

    def _key_material_raw(self, prompt: str, agent_id: str, corpus_id: str) -> str:
        if self._cache_key == "prompt":
            return prompt
        return f"{prompt}|agent={agent_id}|corpus={corpus_id or ''}"

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()
        key = self._key_material(request)
        prompt_hash = hashlib.sha256(key.encode()).hexdigest()

        entry = await self._db[CACHE_ENTRIES].find_one({"prompt_hash": prompt_hash})
        if entry:
            await self._db[CACHE_ENTRIES].update_one(
                {"_id": entry["_id"]},
                {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}},
            )
            saved = entry.get("tokens_saved", 0)
            cached_req = OptimizeRequest(
                prompt=request.prompt,
                context=entry["response"],
                agent_id=request.agent_id,
                framework=request.framework,
                corpus_id=request.corpus_id,
            )
            return cached_req, ModuleResult(
                module=self.name,
                tokens_in=saved,
                tokens_out=0,
                tokens_saved=saved,
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail="exact hash hit",
                short_circuit=True,
                baseline_tokens=saved,
            )

        embedding = await asyncio.to_thread(embed_query, key)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "cache_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 1,
                }
            },
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"_score": {"$gte": self._threshold}}},
        ]
        for doc in await vector_search(self._db[CACHE_ENTRIES], pipeline):
            await self._db[CACHE_ENTRIES].update_one(
                {"_id": doc["_id"]},
                {"$inc": {"hit_count": 1}, "$set": {"last_hit_at": datetime.now(timezone.utc)}},
            )
            score = doc["_score"]
            saved = doc.get("tokens_saved", 0)
            cached_req = OptimizeRequest(
                prompt=request.prompt,
                context=doc["response"],
                agent_id=request.agent_id,
                framework=request.framework,
                corpus_id=request.corpus_id,
            )
            return cached_req, ModuleResult(
                module=self.name,
                tokens_in=saved,
                tokens_out=0,
                tokens_saved=saved,
                latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"semantic hit (similarity={score:.3f})",
                short_circuit=True,
                baseline_tokens=saved,
            )

        return request, ModuleResult(
            module=self.name,
            tokens_in=0, tokens_out=0, tokens_saved=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            detail="cache miss",
        )

    async def store(
        self,
        prompt: str,
        response: str,
        framework: str,
        model: str,
        tokens_saved: int,
        agent_id: str = "",
        corpus_id: str = "",
    ) -> None:
        key = self._key_material_raw(prompt, agent_id, corpus_id)
        prompt_hash = hashlib.sha256(key.encode()).hexdigest()
        embedding = (await asyncio.to_thread(embed_documents, [key]))[0]
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self._ttl_hours)
        await self._db[CACHE_ENTRIES].update_one(
            {"prompt_hash": prompt_hash},
            {"$setOnInsert": {
                "prompt_hash": prompt_hash,
                "embedding": embedding,
                "prompt_preview": prompt[:200],
                "response": response,
                "framework": framework,
                "model": model,
                "tokens_saved": tokens_saved,
                "hit_count": 0,
                "created_at": now,
                "last_hit_at": None,
                "expires_at": expires_at,
            }},
            upsert=True,
        )
