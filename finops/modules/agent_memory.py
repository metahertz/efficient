import os
import time
from datetime import datetime, timezone, timedelta

from langchain_anthropic import ChatAnthropic
from openai import OpenAI
from motor.motor_asyncio import AsyncIOMotorDatabase

from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult
from finops.modules.embeddings import embed_query, embed_documents
from finops.db.collections import WORKING_MEMORY, EPISODIC_MEMORY, SEMANTIC_MEMORY
from finops.db.vector import vector_search

_FACT_PROMPT = (
    "Extract factual statements from this conversation. "
    "Return one fact per line. Return empty string if no facts.\n\n"
    "Turn: {turn}\nResponse: {response}\n\nFacts:"
)

_DEDUP_THRESHOLD = 0.95


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class AgentMemory(BaseModule):
    name = "agent_memory"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._working_turns = config.get("working_memory_turns", 20)
        self._episodic_ttl = config.get("episodic_ttl_days", 30)
        self._semantic_ttl = config.get("semantic_ttl_days", 90)
        fe = config.get("fact_extraction", {})
        self._fact_provider = fe.get("provider", "anthropic")
        self._fact_base_url = fe.get("base_url", "")
        self._fact_model = fe.get("model", "")

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()
        working = await self._get_working_memory(request.agent_id)
        episodic = await self._get_episodic_memory(request.agent_id, request.prompt)
        semantic = await self._get_semantic_memory(request.agent_id, request.prompt)

        if not working and not episodic and not semantic:
            return request, ModuleResult(
                module=self.name, tokens_in=0, tokens_out=0, tokens_saved=0,
                latency_ms=(time.perf_counter() - t0) * 1000, detail="no memory found",
            )

        memory_ctx = self._format_memory(working, episodic, semantic)
        baseline_tokens = await self._full_history_tokens(request.agent_id)
        tokens_added = _count_tokens(memory_ctx)
        tokens_in = _count_tokens(request.context)
        section = "## Memory\n" + memory_ctx
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
            detail=f"working={len(working)}, episodic={len(episodic)}, semantic={len(semantic)}",
            tokens_added=tokens_added,
            baseline_tokens=baseline_tokens,
        )

    async def store_turn(self, agent_id: str, session_id: str, turn: str, response: str) -> None:
        now = datetime.now(timezone.utc)
        await self._db[WORKING_MEMORY].update_one(
            {"agent_id": agent_id, "session_id": session_id},
            {
                "$push": {"messages": {"$each": [
                    {"role": "user", "content": turn, "timestamp": now},
                    {"role": "assistant", "content": response, "timestamp": now},
                ]}},
                "$set": {"updated_at": now},
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
        )
        await self._extract_and_store_facts(agent_id, turn, response)

    async def _get_working_memory(self, agent_id: str) -> list[dict]:
        doc = await self._db[WORKING_MEMORY].find_one({"agent_id": agent_id})
        if not doc:
            return []
        messages = doc.get("messages", [])
        keep = self._working_turns * 2
        return messages[-keep:]

    async def _full_history_tokens(self, agent_id: str) -> int:
        total = 0
        async for doc in self._db[WORKING_MEMORY].find({"agent_id": agent_id}, {"messages": 1}):
            for m in doc.get("messages", []):
                total += _count_tokens(m.get("content", ""))
        return total

    async def _get_episodic_memory(self, agent_id: str, query: str) -> list[str]:
        embedding = embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "episodic_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 3,
                    "filter": {"agent_id": {"$eq": agent_id}},
                }
            },
        ]
        results = []
        for doc in await vector_search(self._db[EPISODIC_MEMORY], pipeline):
            results.append(doc["content"])
        return results

    async def _get_semantic_memory(self, agent_id: str, query: str) -> list[str]:
        embedding = embed_query(query)
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "semantic_vector_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 20,
                    "limit": 5,
                    "filter": {"agent_id": {"$eq": agent_id}},
                }
            },
        ]
        results = []
        for doc in await vector_search(self._db[SEMANTIC_MEMORY], pipeline):
            results.append(doc["fact"])
        return results

    def _extract_facts(self, turn: str, response: str) -> list[str]:
        provider = self._fact_provider
        if provider == "off":
            return []
        prompt = _FACT_PROMPT.format(turn=turn, response=response)
        raw = ""
        if provider == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                return []
            model = self._fact_model or "claude-haiku-4-5-20251001"
            llm = ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY", ""), max_tokens=256)
            raw = llm.invoke(prompt).content or ""
        elif provider == "local":
            if not self._fact_base_url or not self._fact_model:
                return []
            client = OpenAI(base_url=self._fact_base_url, api_key=os.getenv("OPENAI_API_KEY", "not-needed"))
            resp = client.chat.completions.create(
                model=self._fact_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
            )
            raw = resp.choices[0].message.content or ""
        else:
            return []
        raw = raw.strip()
        return [f.strip() for f in raw.splitlines() if f.strip()]

    async def _extract_and_store_facts(self, agent_id: str, turn: str, response: str) -> None:
        facts = self._extract_facts(turn, response)
        if not facts:
            return
        fact_embeddings = embed_documents(facts)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self._semantic_ttl)
        for fact, emb in zip(facts, fact_embeddings):
            dedup_pipeline = [
                {
                    "$vectorSearch": {
                        "index": "semantic_vector_index",
                        "path": "embedding",
                        "queryVector": emb,
                        "numCandidates": 10,
                        "limit": 1,
                        "filter": {"agent_id": {"$eq": agent_id}},
                    }
                },
                {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
                {"$match": {"_score": {"$gte": _DEDUP_THRESHOLD}}},
            ]
            existing = None
            for doc in await vector_search(self._db[SEMANTIC_MEMORY], dedup_pipeline):
                existing = doc
                break
            if existing:
                await self._db[SEMANTIC_MEMORY].update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"fact": fact, "updated_at": now, "expires_at": expires_at}},
                )
            else:
                await self._db[SEMANTIC_MEMORY].insert_one({
                    "agent_id": agent_id,
                    "fact": fact,
                    "embedding": emb,
                    "confidence": 1.0,
                    "source_session": None,
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": expires_at,
                })

    def _format_memory(self, working: list[dict], episodic: list[str], semantic: list[str]) -> str:
        parts = []
        if semantic:
            parts.append("### Known Facts\n" + "\n".join(f"- {f}" for f in semantic))
        if episodic:
            parts.append("### Recent Context\n" + "\n".join(f"- {e}" for e in episodic))
        if working:
            msgs = "\n".join(f"{m['role']}: {m['content']}" for m in working)
            parts.append("### Conversation\n" + msgs)
        return "\n\n".join(parts)
