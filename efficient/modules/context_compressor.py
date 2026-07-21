import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone

from llmlingua import PromptCompressor
from motor.motor_asyncio import AsyncIOMotorDatabase

from efficient.modules._base import BaseModule, OptimizeRequest, ModuleResult
from efficient.db.collections import COMPRESSION_STATS

_MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
_compressor: PromptCompressor | None = None
_compressor_lock = threading.Lock()


def _get_compressor() -> PromptCompressor:
    global _compressor
    if _compressor is None:
        with _compressor_lock:
            if _compressor is None:
                _compressor = PromptCompressor(
                    model_name=_MODEL_NAME,
                    use_llmlingua2=True,
                    device_map="cpu",
                )
    return _compressor


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ContextCompressor(BaseModule):
    name = "context_compressor"

    def __init__(self, db: AsyncIOMotorDatabase, config: dict):
        super().__init__()
        self._db = db
        self._threshold = config.get("token_threshold", 8000)
        self._target_ratio = config.get("target_ratio", 4.0)

    def is_enabled(self) -> bool:
        return True

    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        t0 = time.perf_counter()
        original_tokens = _count_tokens(request.context)

        if original_tokens < self._threshold:
            return request, ModuleResult(
                module=self.name, tokens_in=original_tokens, tokens_out=original_tokens,
                tokens_saved=0, latency_ms=(time.perf_counter() - t0) * 1000,
                detail=f"bypass (tokens={original_tokens} < threshold={self._threshold})",
                baseline_tokens=original_tokens,
            )

        rate = 1.0 / self._target_ratio
        result = await asyncio.to_thread(
            lambda: _get_compressor().compress_prompt(
                [request.context],
                rate=rate,
                force_tokens=["\n", "?"],
            )
        )
        compressed = result["compressed_prompt"]
        compressed_tokens = _count_tokens(compressed)
        tokens_saved = max(0, original_tokens - compressed_tokens)
        latency = (time.perf_counter() - t0) * 1000
        ratio = original_tokens / max(1, compressed_tokens)

        await self._db[COMPRESSION_STATS].insert_one({
            "request_id": str(uuid.uuid4()),
            "framework": request.framework,
            "model": _MODEL_NAME,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "ratio": ratio,
            "latency_ms": latency,
            "created_at": datetime.now(timezone.utc),
        })

        new_req = OptimizeRequest(
            prompt=request.prompt,
            context=compressed,
            agent_id=request.agent_id,
            framework=request.framework,
            corpus_id=request.corpus_id,
        )
        return new_req, ModuleResult(
            module=self.name,
            tokens_in=original_tokens,
            tokens_out=compressed_tokens,
            tokens_saved=tokens_saved,
            latency_ms=latency,
            detail=f"compressed {ratio:.1f}x ({original_tokens}->{compressed_tokens} tokens)",
            baseline_tokens=original_tokens,
        )
