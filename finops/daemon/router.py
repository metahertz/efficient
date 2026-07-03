from finops.daemon.strategies import get_strategy, Strategy
from finops.modules._base import OptimizeRequest, ModuleResult
from finops.modules.codebase_graph import CodebaseGraph
from finops.modules.semantic_cache import SemanticCache
from finops.modules.agent_memory import AgentMemory
from finops.modules.context_compressor import ContextCompressor
from finops.modules.hybrid_retrieval import HybridRetrieval

_MODULE_CLASSES = {
    "codebase_graph":     CodebaseGraph,
    "semantic_cache":     SemanticCache,
    "agent_memory":       AgentMemory,
    "context_compressor": ContextCompressor,
    "hybrid_retrieval":   HybridRetrieval,
}


def _result_dict(r: ModuleResult) -> dict:
    return {
        "module": r.module, "tokens_in": r.tokens_in, "tokens_out": r.tokens_out,
        "tokens_saved": r.tokens_saved, "tokens_added": r.tokens_added,
        "baseline_tokens": r.baseline_tokens, "latency_ms": r.latency_ms, "detail": r.detail,
    }


class ModulePipeline:
    def __init__(self, db, module_configs: dict, strategy: Strategy):
        self._strategy = strategy
        merged = {}
        for name in _MODULE_CLASSES:
            cfg = dict(module_configs.get(name, {}))
            cfg.update(strategy.overrides.get(name, {}))
            merged[name] = cfg
        merged["semantic_cache"]["cache_key"] = strategy.cache_key
        self._modules = {name: cls(db, merged[name]) for name, cls in _MODULE_CLASSES.items()}
        self._enabled = {name: module_configs.get(name, {}).get("enabled", False) for name in _MODULE_CLASSES}

    async def run(self, request: OptimizeRequest) -> dict:
        collected: list[ModuleResult] = []
        for name in self._strategy.order:
            if not self._enabled.get(name, False):
                continue
            request, result = await self._modules[name].process(request)
            collected.append(result)
            if name in self._strategy.short_circuit_on and result.short_circuit:
                return {
                    "optimized_prompt": request.prompt, "optimized_context": request.context,
                    "cache_hit": True, "strategy": self._strategy.name,
                    "tokens_saved": result.tokens_saved,
                    "module_results": [_result_dict(r) for r in collected],
                }
        return {
            "optimized_prompt": request.prompt, "optimized_context": request.context,
            "cache_hit": False, "strategy": self._strategy.name,
            "tokens_saved": sum(r.tokens_saved for r in collected),
            "module_results": [_result_dict(r) for r in collected],
        }
