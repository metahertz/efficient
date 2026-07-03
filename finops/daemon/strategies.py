from dataclasses import dataclass, field


@dataclass(frozen=True)
class Strategy:
    name:             str
    order:            tuple[str, ...]
    composition:      str
    cache_key:        str
    short_circuit_on: tuple[str, ...] = ()
    overrides:        dict = field(default_factory=dict)


COMPOSE_THEN_COMPRESS = Strategy(
    name="compose_then_compress",
    order=("semantic_cache", "codebase_graph", "hybrid_retrieval", "agent_memory", "context_compressor"),
    composition="compose",
    cache_key="prompt+scope",
    short_circuit_on=("semantic_cache",),
)

CACHE_FIRST_AGGRESSIVE = Strategy(
    name="cache_first_aggressive",
    order=("semantic_cache", "codebase_graph", "hybrid_retrieval", "agent_memory"),
    composition="compose",
    cache_key="prompt",
    short_circuit_on=("semantic_cache",),
    overrides={"semantic_cache": {"similarity_threshold": 0.85}},
)

_REGISTRY = {s.name: s for s in (COMPOSE_THEN_COMPRESS, CACHE_FIRST_AGGRESSIVE)}


def get_strategy(name: str | None) -> Strategy:
    return _REGISTRY.get(name or "", COMPOSE_THEN_COMPRESS)


def list_strategies() -> list[str]:
    return list(_REGISTRY)
