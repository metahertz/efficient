import pytest
import dataclasses
from efficient.daemon.strategies import (
    Strategy, get_strategy, list_strategies,
    COMPOSE_THEN_COMPRESS, CACHE_FIRST_AGGRESSIVE,
)


def test_get_strategy_default_for_none():
    assert get_strategy(None) is COMPOSE_THEN_COMPRESS


def test_get_strategy_default_for_unknown():
    assert get_strategy("does-not-exist") is COMPOSE_THEN_COMPRESS


def test_get_strategy_returns_named():
    assert get_strategy("cache_first_aggressive") is CACHE_FIRST_AGGRESSIVE


def test_both_builtins_listed():
    names = list_strategies()
    assert "compose_then_compress" in names
    assert "cache_first_aggressive" in names


def test_strategy_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        COMPOSE_THEN_COMPRESS.name = "x"


def test_default_strategy_order_and_policy():
    s = COMPOSE_THEN_COMPRESS
    assert s.order[0] == "semantic_cache"
    assert s.order[-1] == "context_compressor"
    assert s.composition == "compose"
    assert s.cache_key == "prompt+scope"
    assert s.short_circuit_on == ("semantic_cache",)


def test_aggressive_strategy_overrides():
    s = CACHE_FIRST_AGGRESSIVE
    assert s.cache_key == "prompt"
    assert "context_compressor" not in s.order
    assert s.overrides["semantic_cache"]["similarity_threshold"] == 0.85


def test_config_default_has_strategy():
    from efficient.daemon.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["strategy"] == "compose_then_compress"
