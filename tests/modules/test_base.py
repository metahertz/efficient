import pytest
from finops.modules._base import BaseModule, ModuleResult, OptimizeRequest


class PassthroughModule(BaseModule):
    name = "passthrough"

    async def process(
        self, request: OptimizeRequest
    ) -> tuple[OptimizeRequest, ModuleResult]:
        result = ModuleResult(
            module=self.name, tokens_in=10, tokens_out=10,
            tokens_saved=0, latency_ms=0.5, detail="no-op",
        )
        return request, result

    def is_enabled(self) -> bool:
        return True


def test_module_subclass_instantiates():
    mod = PassthroughModule()
    assert mod.is_enabled()
    assert mod.name == "passthrough"


async def test_process_returns_original_request_and_result():
    mod = PassthroughModule()
    req = OptimizeRequest(
        prompt="hello", context="ctx", agent_id="agent1", framework="test"
    )
    req_out, result = await mod.process(req)
    assert req_out is req
    assert result.module == "passthrough"
    assert result.tokens_saved == 0
    assert result.latency_ms == 0.5


def test_cannot_instantiate_base_directly():
    with pytest.raises(TypeError):
        BaseModule()


def test_optimize_request_corpus_id_defaults_to_none():
    req = OptimizeRequest(prompt="p", context="c", agent_id="a", framework="f")
    assert req.corpus_id is None


def test_module_result_fields():
    r = ModuleResult(
        module="test", tokens_in=100, tokens_out=50,
        tokens_saved=50, latency_ms=12.3, detail="compressed"
    )
    assert r.tokens_saved == 50
    assert r.detail == "compressed"


def test_subclass_without_name_raises():
    class Unnamed(BaseModule):
        async def process(self, request):
            return request, None
        def is_enabled(self):
            return True
    with pytest.raises(TypeError, match="must define a non-empty 'name'"):
        Unnamed()


def test_module_result_has_honest_metric_defaults():
    r = ModuleResult(
        module="x", tokens_in=0, tokens_out=0,
        tokens_saved=0, latency_ms=0.0, detail="",
    )
    assert r.short_circuit is False
    assert r.tokens_added == 0
    assert r.baseline_tokens == 0


def test_module_result_honest_metric_fields_can_be_set():
    r = ModuleResult(
        module="x", tokens_in=0, tokens_out=0,
        tokens_saved=0, latency_ms=0.0, detail="",
        short_circuit=True, tokens_added=42, baseline_tokens=1000,
    )
    assert r.short_circuit is True
    assert r.tokens_added == 42
    assert r.baseline_tokens == 1000
