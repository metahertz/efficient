import pytest
from finops.modules._base import BaseModule, OptimizeRequest, ModuleResult


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
