from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OptimizeRequest:
    prompt:    str
    context:   str
    agent_id:  str
    framework: str
    corpus_id: str | None = None


@dataclass
class ModuleResult:
    module:       str
    tokens_in:    int
    tokens_out:   int
    tokens_saved: int
    latency_ms:   float
    detail:       str


class BaseModule(ABC):
    name: str = ""

    @abstractmethod
    async def process(
        self, request: OptimizeRequest
    ) -> tuple[OptimizeRequest, ModuleResult]:
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        ...
