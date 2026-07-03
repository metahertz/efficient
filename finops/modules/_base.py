from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OptimizeRequest:
    prompt:    str
    context:   str
    agent_id:  str
    framework: str
    corpus_id: str | None = None


@dataclass
class ModuleResult:
    module:          str
    tokens_in:       int
    tokens_out:      int
    tokens_saved:    int
    latency_ms:      float
    detail:          str
    short_circuit:   bool = field(default=False)
    tokens_added:    int = field(default=0)
    baseline_tokens: int = field(default=0)


class BaseModule(ABC):
    name: str = ""

    def __init__(self):
        if not self.__class__.name:
            raise TypeError(
                f"{self.__class__.__name__} must define a non-empty 'name' class attribute"
            )

    @abstractmethod
    async def process(self, request: OptimizeRequest) -> tuple[OptimizeRequest, ModuleResult]:
        ...

    @abstractmethod
    def is_enabled(self) -> bool:
        ...
