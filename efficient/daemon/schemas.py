from pydantic import BaseModel, Field


class OptimizeBody(BaseModel):
    prompt: str = ""
    context: str = ""
    agent_id: str = "default"
    framework: str = "unknown"
    corpus_id: str | None = None
    strategy: str | None = None


class CompleteBody(OptimizeBody):
    provider: str = "anthropic"
    model: str = ""
    session_id: str = "default"


class CacheStoreBody(BaseModel):
    prompt: str = ""
    response: str = ""
    framework: str = "unknown"
    model: str = ""
    tokens_saved: int = 0
    agent_id: str = ""
    corpus_id: str = ""


class MemoryRetrieveBody(BaseModel):
    agent_id: str = "default"
    query: str = ""


class MemoryStoreBody(BaseModel):
    agent_id: str = "default"
    session_id: str = "default"
    turn: str = ""
    response: str = ""


class CodebaseIndexBody(BaseModel):
    repo_id: str = "default"
    path: str = ""


class CodebaseQueryBody(BaseModel):
    repo_id: str = "default"
    query: str = ""
    k: int = Field(5, ge=1, le=50)


class CodebaseIndexFileBody(BaseModel):
    repo_id: str = "default"
    file_path: str = ""
    source: str = ""


class CodebaseReferencesBody(BaseModel):
    repo_id: str = "default"
    symbol: str = ""
