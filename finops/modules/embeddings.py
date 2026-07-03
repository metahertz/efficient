import os
from sentence_transformers import SentenceTransformer

_MODEL_ID = "voyageai/voyage-4-nano"
_DIM = 1024
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        name = os.getenv("FINOPS_EMBEDDING_MODEL", _MODEL_ID)
        _model = SentenceTransformer(name, trust_remote_code=True, truncate_dim=_DIM)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model()
    vecs = model.encode_document(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    model = _get_model()
    vec = model.encode_query([text], normalize_embeddings=True)[0]
    return vec.tolist()


def reset_model() -> None:
    global _model
    _model = None
