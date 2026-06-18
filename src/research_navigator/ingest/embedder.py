"""Embedding — loads model once via lru_cache."""
from __future__ import annotations
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _get_model(model_name: str):  # type: ignore[return]
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    logger.info("loading_embedding_model", model=model_name)
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], model_name: str) -> list[list[float]]:
    if not texts:
        return []
    model = _get_model(model_name)
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=len(texts) > 20)
    return [v.tolist() for v in vecs]  # type: ignore[union-attr]
