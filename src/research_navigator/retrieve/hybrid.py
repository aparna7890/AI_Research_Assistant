"""
Hybrid retrieval: dense + sparse with Reciprocal Rank Fusion.

Why RRF over weighted sum:
Dense and sparse scores are on completely different scales.
RRF only cares about rank position, not raw score — so it's
safe to combine them without any calibration.
Formula: score(d) = 1/(60 + rank_dense) + 1/(60 + rank_sparse)
"""
from __future__ import annotations

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from research_navigator.ingest.embedder import embed_texts
from research_navigator.ingest.sparse import build_sparse_vector
from research_navigator.models import ChunkPayload, QueryFilters, RetrievedChunk
from research_navigator.settings import Settings

logger = structlog.get_logger(__name__)

_RRF_K = 60
_DENSE = "dense"
_SPARSE = "sparse"


def retrieve(
    query: str,
    filters: QueryFilters,
    settings: Settings,
    client: QdrantClient,
) -> list[RetrievedChunk]:
    """
    Run hybrid search and return top-k chunks ranked by RRF score.
    """
    qdrant_filter = _build_filter(filters)
    fetch_limit = settings.retrieval_top_k * 2  # fetch extra, trim after fusion

    # --- Dense search ---
    dense_vec = embed_texts([query], settings.embedding_model)[0]
    dense_results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=qm.NamedVector(name=_DENSE, vector=dense_vec),
        query_filter=qdrant_filter,
        limit=fetch_limit,
        with_payload=True,
        with_vectors=False,
    )

    # --- Sparse search ---
    sparse_vec = build_sparse_vector(query)
    sparse_results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=qm.NamedSparseVector(
            name=_SPARSE,
            vector=qm.SparseVector(
                indices=sparse_vec["indices"],  # type: ignore[arg-type]
                values=sparse_vec["values"],    # type: ignore[arg-type]
            ),
        ),
        query_filter=qdrant_filter,
        limit=fetch_limit,
        with_payload=True,
        with_vectors=False,
    )

    logger.debug(
        "raw_retrieval",
        dense_hits=len(dense_results),
        sparse_hits=len(sparse_results),
        query=query[:60],
    )

    # --- RRF fusion ---
    fused = _rrf_fuse(dense_results, sparse_results)
    top_k = fused[: settings.retrieval_top_k]

    chunks = [
        RetrievedChunk(
            chunk_id=point_id,
            score=score,
            payload=ChunkPayload(**payload),
        )
        for point_id, score, payload in top_k
    ]

    logger.info(
        "retrieval_complete",
        query=query[:60],
        results=len(chunks),
        top_score=round(chunks[0].score, 4) if chunks else 0,
    )
    return chunks


def _rrf_fuse(
    dense: list[qm.ScoredPoint],
    sparse: list[qm.ScoredPoint],
    k: int = _RRF_K,
) -> list[tuple[str, float, dict]]:  # type: ignore[type-arg]
    """Merge two ranked lists with RRF. Returns (id, score, payload) tuples."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}  # type: ignore[type-arg]

    for rank, point in enumerate(dense, start=1):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if point.payload:
            payloads[pid] = point.payload

    for rank, point in enumerate(sparse, start=1):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if point.payload and pid not in payloads:
            payloads[pid] = point.payload

    ranked = sorted(scores, key=lambda p: scores[p], reverse=True)
    return [(pid, scores[pid], payloads.get(pid, {})) for pid in ranked]


def _build_filter(filters: QueryFilters) -> qm.Filter | None:
    """Convert QueryFilters → Qdrant Filter. Returns None if no filters set."""
    must: list[qm.Condition] = []

    if filters.doc_ids:
        must.append(qm.FieldCondition(
            key="doc_id", match=qm.MatchAny(any=filters.doc_ids)
        ))
    if filters.content_types:
        must.append(qm.FieldCondition(
            key="content_type",
            match=qm.MatchAny(any=[ct.value for ct in filters.content_types]),
        ))
    if filters.tags:
        for tag in filters.tags:
            must.append(qm.FieldCondition(
                key="tags", match=qm.MatchValue(value=tag)
            ))
    if filters.year_gte is not None:
        must.append(qm.FieldCondition(
            key="year", range=qm.Range(gte=filters.year_gte)
        ))
    if filters.year_lte is not None:
        must.append(qm.FieldCondition(
            key="year", range=qm.Range(lte=filters.year_lte)
        ))
    if filters.is_foundational is not None:
        must.append(qm.FieldCondition(
            key="is_foundational",
            match=qm.MatchValue(value=filters.is_foundational),
        ))

    return qm.Filter(must=must) if must else None