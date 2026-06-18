"""
M2 query pipeline — the main entry point for answering questions.

Every agent route in M3 calls ask() with different parameters.
"""
from __future__ import annotations

import structlog
from qdrant_client import QdrantClient

from research_navigator.generate.generator import generate_answer
from research_navigator.ingest.qdrant_store import get_client
from research_navigator.models import AgentRoute, GeneratedAnswer, QueryFilters
from research_navigator.retrieve.filter_inference import infer_filters
from research_navigator.retrieve.hybrid import retrieve
from research_navigator.settings import Settings, settings as default_settings

logger = structlog.get_logger(__name__)


def ask(
    query: str,
    route: AgentRoute = AgentRoute.CONCEPT_EXPLANATION,
    extra_filters: QueryFilters | None = None,
    settings: Settings = default_settings,
    client: QdrantClient | None = None,
) -> GeneratedAnswer:
    """
    Full M2 pipeline: query → filters → retrieval → generation → answer.

    Args:
        query:         The user's natural language question
        route:         Which agent route triggered this call (for metadata)
        extra_filters: Additional filters set by the agent (e.g. recency for
                       RecentDevelopments, doc_ids for PaperDeepDive)
        settings:      Config — injected for testability
        client:        Qdrant client — injected for testability
    """
    if client is None:
        client = get_client(settings)

    # Step 1 — infer filters from query text
    inferred = infer_filters(query)

    # Step 2 — merge with any agent-supplied extra filters
    # Extra filters take precedence over inferred ones
    merged = _merge_filters(inferred, extra_filters)

    logger.info("pipeline_start", query=query[:80], route=route, filters=merged.model_dump(exclude_none=True))

    # Step 3 — hybrid retrieval
    chunks = retrieve(query, merged, settings, client)

    # Step 4 — generate cited answer (or refuse)
    answer = generate_answer(query, chunks, route, settings)

    logger.info(
        "pipeline_complete",
        route=route,
        chunks_retrieved=len(chunks),
        refused=answer.refused,
        citations=len(answer.citations),
    )
    return answer


def _merge_filters(inferred: QueryFilters, extra: QueryFilters | None) -> QueryFilters:
    """
    Merge inferred filters with agent-supplied extra filters.
    Extra filters win on conflict (agent knows best for its route).
    """
    if extra is None:
        return inferred

    return QueryFilters(
        doc_ids=extra.doc_ids or inferred.doc_ids,
        content_types=extra.content_types or inferred.content_types,
        tags=extra.tags or inferred.tags,
        year_gte=extra.year_gte if extra.year_gte is not None else inferred.year_gte,
        year_lte=extra.year_lte if extra.year_lte is not None else inferred.year_lte,
        is_foundational=extra.is_foundational if extra.is_foundational is not None else inferred.is_foundational,
    )