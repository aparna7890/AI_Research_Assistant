"""Qdrant collection management and upsert."""
from __future__ import annotations
import uuid
import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from research_navigator.models import Chunk
from research_navigator.settings import Settings

logger = structlog.get_logger(__name__)

_DENSE = "dense"
_SPARSE = "sparse"

_INDEXED_FIELDS = [
    ("content_type",     qm.PayloadSchemaType.KEYWORD),
    ("year",             qm.PayloadSchemaType.INTEGER),
    ("is_foundational",  qm.PayloadSchemaType.BOOL),
    ("primary_category", qm.PayloadSchemaType.KEYWORD),
    ("doc_id",           qm.PayloadSchemaType.KEYWORD),
    ("tags",             qm.PayloadSchemaType.KEYWORD),
]


def get_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60)


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    name = settings.qdrant_collection
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        logger.info("collection_exists", collection=name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            _DENSE: qm.VectorParams(size=settings.embedding_dim, distance=qm.Distance.COSINE),
        },
        sparse_vectors_config={
            _SPARSE: qm.SparseVectorParams(index=qm.SparseIndexParams(on_disk=False)),
        },
    )
    for field, schema in _INDEXED_FIELDS:
        client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
    logger.info("collection_created", collection=name)


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    dense_vectors: list[list[float]],
    sparse_vectors: list[dict[str, list[int] | list[float]]],
    batch_size: int = 64,
) -> None:
    for i in range(0, len(chunks), batch_size):
        bc = chunks[i:i + batch_size]
        bd = dense_vectors[i:i + batch_size]
        bs = sparse_vectors[i:i + batch_size]
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c.id)),
                vector={
                    _DENSE: d,
                    _SPARSE: qm.SparseVector(indices=s["indices"], values=s["values"]),  # type: ignore[arg-type]
                },
                payload=c.payload.model_dump(),
            )
            for c, d, s in zip(bc, bd, bs)
        ]
        client.upsert(collection_name=collection, points=points)
        logger.debug("batch_upserted", batch=i // batch_size + 1, count=len(points))


def collection_stats(client: QdrantClient, collection: str) -> dict[str, int]:
    info = client.get_collection(collection)
    counts: dict[str, int] = {"total": info.points_count or 0}
    from research_navigator.models import ContentType
    for ct in ContentType:
        result = client.count(
            collection_name=collection,
            count_filter=qm.Filter(must=[
                qm.FieldCondition(key="content_type", match=qm.MatchValue(value=ct.value))
            ]),
        )
        counts[ct.value] = result.count
    return counts
