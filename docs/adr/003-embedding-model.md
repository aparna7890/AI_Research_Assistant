# ADR 003 — Embedding model

**Status:** Accepted

## Decision
`sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim, ~80MB)

## Rationale
- No API key, no cost, no rate limits at ingest time
- Runs fully local — reproducible
- Easily swappable via `EMBEDDING_MODEL` env var

## Upgrade path
Set `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5` and `EMBEDDING_DIM=768` for better quality.
