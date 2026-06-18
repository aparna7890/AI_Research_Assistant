# ADR 002 — Chunking strategy

**Status:** Accepted

| Content type     | Tokens | Overlap | Notes |
|------------------|--------|---------|-------|
| arxiv_paper      | 400    | 50      | Abstract = 1 chunk; references excluded |
| course_chapter   | 300    | 40      | Code blocks kept intact |
| survey_blog      | 400    | 50      | Code blocks kept intact |
| lab_blog_post    | 300    | 40      | — |

## Rationale
- Abstracts are dense self-contained summaries — splitting loses context
- References are excluded from retrieval (citation noise)
- Code blocks kept intact — mid-block splits produce meaningless fragments
- 1 token ≈ 4 chars (avoids tokenizer dependency at ingest time)
