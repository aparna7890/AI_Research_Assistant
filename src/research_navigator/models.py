"""Shared data models used across all modules."""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ContentType(str, Enum):
    ARXIV_PAPER = "arxiv_paper"
    COURSE_CHAPTER = "course_chapter"
    SURVEY_BLOG = "survey_blog"
    LAB_BLOG_POST = "lab_blog_post"


class AgentRoute(str, Enum):
    CONCEPT_EXPLANATION = "concept_explanation"
    PAPER_DEEP_DIVE = "paper_deep_dive"
    COMPARE_APPROACHES = "compare_approaches"
    RECENT_DEVELOPMENTS = "recent_developments"
    FIND_PAPERS = "find_papers"
    OUT_OF_SCOPE = "out_of_scope"


class DocumentMeta(BaseModel):
    doc_id: str
    content_type: ContentType
    title: str
    authors: list[str]
    year: int
    month: int | None = None
    primary_category: str
    secondary_categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    is_foundational: bool = False
    citation_count: int | None = None
    source_url: str
    local_path: str


class Corpus(BaseModel):
    schema_version: str
    generated_for: str
    documents: list[DocumentMeta]


class ChunkPayload(BaseModel):
    doc_id: str
    content_type: ContentType
    title: str
    authors: list[str]
    year: int
    month: int | None
    primary_category: str
    secondary_categories: list[str]
    tags: list[str]
    is_foundational: bool
    citation_count: int | None
    source_url: str
    section_title: str
    section_index: int
    chunk_index: int
    content_hash: str
    chunk_text: str


class Chunk(BaseModel):
    id: str
    text: str
    payload: ChunkPayload


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    payload: ChunkPayload


class QueryFilters(BaseModel):
    content_types: list[ContentType] | None = None
    tags: list[str] | None = None
    year_gte: int | None = None
    year_lte: int | None = None
    is_foundational: bool | None = None
    doc_ids: list[str] | None = None


class Citation(BaseModel):
    index: int
    doc_id: str
    title: str
    authors_display: str
    year: int
    source_label: str
    section_title: str
    url: str


class GeneratedAnswer(BaseModel):
    answer_text: str
    citations: list[Citation]
    route: AgentRoute
    refused: bool = False
    refusal_reason: str | None = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class NavigatorState(BaseModel):
    query: str
    route: AgentRoute | None = None
    filters: QueryFilters | None = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    answer: GeneratedAnswer | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
