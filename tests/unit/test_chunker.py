"""Unit tests for chunker."""
from __future__ import annotations
from research_navigator.ingest.chunker import chunk_document, _split_text, ChunkConfig
from research_navigator.ingest.parsers import Section
from research_navigator.models import ContentType, DocumentMeta


def _doc(ct: ContentType = ContentType.ARXIV_PAPER) -> DocumentMeta:
    return DocumentMeta(
        doc_id="test-doc", content_type=ct, title="Test", authors=["A"],
        year=2024, primary_category="cs.CL", tags=["LLM"],
        source_url="https://example.com", local_path="documents/arxiv/test.pdf",
    )


def test_short_text_single_chunk() -> None:
    assert _split_text("hello world", ChunkConfig(400, 50)) == ["hello world"]


def test_long_text_multiple_chunks() -> None:
    result = _split_text("word " * 400, ChunkConfig(100, 10))
    assert len(result) > 1


def test_abstract_is_one_chunk() -> None:
    sections = [Section(title="Abstract", index=0, text="X " * 50, is_abstract=True)]
    chunks = chunk_document(_doc(), sections)
    assert len(chunks) == 1


def test_references_excluded() -> None:
    sections = [
        Section(title="Intro", index=0, text="Some intro text. " * 10),
        Section(title="References", index=1, text="[1] Vaswani", is_references=True),
    ]
    titles = [c.payload.section_title for c in chunk_document(_doc(), sections)]
    assert "References" not in titles


def test_chunk_ids_deterministic() -> None:
    sections = [Section(title="S", index=0, text="Hello world " * 10)]
    assert [c.id for c in chunk_document(_doc(), sections)] == \
           [c.id for c in chunk_document(_doc(), sections)]


def test_metadata_carried_through() -> None:
    sections = [Section(title="S", index=0, text="Some text " * 5)]
    chunk = chunk_document(_doc(ContentType.SURVEY_BLOG), sections)[0]
    assert chunk.payload.doc_id == "test-doc"
    assert chunk.payload.content_type == ContentType.SURVEY_BLOG
