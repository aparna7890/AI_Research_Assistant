"""Content-type-aware chunker."""
from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass
from research_navigator.ingest.parsers import Section
from research_navigator.models import Chunk, ChunkPayload, ContentType, DocumentMeta

_CHARS_PER_TOKEN = 4


@dataclass
class ChunkConfig:
    max_tokens: int
    overlap_tokens: int


_CONFIGS: dict[ContentType, ChunkConfig] = {
    ContentType.ARXIV_PAPER:    ChunkConfig(max_tokens=400, overlap_tokens=50),
    ContentType.COURSE_CHAPTER: ChunkConfig(max_tokens=300, overlap_tokens=40),
    ContentType.SURVEY_BLOG:    ChunkConfig(max_tokens=400, overlap_tokens=50),
    ContentType.LAB_BLOG_POST:  ChunkConfig(max_tokens=300, overlap_tokens=40),
}


def chunk_document(doc: DocumentMeta, sections: list[Section]) -> list[Chunk]:
    config = _CONFIGS[doc.content_type]
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        if section.is_references:
            continue

        if section.is_abstract:
            chunks.append(_make_chunk(doc, section.text, section.title, section.index, chunk_index))
            chunk_index += 1
            continue

        for part in _split_preserve_code(section.text):
            for sub in _split_text(part, config):
                if sub.strip():
                    chunks.append(_make_chunk(doc, sub, section.title, section.index, chunk_index))
                    chunk_index += 1

    return chunks


def _make_chunk(doc: DocumentMeta, text: str, section_title: str, section_index: int, chunk_index: int) -> Chunk:
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    chunk_id = f"{doc.doc_id}-{chunk_index}-{content_hash[:8]}"
    payload = ChunkPayload(
        doc_id=doc.doc_id,
        content_type=doc.content_type,
        title=doc.title,
        authors=doc.authors,
        year=doc.year,
        month=doc.month,
        primary_category=doc.primary_category,
        secondary_categories=doc.secondary_categories,
        tags=doc.tags,
        is_foundational=doc.is_foundational,
        citation_count=doc.citation_count,
        source_url=doc.source_url,
        section_title=section_title,
        section_index=section_index,
        chunk_index=chunk_index,
        content_hash=content_hash,
        chunk_text=text,
    )
    return Chunk(id=chunk_id, text=text, payload=payload)


def _split_text(text: str, config: ChunkConfig) -> list[str]:
    max_chars = config.max_tokens * _CHARS_PER_TOKEN
    overlap_chars = config.overlap_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        split_pos = text.rfind("\n\n", start, end)
        if split_pos == -1:
            split_pos = _rfind_sentence_end(text, start, end)
        if split_pos == -1:
            split_pos = end
        chunks.append(text[start:split_pos].strip())
        start = max(start + 1, split_pos - overlap_chars)
    return [c for c in chunks if c.strip()]


def _split_preserve_code(text: str) -> list[str]:
    parts = re.compile(r"(```[\s\S]*?```)", re.MULTILINE).split(text)
    return [p for p in parts if p.strip()]


def _rfind_sentence_end(text: str, start: int, end: int) -> int:
    for punct in (". ", "? ", "! ", ".\n"):
        pos = text.rfind(punct, start, end)
        if pos != -1:
            return pos + len(punct)
    return -1
