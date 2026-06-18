"""
Citation-grounded answer generation.

Flow:
1. Check if top chunk score is above refusal threshold
2. Build a prompt with retrieved chunks as numbered context
3. Call Claude — instruct it to cite [1],[2] inline
4. Parse the response and build structured Citation objects
5. Return GeneratedAnswer
"""
from __future__ import annotations

import structlog
import anthropic

from research_navigator.models import (
    AgentRoute,
    Citation,
    GeneratedAnswer,
    RetrievedChunk,
)
from research_navigator.settings import Settings

logger = structlog.get_logger(__name__)


def generate_answer(
    query: str,
    chunks: list[RetrievedChunk],
    route: AgentRoute,
    settings: Settings,
) -> GeneratedAnswer:
    """
    Generate a cited answer from retrieved chunks.
    Returns a refusal if chunks are empty or scores are too low.
    """
    # --- Refusal check ---
    if not chunks or chunks[0].score < settings.refusal_threshold:
        logger.info(
            "refusal_triggered",
            query=query[:60],
            top_score=chunks[0].score if chunks else 0,
            threshold=settings.refusal_threshold,
        )
        return GeneratedAnswer(
            answer_text=(
                "I don't have enough relevant material in the corpus to answer "
                "this confidently. Try rephrasing, or ask about a topic covered "
                "in the provided AI/ML research corpus."
            ),
            citations=[],
            route=route,
            refused=True,
            refusal_reason="top_score_below_threshold",
            retrieved_chunks=chunks,
        )

    # --- Build citation map (deduplicate by doc_id) ---
    # Multiple chunks from same doc → single citation entry
    citation_map: dict[str, Citation] = {}
    citation_index = 1

    for chunk in chunks:
        doc_id = chunk.payload.doc_id
        if doc_id not in citation_map:
            citation_map[doc_id] = Citation(
                index=citation_index,
                doc_id=doc_id,
                title=chunk.payload.title,
                authors_display=_format_authors(chunk.payload.authors),
                year=chunk.payload.year,
                source_label=_source_label(chunk.payload.doc_id, chunk.payload.content_type.value),
                section_title=chunk.payload.section_title,
                url=chunk.payload.source_url,
            )
            citation_index += 1

    # --- Build context block for prompt ---
    context_lines: list[str] = []
    for chunk in chunks:
        idx = citation_map[chunk.payload.doc_id].index
        context_lines.append(
            f"[{idx}] {chunk.payload.title} ({chunk.payload.year})\n"
            f"Section: {chunk.payload.section_title}\n"
            f"{chunk.text}\n"
        )
    context_block = "\n---\n".join(context_lines)

    # --- Prompt ---
    system_prompt = """You are an AI research assistant helping learners understand AI/ML concepts.
Answer using ONLY the provided context. After every factual claim, add an inline citation marker like [1] or [2].
Be clear and educational. If the context doesn't fully answer the question, say so honestly.
Do NOT make up information not present in the context."""

    user_prompt = f"""Context from the AI research corpus:

{context_block}

Question: {query}

Answer with inline citations:"""

    # --- Call Claude ---
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    logger.info("calling_claude", model=settings.generation_model, chunks=len(chunks))

    message = client.messages.create(
        model=settings.generation_model,
        max_tokens=settings.max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    answer_text = message.content[0].text  # type: ignore[union-attr]

    logger.info(
        "generation_complete",
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )

    return GeneratedAnswer(
        answer_text=answer_text,
        citations=list(citation_map.values()),
        route=route,
        refused=False,
        retrieved_chunks=chunks,
    )


def _format_authors(authors: list[str]) -> str:
    """'Vaswani et al.' for 3+ authors, otherwise join with 'and'."""
    if not authors:
        return "Unknown"
    if len(authors) >= 3:
        return f"{authors[0].split()[-1]} et al."
    return " and ".join(a.split()[-1] for a in authors)


def _source_label(doc_id: str, content_type: str) -> str:
    """Human-readable source label for citation block."""
    if content_type == "arxiv_paper":
        arxiv_id = doc_id.removeprefix("arxiv-")
        return f"arXiv:{arxiv_id}"
    if content_type == "course_chapter":
        return "Hugging Face Learn"
    if content_type == "survey_blog":
        return "Lil'Log"
    if content_type == "lab_blog_post":
        if "anthropic" in doc_id:
            return "Anthropic Blog"
        if "openai" in doc_id:
            return "OpenAI Blog"
        if "deepmind" in doc_id:
            return "Google DeepMind Blog"
    return "Unknown source"