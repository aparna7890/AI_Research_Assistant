"""
Query understanding: infer Qdrant metadata filters from natural language.

Strategy: rule-based pattern matching. Fast, deterministic, and testable.
No LLM call needed here — simple regex covers 90% of cases.
"""
from __future__ import annotations

import re
from datetime import datetime

import structlog

from research_navigator.models import ContentType, QueryFilters

logger = structlog.get_logger(__name__)

_CURRENT_YEAR = datetime.now().year

# ── Tag patterns ──────────────────────────────────────────────────────────────
# (regex pattern, normalised tag from manifest vocabulary)
_TAG_PATTERNS: list[tuple[str, str]] = [
    (r"\brag\b|retrieval.augment",          "RAG"),
    (r"\bretrieval\b",                       "retrieval"),
    (r"chain.of.thought|\bcot\b",           "chain_of_thought"),
    (r"instruction.tun",                     "instruction_tuning"),
    (r"preference.optim|\bdpo\b|\bkto\b|\bsimpo\b", "preference_optimization"),
    (r"\brlhf\b",                            "RLHF"),
    (r"\brlaif\b",                           "RLAIF"),
    (r"reinforcement.learn|\brl\b",          "RL"),
    (r"mixture.of.expert|\bmoe\b",           "MoE"),
    (r"quantiz|1.bit|bitnet",               "quantization"),
    (r"long.context",                        "long_context"),
    (r"hallucin",                            "safety"),
    (r"alignment|harmless",                  "alignment"),
    (r"safety",                              "safety"),
    (r"interpret|mechanistic",               "interpretability"),
    (r"multimodal|vision.language|\bvlm\b", "multimodal"),
    (r"\bagent\b|tool.use|\breact\b",       "agents"),
    (r"fine.tun|\blora\b|\bpeft\b",         "fine_tuning"),
    (r"transform",                           "transformers"),
    (r"scaling",                             "scaling"),
    (r"\bbert\b|pretraining",               "pretraining"),
    (r"evaluat|benchmark",                   "evaluation"),
    (r"few.shot|prompting",                  "prompting"),
    (r"reasoning",                           "reasoning"),
    (r"open.source|open.weight|\bllama\b",  "open_models"),
    (r"\bllm\b|language.model|\bgpt\b",     "LLM"),
    (r"flash.attention",                     "attention"),
    (r"speculative.decod|inference.optim",  "inference_optimization"),
    (r"survey",                              "survey"),
]

# ── Recency patterns ──────────────────────────────────────────────────────────
_RECENCY_PATTERNS: list[tuple[str, int]] = [
    (r"recent|latest|new|current",  _CURRENT_YEAR - 1),
    (r"last.year|past.year",        _CURRENT_YEAR - 1),
    (r"this.year",                  _CURRENT_YEAR),
    (r"2025",                       2025),
    (r"2024",                       2024),
    (r"2023",                       2023),
]

# ── Foundational patterns ─────────────────────────────────────────────────────
_FOUNDATIONAL_PATTERNS = [
    r"foundational|seminal|landmark|classic|original|pioneering",
    r"where.*began|first.*proposed|introduced",
]

# ── Content type patterns ─────────────────────────────────────────────────────
_CONTENT_TYPE_PATTERNS: list[tuple[str, ContentType]] = [
    (r"\bpaper\b|\barxiv\b|research",       ContentType.ARXIV_PAPER),
    (r"course|tutorial|chapter|hugging.face", ContentType.COURSE_CHAPTER),
    (r"blog.post|lab.blog",                  ContentType.LAB_BLOG_POST),
    (r"survey|lil.log|lilian",              ContentType.SURVEY_BLOG),
]

# ── Known doc anchors (title/name → doc_id) ───────────────────────────────────
# Lets "the attention paper" resolve directly to a doc_id
_DOC_ANCHORS: list[tuple[str, str]] = [
    (r"attention is all you need|vaswani",          "arxiv-1706.03762"),
    (r"\bbert\b",                                    "arxiv-1810.04805"),
    (r"gpt.3|few.shot.learner|brown.*2020",         "arxiv-2005.14165"),
    (r"rag paper|lewis.*retrieval|lewis.*2020",      "arxiv-2005.11401"),
    (r"\blora\b|hu.*2021",                          "arxiv-2106.09685"),
    (r"chain.of.thought.*wei|wei.*2022",            "arxiv-2201.11903"),
    (r"instructgpt|ouyang",                         "arxiv-2203.02155"),
    (r"\breact\b.*paper|yao.*2022",                 "arxiv-2210.03629"),
    (r"constitutional.ai|bai.*2022",                "arxiv-2212.08073"),
    (r"llama.2|touvron",                            "arxiv-2307.09288"),
    (r"mixtral",                                    "arxiv-2401.04088"),
    (r"\bcrag\b|corrective.rag",                    "arxiv-2401.15884"),
    (r"deepseek.r1",                                "arxiv-2501.12948"),
    (r"deepseek.v3",                                "arxiv-2412.19437"),
    (r"llama.3",                                    "arxiv-2407.21783"),
    (r"gemini.1\.5",                                "arxiv-2403.05530"),
    (r"flash.attention.3",                          "arxiv-2407.08608"),
    (r"swe.agent",                                  "arxiv-2405.15793"),
    (r"graph.rag|graphrag",                         "arxiv-2404.16130"),
    (r"\braft\b.*rag",                              "arxiv-2403.10131"),
    (r"mapping.*mind|anthropic.*interpret",         "anthropic-mapping-mind-2024-05"),
    (r"weak.to.strong",                             "openai-weak-to-strong-2023-12"),
    (r"alphaproof|alphageometry|imo.*deepmind",     "deepmind-alphaproof-2024-07"),
]


def infer_filters(query: str) -> QueryFilters:
    """
    Extract metadata filters from a natural language query.
    All fields default to None (= no filter applied).
    """
    q = query.lower()

    filters = QueryFilters(
        doc_ids=_match_doc_anchors(q) or None,
        content_types=_match_content_types(q) or None,
        tags=_match_tags(q) or None,
        year_gte=_match_recency(q),
        is_foundational=_match_foundational(q),
    )

    logger.debug(
        "filters_inferred",
        query=query[:80],
        filters=filters.model_dump(exclude_none=True),
    )
    return filters


def _match_tags(q: str) -> list[str]:
    return [tag for pattern, tag in _TAG_PATTERNS if re.search(pattern, q)]


def _match_recency(q: str) -> int | None:
    for pattern, year in _RECENCY_PATTERNS:
        if re.search(pattern, q):
            return year
    return None


def _match_foundational(q: str) -> bool | None:
    for pattern in _FOUNDATIONAL_PATTERNS:
        if re.search(pattern, q):
            return True
    return None


def _match_content_types(q: str) -> list[ContentType]:
    return [ct for pattern, ct in _CONTENT_TYPE_PATTERNS if re.search(pattern, q)]


def _match_doc_anchors(q: str) -> list[str]:
    return [doc_id for pattern, doc_id in _DOC_ANCHORS if re.search(pattern, q)]