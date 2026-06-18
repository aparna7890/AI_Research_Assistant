"""Unit tests for sparse vectors."""
from __future__ import annotations
from research_navigator.ingest.sparse import build_sparse_vector, tokenise


def test_stopwords_removed() -> None:
    assert "the" not in tokenise("the cat sat on the mat")


def test_lowercased() -> None:
    assert "llm" in tokenise("LLM Transformers")


def test_sparse_vector_structure() -> None:
    v = build_sparse_vector("attention is all you need")
    assert "indices" in v and "values" in v
    assert len(v["indices"]) == len(v["values"]) > 0


def test_empty_gives_empty() -> None:
    v = build_sparse_vector("the and or")
    assert v["indices"] == []


def test_deterministic() -> None:
    t = "retrieval augmented generation"
    assert build_sparse_vector(t) == build_sparse_vector(t)
