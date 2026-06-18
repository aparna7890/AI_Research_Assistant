"""Unit tests for manifest loading."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from research_navigator.ingest.manifest import load_manifest
from research_navigator.models import ContentType

_MANIFEST = {
    "schema_version": "1.0",
    "generated_for": "test",
    "documents": [{
        "doc_id": "arxiv-1706.03762",
        "content_type": "arxiv_paper",
        "title": "Attention Is All You Need",
        "authors": ["Vaswani et al."],
        "year": 2017, "month": 6,
        "primary_category": "cs.CL",
        "secondary_categories": [],
        "tags": ["transformers"],
        "is_foundational": True,
        "citation_count": None,
        "source_url": "https://arxiv.org/abs/1706.03762",
        "local_path": "documents/arxiv/arxiv-1706.03762.pdf",
    }],
}


def test_load_manifest(tmp_path: Path) -> None:
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps(_MANIFEST))
    corpus = load_manifest(f)
    assert len(corpus.documents) == 1
    assert corpus.documents[0].content_type == ContentType.ARXIV_PAPER


def test_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")


def test_invalid_content_type(tmp_path: Path) -> None:
    bad = dict(_MANIFEST)
    bad["documents"] = [{**_MANIFEST["documents"][0], "content_type": "bad_type"}]
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps(bad))
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        load_manifest(f)
