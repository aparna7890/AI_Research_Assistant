"""Load and validate manifest.json."""
from __future__ import annotations
import json
from pathlib import Path
import structlog
from research_navigator.models import Corpus, DocumentMeta

logger = structlog.get_logger(__name__)


def load_manifest(manifest_path: Path) -> Corpus:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus = Corpus.model_validate(raw)
    logger.info("manifest_loaded", path=str(manifest_path), count=len(corpus.documents))
    return corpus


def resolve_document_path(doc: DocumentMeta, corpus_root: Path) -> Path:
    return corpus_root / doc.local_path
