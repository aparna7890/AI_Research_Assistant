"""Ingestion CLI: python -m research_navigator.ingest [ingest|validate|stats|reindex]"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import structlog
import typer
from rich.console import Console
from rich.table import Table
from research_navigator.ingest.chunker import chunk_document
from research_navigator.ingest.embedder import embed_texts
from research_navigator.ingest.manifest import load_manifest, resolve_document_path
from research_navigator.ingest.parsers import parse_markdown, parse_pdf
from research_navigator.ingest.qdrant_store import collection_stats, ensure_collection, get_client, upsert_chunks
from research_navigator.ingest.sparse import build_sparse_vector
from research_navigator.logging import configure_logging
from research_navigator.models import ContentType
from research_navigator.settings import settings

app = typer.Typer(name="ingest", no_args_is_help=True)
console = Console()
logger = structlog.get_logger(__name__)


@app.callback()
def _setup() -> None:
    configure_logging(settings.log_level)


@app.command()
def ingest(
    doc_id: Optional[str] = typer.Option(None, help="Ingest one document by doc_id"),
    dry_run: bool = typer.Option(False, help="Parse + chunk without writing to Qdrant"),
) -> None:
    """Ingest the corpus into Qdrant."""
    corpus = load_manifest(settings.manifest_path)
    docs = corpus.documents
    if doc_id:
        docs = [d for d in docs if d.doc_id == doc_id]
        if not docs:
            typer.echo(f"doc_id '{doc_id}' not found.", err=True)
            raise typer.Exit(1)

    client = get_client(settings)
    ensure_collection(client, settings)
    total = 0

    for doc in docs:
        path = resolve_document_path(doc, settings.corpus_root)
        if not path.exists():
            console.print(f"[yellow]SKIP {doc.doc_id} — file missing: {path}[/yellow]")
            continue
        try:
            sections = parse_pdf(path) if doc.content_type == ContentType.ARXIV_PAPER else parse_markdown(path)
            chunks = chunk_document(doc, sections)
        except Exception as e:
            console.print(f"[red]FAIL {doc.doc_id}: {e}[/red]")
            logger.error("ingest_failed", doc_id=doc.doc_id, error=str(e))
            continue

        if dry_run:
            console.print(f"[dim]dry-run {doc.doc_id} → {len(chunks)} chunks[/dim]")
            total += len(chunks)
            continue

        texts = [c.text for c in chunks]
        dense = embed_texts(texts, settings.embedding_model)
        sparse = [build_sparse_vector(t) for t in texts]
        upsert_chunks(client, settings.qdrant_collection, chunks, dense, sparse)
        total += len(chunks)
        console.print(f"[green]OK[/green] {doc.doc_id} → {len(chunks)} chunks")

    console.print(f"\n[bold]Total: {total} chunks[/bold]")


@app.command()
def validate() -> None:
    """Check every document file exists on disk."""
    corpus = load_manifest(settings.manifest_path)
    missing = [
        f"{d.doc_id} → {resolve_document_path(d, settings.corpus_root)}"
        for d in corpus.documents
        if not resolve_document_path(d, settings.corpus_root).exists()
    ]
    if missing:
        console.print(f"[red]{len(missing)} missing:[/red]")
        for m in missing:
            console.print(f"  {m}")
        raise typer.Exit(1)
    console.print(f"[green]All {len(corpus.documents)} documents present.[/green]")


@app.command()
def stats() -> None:
    """Show chunk counts by content_type in Qdrant."""
    client = get_client(settings)
    counts = collection_stats(client, settings.qdrant_collection)
    t = Table(title=f"Collection: {settings.qdrant_collection}")
    t.add_column("Metric", style="cyan")
    t.add_column("Count", justify="right")
    for k, v in counts.items():
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def reindex(doc_id: str = typer.Argument(...)) -> None:
    """Delete and re-ingest one document."""
    from qdrant_client.http import models as qm
    client = get_client(settings)
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(filter=qm.Filter(must=[
            qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))
        ])),
    )
    ingest(doc_id=doc_id)


if __name__ == "__main__":
    app()
