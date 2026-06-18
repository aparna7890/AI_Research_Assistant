"""
Entry point: python -m research_navigator

Delegates to sub-commands:
  python -m research_navigator.ingest  …
  python -m research_navigator.eval    …
"""
from __future__ import annotations

import typer

from research_navigator.logging import configure_logging
from research_navigator.settings import settings

app = typer.Typer(
    name="research-navigator",
    help="AI Research Navigator — citation-grounded RAG for AI/ML learners.",
    no_args_is_help=True,
)


@app.callback()
def _setup() -> None:
    configure_logging(settings.log_level)


if __name__ == "__main__":
    app()
