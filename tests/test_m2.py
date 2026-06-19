"""Quick M2 smoke test — run from project root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research_navigator.logging import configure_logging
from research_navigator.generate.pipeline import ask
from research_navigator.models import AgentRoute
from research_navigator.settings import settings

configure_logging("INFO")

questions = [
    "How does RAG work?",
    "What is the LoRA paper about?",
    "Explain chain of thought prompting",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Q: {q}")
    print('='*60)
    
    answer = ask(q, route=AgentRoute.CONCEPT_EXPLANATION, settings=settings)
    
    if answer.refused:
        print(f"REFUSED: {answer.refusal_reason}")
    else:
        print(answer.answer_text)
        print("\nCITATIONS:")
        for c in answer.citations:
            print(f"  [{c.index}] {c.title} ({c.year}) — {c.source_label}")
            print(f"       {c.url}")