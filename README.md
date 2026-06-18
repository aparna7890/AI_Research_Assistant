# AI Research Navigator

Citation-grounded RAG chatbot for AI/ML learners — built on Qdrant + LangGraph.

## Quick start (Windows)

### Prerequisites
- Python ≥ 3.11
- Docker Desktop
- uv (`powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`)

### 1. Install dependencies
```powershell
uv sync --all-extras
```

### 2. Configure
```powershell
copy .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and CORPUS_ROOT
```

### 3. Start Qdrant
```powershell
docker compose up -d
```

### 4. Run tests
```powershell
uv run pytest tests/ -v
```

### 5. Ingest corpus
```powershell
uv run python -m research_navigator.ingest ingest
```

## Project layout
```
src/research_navigator/
├── settings.py       # all config via .env
├── models.py         # shared data models
├── logging.py        # structlog setup
├── ingest/           # M1: parse → chunk → embed → upsert
├── retrieve/         # M2: hybrid search
├── generate/         # M2: citation generation
├── agents/           # M3: LangGraph state machine
└── eval/             # M4: evaluation harness
corpus/
├── manifest.json
├── complete_corpus.py
└── documents/
    ├── arxiv/
    ├── hf-learn/
    ├── lillog/
    └── lab-blogs/
docs/adr/             # Architecture Decision Records
tests/
```

## Commands
| Command | What it does |
|---|---|
| `uv sync --all-extras` | Install all dependencies |
| `uv run pytest tests/ -v` | Run all tests |
| `docker compose up -d` | Start Qdrant |
| `uv run python -m research_navigator.ingest ingest` | Ingest corpus |
| `uv run python -m research_navigator.ingest stats` | Show chunk counts |
| `uv run python -m research_navigator.ingest validate` | Check all files exist |
