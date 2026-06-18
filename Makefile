.PHONY: setup install lint format test eval ingest-stats docker-up docker-down clean

# ── Bootstrap ─────────────────────────────────────────────────────────────────
setup: install
	uv run pre-commit install
	cp -n .env.example .env || true
	@echo "✅  Setup complete. Edit .env then run: make docker-up && make ingest"

install:
	uv sync --all-extras

# ── Code quality ──────────────────────────────────────────────────────────────
lint:
	uv run ruff check src/ tests/
	uv run mypy src/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d
	@echo "⏳  Waiting for Qdrant to be healthy..."
	@until curl -sf http://localhost:6333/healthz > /dev/null; do sleep 1; done
	@echo "✅  Qdrant is up at http://localhost:6333"

docker-down:
	docker compose down

# ── Corpus ────────────────────────────────────────────────────────────────────
fetch-corpus:
	uv run python corpus/complete_corpus.py

ingest:
	uv run python -m research_navigator.ingest ingest

ingest-stats:
	uv run python -m research_navigator.ingest stats

validate-corpus:
	uv run python -m research_navigator.ingest validate

# ── Evaluation ────────────────────────────────────────────────────────────────
eval:
	uv run python -m research_navigator.eval run --output eval/report.md

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
