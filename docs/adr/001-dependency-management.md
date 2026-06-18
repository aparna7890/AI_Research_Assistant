# ADR 001 — uv for dependency management

**Status:** Accepted

## Decision
Use `uv` with `pyproject.toml` and a committed `uv.lock`.

## Rationale
- 10–100x faster than pip/poetry
- PEP 517 compliant, standard `pyproject.toml`
- `uv sync` reproduces exact environment on any machine

## Consequences
- Requires uv ≥ 0.4 on dev machines and CI
