"""Central config — reads from .env. Import `settings` everywhere."""
from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    generation_model: str = Field(default="claude-opus-4-6")
    max_tokens: int = Field(default=2048)

    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_collection: str = Field(default="research_navigator")
    qdrant_api_key: str | None = Field(default=None)

    retrieval_top_k: int = Field(default=10)
    refusal_threshold: float = Field(default=0.35)
    hybrid_alpha: float = Field(default=0.7)

    corpus_root: Path = Field(default=Path("./corpus"))
    log_level: str = Field(default="INFO")

    @property
    def manifest_path(self) -> Path:
        return self.corpus_root / "manifest.json"


settings = Settings()
