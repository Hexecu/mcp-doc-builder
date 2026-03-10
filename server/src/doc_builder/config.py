"""
Configuration management using Pydantic Settings.
Supports both Gemini Direct and LiteLLM Gateway modes.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=(".env", str(Path.home() / ".doc-builder" / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Neo4j Configuration
    # ─────────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI (bolt:// or neo4j+s://)",
    )
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(default="password123")

    # ─────────────────────────────────────────────────────────────────────
    # LLM Configuration - Dual Mode Support
    # ─────────────────────────────────────────────────────────────────────
    llm_mode: Literal["gemini_direct", "litellm", "both"] = Field(
        default="litellm",
        description="LLM provider mode: gemini_direct, litellm, or both",
    )
    llm_primary: Literal["gemini", "litellm"] = Field(
        default="litellm",
        description="Primary provider when mode is 'both'",
    )

    # Gemini Direct
    gemini_api_key: str | None = Field(default=None)
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/")
    gemini_model: str = Field(default="gemini-2.5-flash")

    # LiteLLM Gateway
    litellm_base_url: str | None = Field(default=None)
    litellm_api_key: str | None = Field(default=None)
    litellm_model: str = Field(default="gemini-2.5-flash")

    # Embedding Model (Gemini embedding - gemini-embedding-001 produces 3072 dims)
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_dimensions: int = Field(default=3072)

    # Task-specific model routing
    doc_model_default: str | None = Field(
        default=None,
        description="Default model for general tasks (falls back to litellm_model)",
    )
    doc_model_fast: str | None = Field(
        default=None,
        description="Fast model for high-throughput tasks like link evaluation",
    )
    doc_model_reason: str | None = Field(
        default=None,
        description="Reasoning model for complex ontology extraction",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Crawler Configuration
    # ─────────────────────────────────────────────────────────────────────
    crawler_max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum hop depth from root URL",
    )
    crawler_rate_limit: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Minimum seconds between requests to same domain",
    )
    crawler_max_concurrent: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent requests",
    )
    crawler_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Request timeout in seconds",
    )
    crawler_max_pages: int = Field(
        default=500,
        ge=10,
        le=5000,
        description="Maximum pages to crawl per source",
    )
    crawler_user_agent: str = Field(
        default="DocBuilderBot/1.0 (MCP Documentation Indexer)",
        description="User agent string for requests",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Vector Configuration
    # ─────────────────────────────────────────────────────────────────────
    vector_chunk_size: int = Field(
        default=800,
        ge=200,
        le=2000,
        description="Target chunk size in tokens",
    )
    vector_chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks in tokens",
    )
    vector_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for search results",
    )
    vector_index_name: str = Field(
        default="doc_chunk_embeddings",
        description="Name of the Neo4j vector index",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Server Configuration
    # ─────────────────────────────────────────────────────────────────────
    mcp_host: str = Field(default="127.0.0.1")
    mcp_port: int = Field(default=8001)  # Different from kg-memory (8000)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # ─────────────────────────────────────────────────────────────────────
    # Security Configuration
    # ─────────────────────────────────────────────────────────────────────
    doc_mcp_token: str | None = Field(
        default=None,
        description="Bearer token for authentication (optional)",
    )
    doc_allowed_origins: str = Field(
        default="localhost,127.0.0.1",
        description="Comma-separated list of allowed origins",
    )

    # ─────────────────────────────────────────────────────────────────────
    # Validators
    # ─────────────────────────────────────────────────────────────────────
    @field_validator("doc_allowed_origins")
    @classmethod
    def parse_origins(cls, v: str) -> str:
        """Ensure origins string is properly formatted."""
        return ",".join(o.strip() for o in v.split(",") if o.strip())

    # ─────────────────────────────────────────────────────────────────────
    # Computed Properties
    # ─────────────────────────────────────────────────────────────────────
    @property
    def allowed_origins_list(self) -> list[str]:
        """Return allowed origins as a list."""
        return [o.strip() for o in self.doc_allowed_origins.split(",") if o.strip()]

    @property
    def active_model(self) -> str:
        """Get the active model based on configuration."""
        if self.llm_mode == "gemini_direct":
            return f"gemini/{self.gemini_model}"
        elif self.llm_mode == "litellm":
            return self.litellm_model
        else:
            # Both mode - use primary
            if self.llm_primary == "gemini":
                return f"gemini/{self.gemini_model}"
            return self.litellm_model

    @property
    def fast_model(self) -> str:
        """Get the fast model for high-throughput tasks."""
        return self.doc_model_fast or self.active_model

    @property
    def reason_model(self) -> str:
        """Get the reasoning model for complex tasks."""
        return self.doc_model_reason or self.active_model

    def get_litellm_kwargs(self) -> dict:
        """Get kwargs for LiteLLM client initialization."""
        kwargs: dict = {}

        if self.llm_mode in ("litellm", "both"):
            if self.litellm_base_url:
                kwargs["base_url"] = self.litellm_base_url
            if self.litellm_api_key:
                kwargs["api_key"] = self.litellm_api_key

        elif self.llm_mode == "gemini_direct":
            if self.gemini_api_key:
                kwargs["api_key"] = self.gemini_api_key

        return kwargs


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
