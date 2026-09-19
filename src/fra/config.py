"""Environment-backed configuration for local and live integrations."""

from decimal import Decimal
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_WEB_INDUSTRY_AUTHORITY_DOMAINS = frozenset(
    {"ftc.gov", "justice.gov", "commerce.gov", "federalregister.gov"}
)


class Settings(BaseSettings):
    """Runtime settings with live provider credentials optional by default."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    database_url: str = (
        "postgresql+psycopg://financial_evidence:financial_evidence@"
        "localhost:5432/financial_evidence"
    )
    refusal_persistence_connect_timeout_seconds: int = Field(default=1, ge=1, le=10)
    offline_demo: bool = False
    redis_url: str | None = None
    sec_user_agent: str = "Example Research Operator research-operator@example.com"
    sec_cache_ttl_seconds: int = Field(default=86_400, ge=1)
    xbrl_max_response_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    embedding_model: str = "BAAI/bge-m3"
    embedding_cache_dir: str | None = None
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_cache_dir: str | None = None
    context_compressor_model: str = (
        "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    )
    context_compressor_cache_dir: str | None = None
    tokenizer_cache_dir: str | None = None

    fast_model: str | None = None
    analyst_model: str | None = None
    openai_api_key: SecretStr | None = None

    tavily_api_key: SecretStr | None = None
    web_search_timeout_seconds: float = Field(default=10.0, gt=0)
    web_issuer_domains: dict[str, frozenset[str]] = Field(default_factory=dict)
    web_industry_authority_domains: frozenset[str] = Field(
        default_factory=lambda: _DEFAULT_WEB_INDUSTRY_AUTHORITY_DOMAINS
    )

    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str | None = None

    market_data_provider: Literal["alpaca"] = "alpaca"
    alpaca_api_key_id: SecretStr | None = None
    alpaca_api_secret_key: SecretStr | None = None
    alpaca_data_feed: Literal["iex"] = "iex"
    alpaca_trading_environment: Literal["paper", "live"] = "paper"
    market_data_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    market_data_max_bars: int = Field(default=5, ge=1, le=20)
    market_data_max_staleness_seconds: int = Field(default=90, ge=1, le=900)
    market_context_window_days: int = Field(default=3, ge=1, le=7)
    market_abnormal_move_threshold: Decimal = Field(
        default=Decimal("0.05"), gt=Decimal("0"), le=Decimal("1")
    )
    research_quality_max_source_age_days: int = Field(default=365, ge=1, le=3650)
    research_memory_ttl_days: int = Field(default=90, ge=1, le=3650)

    @property
    def has_complete_langfuse_credentials(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key and self.langfuse_host)
