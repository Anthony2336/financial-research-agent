"""Application input and runtime contracts, independent of orchestration."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, field_validator, model_validator

from fra.domain import DEFAULT_RESEARCH_FORMS, FilingForm, Intent, RouterDecision, StrictModel
from fra.storage.run_repositories import RunFinish


class ResearchMode(StrEnum):
    """Caller-selected workflow mode."""

    THESIS = "thesis"
    AUTO = "auto"
    COMPANY_PROFILE = "company-profile"
    EARNINGS_REVIEW = "earnings-review"
    INDUSTRY_RESEARCH = "industry-research"
    MARKET_SNAPSHOT = "market-snapshot"
    QUALITY_SCREEN = "quality-screen"


class ResearchCommand(StrictModel):
    """Validated input accepted by the research application boundary."""

    ticker: str
    request: str = Field(min_length=1, max_length=2_000)
    mode: ResearchMode
    session_id: str | None = Field(default=None, min_length=1)
    forms: tuple[FilingForm, ...] = Field(
        default=DEFAULT_RESEARCH_FORMS,
        min_length=1,
    )
    as_of_date: date | None = None
    market: Literal["US"] = "US"
    corpus_version: str | None = Field(default=None, min_length=1, exclude=True)
    filing_ids: tuple[str, ...] = Field(default=(), exclude=True)
    scope_error: str | None = Field(default=None, exclude=True)
    peer_tickers: tuple[str, ...] = ()
    peer_scope: str | None = None
    with_context: bool = False

    @field_validator("ticker", mode="before")
    @classmethod
    def normalize_ticker(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("request", mode="before")
    @classmethod
    def normalize_request_text(cls, value: object) -> object:
        return " ".join(value.split()) if isinstance(value, str) else value

    @field_validator("forms", mode="before")
    @classmethod
    def normalize_forms(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip().upper() if isinstance(item, str) else item for item in value)
        return value

    @field_validator("market", mode="before")
    @classmethod
    def normalize_market(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("filing_ids", mode="before")
    @classmethod
    def normalize_filing_ids(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip() if isinstance(item, str) else item for item in value)
        return value

    @field_validator("peer_tickers", mode="before")
    @classmethod
    def normalize_peer_tickers(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(item.strip().upper() for item in value if isinstance(item, str))
        return value

    @model_validator(mode="after")
    def validate_mode_specific_request_length(self) -> ResearchCommand:
        if self.mode is ResearchMode.THESIS and not 20 <= len(self.request) <= 500:
            raise ValueError("thesis mode request must contain 20 to 500 characters")
        if any(not filing_id for filing_id in self.filing_ids):
            raise ValueError("filing_ids must not contain empty ids")
        return self


class IntentRouter(Protocol):
    """Synchronous structured intent classifier used only by AUTO mode."""

    def route(self, request: str) -> RouterDecision:
        """Return one validated closed-enum decision."""


class CompanyResolver(Protocol):
    """Local supported-company lookup used before durable execution."""

    def resolve(self, ticker: str) -> str | None:
        """Return the canonical supported ticker from local company metadata."""


@runtime_checkable
class ApplicationResult(Protocol):
    """Common final lifecycle implemented by every accepted application result."""

    run_id: str
    status: str
    rendered_output: str

    def bind_run_id(self, run_id: str) -> ApplicationResult:
        """Return this result correlated to the application-owned run ID."""

    def to_run_finish(self, trace_id: str | None) -> RunFinish:
        """Return the guarded persistence payload for this result."""

    def root_metadata(self) -> dict[str, object]:
        """Return safe final metadata for the root observation."""


class ResearchRuntime(Protocol):
    """One selected workflow runtime."""

    def execute(
        self,
        command: ResearchCommand,
        decision: RouterDecision,
    ) -> ApplicationResult:
        """Execute a preselected safe research intent."""


class ResearchRuntimeFactory(Protocol):
    """Select a runtime only after request routing succeeds."""

    def build(self, command: ResearchCommand, intent: Intent) -> ResearchRuntime:
        """Build the runtime for one effective intent."""
