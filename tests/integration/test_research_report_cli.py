"""Offline CLI acceptance coverage through real storage, MCP, graph, guard, and render."""

import re
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from financial_evidence_agent.application import INVALID_TICKER_TEXT
from financial_evidence_agent.cli import app
from financial_evidence_agent.retrieval.ingest import ingest_fixture
from financial_evidence_agent.storage.database import create_schema
from financial_evidence_agent.storage.models import Chunk, Company, Filing, ResearchCorpus
from financial_evidence_agent.storage.repositories import ChunkToStore, FilingRepository

runner = CliRunner()


@pytest.fixture(autouse=True)
def explicit_offline_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every deterministic thesis acceptance test opts into demo composition."""
    monkeypatch.setenv("OFFLINE_DEMO", "true")


def test_no_key_nvda_cli_uses_explicit_fixture_and_renders_grounded_report(
    tmp_path,
    monkeypatch,
) -> None:
    """An explicitly ingested fixture stays runnable without keys or fabricated citations."""
    database_path = tmp_path / "demo.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FAST_MODEL", raising=False)
    monkeypatch.delenv("ANALYST_MODEL", raising=False)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "Does data center demand support revenue growth despite deployment risks?",
        ],
    )

    assert result.exit_code == 0, result.output
    report = result.output
    assert (
        "Execution mode: deterministic offline demo "
        "(pre-existing local corpus; no model API calls)."
    ) in report
    for section in (
        "## Research question",
        "## Verified evidence supporting thesis",
        "## Counter-evidence and risks",
        "## Inferences",
        "## Open questions / insufficient evidence",
        "## Information sufficiency and confidence",
        "## Sources",
    ):
        assert section in report
    assert (
        "Data center revenue grew as cloud service providers, consumer internet companies, "
        "and enterprise customers deployed accelerated computing systems for artificial "
        "intelligence workloads."
    ) in report
    assert (
        "Our revenue concentration among a limited number of customers and the size of their "
        "individual purchases may cause our operating results to fluctuate."
    ) in report
    assert "Corpus version: NVDA-v1" in report
    assert "Filing cutoff: 2025-05-28" in report
    assert "Form: 10-Q" in report
    assert "Filed: 2025-05-28" in report
    assert "Accession: 0001045810-25-000041" in report
    assert "Section: MD\\&A" in report
    assert "Section: Risk Factors" in report
    assert "Raw characters:" in report
    assert "https://www.sec.gov/Archives/edgar/data/1045810/" in report
    assert "page" not in report.casefold()
    assert not re.search(
        r"\b(?:buy|sell|hold|target price|position size|stop loss|take profit)\b",
        report,
        flags=re.IGNORECASE,
    )
    assert "Research assistance only; not investment advice." in report

    assert repository.latest_corpus_version("NVDA") == "NVDA-v1"


@pytest.mark.parametrize(
    "configuration",
    [
        {"OPENAI_API_KEY": "dummy-key"},
        {"FAST_MODEL": "configured-fast-model"},
        {"ANALYST_MODEL": "configured-analyst-model"},
        {
            "OPENAI_API_KEY": "dummy-key",
            "FAST_MODEL": "configured-fast-model",
            "ANALYST_MODEL": "configured-analyst-model",
        },
    ],
)
def test_configured_or_partial_credentials_still_use_and_label_offline_demo(
    tmp_path,
    monkeypatch,
    configuration: dict[str, str],
) -> None:
    """P0 has no live adapters, so configuration must never imply a provider execution."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'configured.sqlite3'}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    monkeypatch.setenv("DATABASE_URL", database_url)
    for variable in ("OPENAI_API_KEY", "FAST_MODEL", "ANALYST_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in configuration.items():
        monkeypatch.setenv(variable, value)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "Does data center demand support revenue growth despite deployment risks?",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        "Execution mode: deterministic offline demo "
        "(pre-existing local corpus; no model API calls)."
    ) in result.output
    assert "Data center revenue grew" in result.output
    assert repository.latest_corpus_version("NVDA") == "NVDA-v1"


def test_research_on_empty_database_writes_no_evidence_rows(
    tmp_path,
    monkeypatch,
) -> None:
    """A read request cannot materialize the bundled fixture, even for a missing scope."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'empty-research.sqlite3'}"
    monkeypatch.setenv("DATABASE_URL", database_url)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "Does historical annual evidence support sustained revenue growth?",
            "--forms",
            "10-K",
            "--as-of-date",
            "2020-12-31",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == INVALID_TICKER_TEXT
    engine = create_engine(database_url)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Company)) == 0
        assert session.scalar(select(func.count()).select_from(Filing)) == 0
        assert session.scalar(select(func.count()).select_from(ResearchCorpus)) == 0
        assert session.scalar(select(func.count()).select_from(Chunk)) == 0


def test_preexisting_corpus_has_distinct_truthful_deterministic_label(
    tmp_path,
    monkeypatch,
) -> None:
    """A local corpus must not be represented as auto-ingested by the current invocation."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'preexisting.sqlite3'}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")
    monkeypatch.setenv("FAST_MODEL", "configured-fast-model")
    monkeypatch.setenv("ANALYST_MODEL", "configured-analyst-model")

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "Does data center demand support revenue growth despite deployment risks?",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        "Execution mode: deterministic offline demo "
        "(pre-existing local corpus; no model API calls)."
    ) in result.output
    assert "auto-ingested bundled" not in result.output
    assert repository.latest_corpus_version("NVDA") == "NVDA-v1"


def test_research_cli_scopes_an_existing_snapshot_subset_without_creating_a_corpus(
    tmp_path,
    monkeypatch,
) -> None:
    """Explicit forms/date select member IDs in an immutable snapshot before retrieval."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'scoped-subset.sqlite3'}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    quarterly_id = repository.list_recent_filings("NVDA", forms=[], limit=1)[0].id
    current_content = "Eight K marker that must never enter the selected ten Q scope."
    repository.store_filing(
        ticker="NVDA",
        form="8-K",
        accession_no="nvda-eight-k-scope",
        filed_at=date(2026, 6, 2),
        source_url="https://www.sec.gov/nvda-eight-k-scope",
        raw_text=current_content,
        content_hash=sha256(current_content.encode()).hexdigest(),
        chunks=[ChunkToStore("Other Disclosure", 0, current_content, 12, 0, len(current_content))],
    )
    with Session(engine) as session:
        current_id = session.scalar(
            select(Filing.id).where(Filing.accession_no == "nvda-eight-k-scope")
        )
    assert current_id is not None
    version = repository.create_corpus("NVDA", [quarterly_id, current_id], date(2026, 6, 2))
    with Session(engine) as session:
        corpus_count = session.scalar(select(func.count()).select_from(ResearchCorpus))
    monkeypatch.setenv("DATABASE_URL", database_url)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "Does data center demand support revenue growth despite deployment risks?",
            "--forms",
            "10-Q",
            "--as-of-date",
            "2099-01-01",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Corpus version: {version}" in result.output
    assert "Form: 10-Q" in result.output
    assert "Eight K marker" not in result.output
    assert "nvda-eight-k-scope" not in result.output
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(ResearchCorpus)) == corpus_count


@pytest.mark.parametrize(
    "thesis",
    [
        "Does the filing disclose lunar mining revenue on Mars?",
        "Does the filing disclose data center revenue on Mars?",
    ],
)
def test_preexisting_nvda_corpus_does_not_answer_an_unrelated_thesis(
    tmp_path,
    monkeypatch,
    thesis: str,
) -> None:
    """Top-k results alone must not turn an unrelated same-ticker question into evidence."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'irrelevant.sqlite3'}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    monkeypatch.setenv("DATABASE_URL", database_url)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            thesis,
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Information sufficiency: C" in result.output
    assert "Confidence: low" in result.output
    assert "No verified supporting evidence retained." in result.output
    assert "No verified counter-evidence retained." in result.output
    assert "Data center revenue grew" not in result.output
    assert "Our revenue concentration" not in result.output
    assert "No cited source retained." in result.output


@pytest.mark.parametrize(
    "thesis",
    [
        "What customer concentration risks are disclosed?",
        "这份文件披露了哪些主要客户集中度风险和相关经营影响？",
    ],
)
def test_customer_concentration_question_retains_the_disclosed_risk_and_source(
    tmp_path,
    monkeypatch,
    thesis: str,
) -> None:
    """A risk-only question must retain its exact filing sentence and canonical source."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'concentration.sqlite3'}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    monkeypatch.setenv("DATABASE_URL", database_url)

    result = runner.invoke(app, ["research", "NVDA", "--thesis", thesis])

    assert result.exit_code == 0, result.output
    assert "No verified supporting evidence retained." in result.output
    assert (
        "Our revenue concentration among a limited number of customers and the size of their "
        "individual purchases may cause our operating results to fluctuate."
    ) in result.output
    assert "Data center revenue grew" not in result.output
    assert "Information sufficiency: C" in result.output
    assert "Confidence: low" in result.output
    assert (
        "URL: <https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581025000041/nvda-20250427.htm>"
    ) in result.output
    assert "Accession: 0001045810-25-000041" in result.output
    assert "Section: Risk Factors" in result.output


def test_decline_thesis_treats_disclosed_growth_as_counter_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    """A positive growth sentence cannot support a thesis claiming revenue declined."""
    database_url = f"sqlite+pysqlite:///{tmp_path / 'decline.sqlite3'}"
    engine = create_engine(database_url)
    create_schema(engine)
    repository = FilingRepository(engine)
    ingest_fixture(Path("tests/fixtures/nvda_10q.html"), "NVDA", "10-Q", repository)
    monkeypatch.setenv("DATABASE_URL", database_url)

    result = runner.invoke(
        app,
        [
            "research",
            "NVDA",
            "--thesis",
            "Does the filing show that data center revenue declined despite strong demand?",
        ],
    )

    assert result.exit_code == 0, result.output
    supporting_section = result.output.split("## Verified evidence supporting thesis", maxsplit=1)[
        1
    ].split("## Counter-evidence and risks", maxsplit=1)[0]
    counter_section = result.output.split("## Counter-evidence and risks", maxsplit=1)[1].split(
        "## Inferences", maxsplit=1
    )[0]
    assert "Data center revenue grew" not in supporting_section
    assert "Data center revenue grew" in counter_section
    assert "Our revenue concentration" in counter_section
    assert "Information sufficiency: C" in result.output
    assert "Confidence: low" in result.output


def test_no_key_demo_does_not_auto_ingest_an_arbitrary_ticker(tmp_path, monkeypatch) -> None:
    """The bundled NVIDIA filing must never silently become another company's evidence."""
    database_path = tmp_path / "demo.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(
        app,
        [
            "research",
            "AMD",
            "--thesis",
            "Does data center demand support revenue growth despite deployment risks?",
        ],
    )

    assert result.exit_code == 0
    assert result.output.strip() == INVALID_TICKER_TEXT
    assert "AMD" not in result.output
    repository = FilingRepository(create_engine(database_url))
    assert repository.get_company("AMD") is None
    assert repository.get_company("NVDA") is None
