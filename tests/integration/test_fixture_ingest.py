import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from hashlib import sha256
from pathlib import Path
from threading import Barrier

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session

from fra.retrieval.hybrid import HybridRetriever
from fra.retrieval.ingest import ingest_fixture
from fra.storage.cache import RedisJsonCache
from fra.storage.database import create_schema
from fra.storage.models import Chunk, Filing, ResearchCorpus
from fra.storage.repositories import ChunkToStore, FilingRepository


class RecordingEmbeddings:
    version = "hash-1024-v1"
    dimensions = 1024

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return [[1.0, *([0.0] * 1023)] for _ in texts]


class RecordingRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.set_calls: list[tuple[str, str, int]] = []

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, *, ex: int) -> None:
        self.set_calls.append((key, value, ex))
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


@pytest.fixture
def engine():
    database_engine = create_engine("sqlite+pysqlite:///:memory:")
    create_schema(database_engine)
    return database_engine


@pytest.fixture
def repository(engine) -> FilingRepository:
    return FilingRepository(engine)


@pytest.fixture
def nvda_fixture_path() -> Path:
    return Path("tests/fixtures/nvda_10q.html")


def test_fixture_ingest_persists_citable_chunks(
    engine, repository: FilingRepository, nvda_fixture_path: Path
) -> None:
    """Missing filing metadata or source spans would make citations unverifiable."""
    corpus_version = ingest_fixture(nvda_fixture_path, "NVDA", "10-Q", repository)

    chunks = repository.list_chunks("NVDA", corpus_version)

    assert corpus_version == "NVDA-v1"
    assert {chunk.section for chunk in chunks} >= {"MD&A", "Risk Factors"}
    assert all(chunk.accession_no and chunk.source_url for chunk in chunks)
    assert all(chunk.raw_end > chunk.raw_start for chunk in chunks)
    assert all(
        350 <= len(chunk.content.split()) <= 500
        for chunk in chunks
        if chunk.section in {"MD&A", "Risk Factors"}
    )
    with Session(engine) as session:
        raw_text = session.scalar(
            select(Filing.raw_text).where(Filing.corpus_version == corpus_version)
        )
    assert raw_text is not None
    assert all(raw_text[chunk.raw_start : chunk.raw_end] == chunk.content for chunk in chunks)


def test_fixture_ingest_reuses_existing_corpus_for_identical_content(
    repository: FilingRepository, nvda_fixture_path: Path
) -> None:
    """Reprocessing an unchanged filing must not create a duplicate corpus or chunks."""
    first_version = ingest_fixture(nvda_fixture_path, "NVDA", "10-Q", repository)
    first_chunks = repository.list_chunks("NVDA", first_version)

    second_version = ingest_fixture(nvda_fixture_path, "NVDA", "10-Q", repository)
    second_chunks = repository.list_chunks("NVDA", second_version)

    assert second_version == first_version == "NVDA-v1"
    assert [chunk.id for chunk in second_chunks] == [chunk.id for chunk in first_chunks]


def test_ingest_indexes_a_corpus_once_when_a_provider_is_supplied(
    repository: FilingRepository,
    nvda_fixture_path: Path,
) -> None:
    """Runtime ingest must persist document vectors without repeating unchanged work."""
    embeddings = RecordingEmbeddings()

    ingest_fixture(
        nvda_fixture_path,
        "NVDA",
        "10-Q",
        repository,
        embedding_provider=embeddings,
    )
    ingest_fixture(
        nvda_fixture_path,
        "NVDA",
        "10-Q",
        repository,
        embedding_provider=embeddings,
    )

    assert len(embeddings.document_batches) == 1
    assert embeddings.document_batches[0]


def test_document_vectors_stay_in_repository_while_only_results_reach_redis(
    engine,
    repository: FilingRepository,
    nvda_fixture_path: Path,
) -> None:
    """Durable vectors must not create a second Redis vector store."""
    corpus_version = ingest_fixture(nvda_fixture_path, "NVDA", "10-Q", repository)
    embeddings = RecordingEmbeddings()
    redis = RecordingRedis()
    cache = RedisJsonCache(object(), sync_client=redis)  # type: ignore[arg-type]
    retriever = HybridRetriever(repository, embeddings, cache=cache)

    indexed_count = retriever.ensure_indexed("NVDA", corpus_version)

    with Session(engine) as session:
        persisted = session.execute(
            select(Chunk.embedding, Chunk.embedding_model).order_by(Chunk.id)
        ).all()
    assert indexed_count == len(persisted) > 0
    assert all(model == embeddings.version for _, model in persisted)
    assert all(vector is not None and len(vector) == 1024 for vector, _ in persisted)
    assert redis.set_calls == []

    evidence = retriever.search("NVDA", "data center revenue", corpus_version, k=1)

    assert evidence
    assert len(redis.set_calls) == 1
    _, encoded_payload, _ = redis.set_calls[0]
    cached_evidence = json.loads(encoded_payload)
    assert isinstance(cached_evidence, list)
    assert all(
        "embedding" not in item and "embedding_model" not in item
        for item in cached_evidence
    )


def test_changed_fixture_content_creates_the_next_corpus_version(
    repository: FilingRepository, nvda_fixture_path: Path, tmp_path: Path
) -> None:
    """New filing content must get a new per-ticker corpus version."""
    updated_fixture = tmp_path / "nvda_10q_updated.html"
    updated_fixture.write_text(
        nvda_fixture_path.read_text(encoding="utf-8").replace(
            "The pace of revenue growth", "The near-term pace of revenue growth"
        ),
        encoding="utf-8",
    )

    first_version = ingest_fixture(nvda_fixture_path, "NVDA", "10-Q", repository)
    second_version = ingest_fixture(updated_fixture, "NVDA", "10-Q", repository)

    assert first_version == "NVDA-v1"
    assert second_version == "NVDA-v2"
    assert repository.list_chunks("NVDA", first_version)
    assert repository.list_chunks("NVDA", second_version)


def test_chunks_are_isolated_by_ticker_and_corpus_version(
    repository: FilingRepository, nvda_fixture_path: Path
) -> None:
    """A retrieval scope must not cross ticker or corpus-version boundaries."""
    nvda_version = ingest_fixture(nvda_fixture_path, "NVDA", "10-Q", repository)
    amd_version = ingest_fixture(nvda_fixture_path, "AMD", "10-Q", repository)

    assert all(chunk.ticker == "NVDA" for chunk in repository.list_chunks("NVDA", nvda_version))
    assert all(chunk.ticker == "AMD" for chunk in repository.list_chunks("AMD", amd_version))
    assert repository.list_chunks("NVDA", amd_version) == []


def test_filings_enforce_one_corpus_version_per_company(engine) -> None:
    """A schema regression must not permit two filings in one ticker/corpus scope."""
    constraints = inspect(engine).get_unique_constraints("filings")

    assert any(
        set(constraint["column_names"]) == {"company_id", "corpus_version"}
        for constraint in constraints
    )


def test_repository_creates_immutable_multi_filing_snapshots_and_rejects_cross_ticker(
    engine,
    repository: FilingRepository,
) -> None:
    """A corpus must reproduce its exact same-ticker membership after later snapshots exist."""

    def store(ticker: str, form: str, accession: str, filed_at: date) -> str:
        content = f"{ticker} {form} {accession} evidence"
        repository.store_filing(
            ticker=ticker,
            form=form,
            accession_no=accession,
            filed_at=filed_at,
            source_url=f"https://www.sec.gov/{accession}",
            raw_text=content,
            content_hash=sha256(content.encode()).hexdigest(),
            chunks=[ChunkToStore("MD&A", 0, content, 4, 0, len(content))],
        )
        with Session(engine) as session:
            return session.scalar(select(Filing.id).where(Filing.accession_no == accession))

    ten_k = store("NVDA", "10-K", "nvda-10k", date(2025, 2, 20))
    ten_q = store("NVDA", "10-Q", "nvda-10q", date(2025, 5, 28))
    eight_k = store("NVDA", "8-K", "nvda-8k", date(2025, 6, 2))
    amd = store("AMD", "10-Q", "amd-10q", date(2025, 5, 28))
    assert all((ten_k, ten_q, eight_k, amd))

    old_version = repository.create_corpus(
        "NVDA", [ten_k, ten_q], date(2025, 5, 28)
    )
    old_chunk_ids = [chunk.id for chunk in repository.list_chunks("NVDA", old_version)]
    current_version = repository.create_corpus(
        "NVDA", [ten_k, ten_q, eight_k], date(2025, 6, 2)
    )

    assert {chunk.form for chunk in repository.list_chunks("NVDA", current_version)} == {
        "10-K",
        "10-Q",
        "8-K",
    }
    assert [chunk.id for chunk in repository.list_chunks("NVDA", old_version)] == old_chunk_ids
    assert (
        repository.create_corpus(
            "NVDA", [eight_k, ten_q, ten_k], date(2025, 6, 30)
        )
        == current_version
    )
    with pytest.raises(ValueError, match="same company"):
        repository.create_corpus("NVDA", [ten_q, amd], date(2025, 5, 28))


def test_corpus_cutoff_is_derived_from_membership_and_reused_for_historical_lookup(
    engine,
    repository: FilingRepository,
) -> None:
    """A future selection cutoff must not hide identical evidence at its filing date."""

    def store(ticker: str, accession: str, filed_at: date) -> str:
        content = f"{ticker} {accession} evidence"
        repository.store_filing(
            ticker=ticker,
            form="10-Q",
            accession_no=accession,
            filed_at=filed_at,
            source_url=f"https://www.sec.gov/{accession}",
            raw_text=content,
            content_hash=sha256(content.encode()).hexdigest(),
            chunks=[ChunkToStore("MD&A", 0, content, 3, 0, len(content))],
        )
        with Session(engine) as session:
            filing_id = session.scalar(
                select(Filing.id).where(Filing.accession_no == accession)
            )
        assert filing_id is not None
        return filing_id

    older = store("NVDA", "nvda-old", date(2025, 2, 20))
    newer = store("NVDA", "nvda-new", date(2025, 5, 28))
    amd = store("AMD", "amd-current", date(2025, 4, 30))
    amd_version = repository.corpus_version_for_filings("AMD", [amd])

    future_first = repository.create_corpus(
        "NVDA",
        [newer, older],
        date(2099, 1, 1),
    )
    reused = repository.create_corpus(
        "NVDA",
        [older, newer],
        date(2025, 5, 28),
    )

    with Session(engine) as session:
        persisted_cutoff = session.scalar(
            select(ResearchCorpus.as_of_date).where(
                ResearchCorpus.version == future_first
            )
        )
    assert reused == future_first
    assert persisted_cutoff == date(2025, 5, 28)
    assert (
        repository.latest_corpus_version("NVDA", as_of_date=date(2025, 5, 28))
        == future_first
    )
    assert repository.latest_corpus_version("AMD") == amd_version


def test_concurrent_identical_membership_creates_one_durable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent repositories must return one version backed by one membership identity."""
    engine = create_engine(
        f"sqlite+pysqlite:///{tmp_path / 'concurrent-corpus.sqlite3'}",
        connect_args={"timeout": 30},
    )
    create_schema(engine)
    repository = FilingRepository(engine)

    def store(accession: str, filed_at: date) -> str:
        content = f"{accession} evidence"
        repository.store_filing(
            ticker="NVDA",
            form="10-Q",
            accession_no=accession,
            filed_at=filed_at,
            source_url=f"https://www.sec.gov/{accession}",
            raw_text=content,
            content_hash=sha256(content.encode()).hexdigest(),
            chunks=[ChunkToStore("MD&A", 0, content, 2, 0, len(content))],
        )
        with Session(engine) as session:
            filing_id = session.scalar(
                select(Filing.id).where(Filing.accession_no == accession)
            )
        assert filing_id is not None
        return filing_id

    filing_ids = [
        store("nvda-quarterly-one", date(2025, 5, 20)),
        store("nvda-quarterly-two", date(2025, 5, 28)),
    ]
    start = Barrier(2)
    before_insert = Barrier(2)
    original_next_version = FilingRepository._next_corpus_version

    def synchronized_next_version(session, company_id: str, ticker: str) -> str:
        version = original_next_version(session, company_id, ticker)
        before_insert.wait(timeout=5)
        return version

    monkeypatch.setattr(
        FilingRepository,
        "_next_corpus_version",
        staticmethod(synchronized_next_version),
    )

    def create() -> str:
        start.wait(timeout=5)
        return FilingRepository(engine).create_corpus(
            "NVDA", filing_ids, date(2025, 5, 28)
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = list(executor.map(lambda _: create(), range(2)))

    with Session(engine) as session:
        memberships = session.scalars(
            select(ResearchCorpus).where(
                ResearchCorpus.membership_hash
                == sha256("\n".join(sorted(filing_ids)).encode()).hexdigest()
            )
        ).all()

    assert len(set(versions)) == 1
    assert len(memberships) == 1
