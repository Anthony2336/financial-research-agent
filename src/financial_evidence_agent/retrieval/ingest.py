"""Offline fixture and explicit opt-in SEC filing ingestion."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from hashlib import sha256
from pathlib import Path

from bs4 import BeautifulSoup
from pydantic import Field
from sqlalchemy.orm import Session

from financial_evidence_agent.domain import StrictModel
from financial_evidence_agent.retrieval.chunking import chunk_sections, parse_supported_sections
from financial_evidence_agent.retrieval.indexing import (
    PERSISTED_EMBEDDING_DIMENSIONS,
    EmbeddingIndexer,
    EmbeddingProvider,
)
from financial_evidence_agent.retrieval.sec import (
    HttpSecGateway,
    SecGateway,
    SecRawResponseCache,
    SecRequestRateLimiter,
    validate_filing_candidate,
    validate_filing_document,
    validate_forms,
    validate_sec_user_agent,
    validate_ticker,
)
from financial_evidence_agent.retrieval.xbrl import (
    CompanyFactsGateway,
    normalize_company_facts,
)
from financial_evidence_agent.storage.cache import SyncJsonCache
from financial_evidence_agent.storage.fact_repositories import CompanyFactRepository
from financial_evidence_agent.storage.repositories import (
    ChunkToStore,
    FilingRepository,
    FilingToStore,
)
from financial_evidence_agent.web_evidence.source_policy import configured_primary_ir_domain


class IngestSummary(StrictModel):
    """Requested scope and newest filing in the immutable corpus actually stored."""

    ticker: str
    requested_forms: list[str] = Field(min_length=1)
    selected_form: str
    accession_no: str
    filed_at: date
    source_url: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_version: str
    chunk_count: int = Field(ge=0)


def ingest_fixture(
    path: Path,
    ticker: str,
    form: str,
    repository: FilingRepository,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> str:
    """Parse and idempotently persist a fixed filing fixture without network access."""
    raw_bytes = path.read_bytes()
    raw_html = raw_bytes.decode("utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")
    parsed = parse_supported_sections(raw_html)
    chunks = chunk_sections(parsed)
    corpus_version = repository.store_filing(
        ticker=ticker,
        form=form,
        accession_no=_metadata(soup, "accession-no"),
        filed_at=date.fromisoformat(_metadata(soup, "filed-at")),
        source_url=_metadata(soup, "source-url"),
        raw_text=parsed.raw_text,
        content_hash=sha256(raw_bytes).hexdigest(),
        chunks=(
            ChunkToStore(
                section=chunk.section,
                chunk_index=index,
                content=chunk.content,
                token_count=chunk.token_count,
                raw_start=chunk.raw_start,
                raw_end=chunk.raw_end,
            )
            for index, chunk in enumerate(chunks)
        ),
    )
    if embedding_provider is not None:
        EmbeddingIndexer(repository, embedding_provider).ensure_indexed(ticker, corpus_version)
    return corpus_version


def ingest_fixture_summary(
    path: Path,
    ticker: str,
    form: str,
    repository: FilingRepository,
    *,
    embedding_provider: EmbeddingProvider | None = None,
) -> IngestSummary:
    """Ingest a fixture through the stable API and describe the selected local corpus."""
    normalized_ticker = validate_ticker(ticker)
    requested_form = validate_forms([form])[0]
    corpus_version = ingest_fixture(
        path,
        normalized_ticker,
        requested_form,
        repository,
        embedding_provider=embedding_provider,
    )
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    return _persisted_summary(
        repository=repository,
        ticker=normalized_ticker,
        requested_forms=[requested_form],
        corpus_version=corpus_version,
        expected_form=requested_form,
        expected_accession_no=_metadata(soup, "accession-no"),
        expected_filed_at=date.fromisoformat(_metadata(soup, "filed-at")),
        expected_source_url=_metadata(soup, "source-url"),
        expected_content_hash=sha256(path.read_bytes()).hexdigest(),
    )


def ingest_sec(
    ticker: str,
    forms: list[str],
    as_of_date: date | None,
    *,
    repository: FilingRepository,
    user_agent: str,
    gateway: SecGateway | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    cache: SyncJsonCache | None = None,
    sec_cache_ttl_seconds: int = 86_400,
    configured_issuer_domains: Mapping[str, frozenset[str]] | None = None,
    fact_repository: CompanyFactRepository | None = None,
    company_facts_gateway: CompanyFactsGateway | None = None,
    rate_limiter: SecRequestRateLimiter | None = None,
) -> IngestSummary:
    """Fetch and store an immutable corpus of up to four eligible SEC filings."""
    normalized_ticker = validate_ticker(ticker)
    requested_forms = validate_forms(forms)
    validated_user_agent = validate_sec_user_agent(user_agent)
    if sec_cache_ttl_seconds <= 0:
        raise ValueError("SEC cache TTL must be positive")
    if (fact_repository is None) != (company_facts_gateway is None):
        raise ValueError(
            "fact_repository and company_facts_gateway must be configured together"
        )
    if as_of_date is not None and type(as_of_date) is not date:
        raise ValueError("as_of_date must be a date")
    ir_domain = configured_primary_ir_domain(
        normalized_ticker,
        configured_issuer_domains or {},
    )

    resolved_gateway = gateway or HttpSecGateway(rate_limiter=rate_limiter)
    raw_candidates = resolved_gateway.list_filings(normalized_ticker, validated_user_agent)
    candidates = [validate_filing_candidate(candidate) for candidate in raw_candidates]
    if any(
        candidate.ticker != normalized_ticker or int(candidate.cik) != int(candidate.resolved_cik)
        for candidate in candidates
    ):
        raise ValueError("SEC filing candidate issuer does not match the requested ticker")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.form in requested_forms
        and (as_of_date is None or candidate.filed_at <= as_of_date)
    ]
    if not eligible:
        cutoff = as_of_date.isoformat() if as_of_date is not None else "latest available date"
        raise ValueError(
            f"no eligible {normalized_ticker} filing for {requested_forms} at {cutoff}"
        )
    selected_candidates = _select_corpus_candidates(eligible, requested_forms)
    legal_names = {candidate.legal_name for candidate in candidates}
    if len(legal_names) != 1:
        raise ValueError("SEC filing candidates contain conflicting issuer legal names")
    legal_name = next(iter(legal_names))
    issuer_cik = selected_candidates[0].cik.zfill(10)
    facts = ()
    if fact_repository is not None and company_facts_gateway is not None:
        facts_document = company_facts_gateway.fetch_company_facts(
            normalized_ticker,
            issuer_cik,
            validated_user_agent,
        )
        facts_snapshot = normalize_company_facts(
            facts_document,
            ticker=normalized_ticker,
            cik=issuer_cik,
        )
        facts = facts_snapshot.facts
    sec_cache = (
        SecRawResponseCache(cache, ttl_seconds=sec_cache_ttl_seconds)
        if cache is not None
        else None
    )
    prepared_filings: list[FilingToStore] = []
    selected_metadata: list[tuple[str, str, date, str, str]] = []
    seen_content_hashes: set[str] = set()
    for selected in selected_candidates:
        document = sec_cache.get(selected) if sec_cache is not None else None
        if document is None:
            document = validate_filing_document(
                resolved_gateway.fetch_document(selected, validated_user_agent),
                selected,
            )
            if sec_cache is not None:
                sec_cache.set(selected, document)
        content_hash = sha256(document.raw_bytes).hexdigest()
        if content_hash in seen_content_hashes:
            continue
        seen_content_hashes.add(content_hash)
        raw_html = document.raw_bytes.decode("utf-8", errors="replace")
        parsed = parse_supported_sections(raw_html)
        parsed_chunks = chunk_sections(parsed)
        raw_prefix = f"{raw_html}\n"
        stored_chunks = [
            ChunkToStore(
                section=chunk.section,
                chunk_index=index,
                content=chunk.content,
                token_count=chunk.token_count,
                raw_start=len(raw_prefix) + chunk.raw_start,
                raw_end=len(raw_prefix) + chunk.raw_end,
            )
            for index, chunk in enumerate(parsed_chunks)
        ]
        prepared_filings.append(
            FilingToStore(
                form=selected.form,
                accession_no=selected.accession_no,
                filed_at=selected.filed_at,
                source_url=document.source_url,
                raw_text=raw_prefix + parsed.raw_text,
                content_hash=content_hash,
                chunks=tuple(stored_chunks),
            )
        )
        selected_metadata.append(
            (
                selected.form,
                selected.accession_no,
                selected.filed_at,
                document.source_url,
                content_hash,
            )
        )
        if len(prepared_filings) == 4:
            break
    resolved_fact_repository = fact_repository or CompanyFactRepository(repository.engine)
    with repository.embedding_write_context(normalized_ticker) as session:
        prepared_filings = _prepare_filing_embeddings(
            repository,
            session,
            normalized_ticker,
            prepared_filings,
            embedding_provider,
        )
        committed = repository.commit_ingest_batch_in_session(
            session,
            ticker=normalized_ticker,
            cik=issuer_cik,
            legal_name=legal_name,
            ir_domain=ir_domain,
            filings=prepared_filings,
            facts=facts,
            fact_repository=resolved_fact_repository,
            as_of_date=as_of_date or selected_candidates[0].filed_at,
        )
    corpus_version = committed.corpus_version
    primary_form, primary_accession, primary_filed_at, primary_url, primary_hash = (
        selected_metadata[0]
    )
    return _persisted_summary(
        repository=repository,
        ticker=normalized_ticker,
        requested_forms=requested_forms,
        corpus_version=corpus_version,
        expected_form=primary_form,
        expected_accession_no=primary_accession,
        expected_filed_at=primary_filed_at,
        expected_source_url=primary_url,
        expected_content_hash=primary_hash,
    )


def _select_corpus_candidates(candidates, requested_forms: list[str]):
    """Choose per-form newest filings first, then fill by filing date/accession."""
    ranked = sorted(
        candidates,
        key=lambda candidate: (candidate.filed_at, candidate.accession_no),
        reverse=True,
    )
    selected = []
    selected_accessions: set[str] = set()
    for form in requested_forms:
        newest = next((candidate for candidate in ranked if candidate.form == form), None)
        if newest is not None and newest.accession_no not in selected_accessions:
            selected.append(newest)
            selected_accessions.add(newest.accession_no)
    selected.sort(
        key=lambda candidate: (candidate.filed_at, candidate.accession_no),
        reverse=True,
    )
    for candidate in ranked:
        if candidate.accession_no not in selected_accessions:
            selected.append(candidate)
            selected_accessions.add(candidate.accession_no)
    return selected


def _prepare_filing_embeddings(
    repository: FilingRepository,
    session: Session,
    ticker: str,
    filings: list[FilingToStore],
    provider: EmbeddingProvider | None,
) -> list[FilingToStore]:
    """Compute every needed vector before the atomic database transaction starts."""
    if provider is None:
        return filings
    if provider.dimensions != PERSISTED_EMBEDDING_DIMENSIONS:
        raise ValueError(
            "repository-backed embeddings must have "
            f"{PERSISTED_EMBEDDING_DIMENSIONS} dimensions"
        )
    embedding_model = provider.version.strip()
    if not embedding_model:
        raise ValueError("embedding model version must not be blank")
    desired_filings = [
        replace(filing, embedding_model=embedding_model)
        if filing.chunks
        else filing
        for filing in filings
    ]
    filing_indexes: list[int] = []
    texts: list[str] = []
    for filing_index, filing in enumerate(filings):
        if not filing.chunks or not repository.filing_requires_embedding_in_session(
            session,
            ticker=ticker,
            content_hash=filing.content_hash,
            embedding_model=embedding_model,
        ):
            continue
        for chunk in filing.chunks:
            filing_indexes.append(filing_index)
            texts.append(chunk.content)
    if not filing_indexes:
        return desired_filings
    vectors = provider.embed(texts)
    if len(vectors) != len(filing_indexes) or any(
        len(vector) != PERSISTED_EMBEDDING_DIMENSIONS for vector in vectors
    ):
        raise ValueError("embedding vector count or dimensions are invalid")
    by_filing: dict[int, list[tuple[float, ...]]] = {}
    for filing_index, vector in zip(filing_indexes, vectors, strict=True):
        by_filing.setdefault(filing_index, []).append(tuple(float(value) for value in vector))
    return [
        replace(
            desired_filings[index],
            embedding_model=embedding_model,
            embedding_vectors=tuple(by_filing[index]),
        )
        if index in by_filing
        else filing
        for index, filing in enumerate(desired_filings)
    ]


def _persisted_summary(
    *,
    repository: FilingRepository,
    ticker: str,
    requested_forms: list[str],
    corpus_version: str,
    expected_form: str,
    expected_accession_no: str,
    expected_filed_at: date,
    expected_source_url: str,
    expected_content_hash: str,
) -> IngestSummary:
    persisted = repository.get_filing(ticker, corpus_version)
    if persisted is None:
        raise RuntimeError("stored filing could not be read back")
    expected_metadata = (
        expected_form,
        expected_accession_no,
        expected_filed_at,
        expected_source_url,
        expected_content_hash,
    )
    persisted_metadata = (
        persisted.form,
        persisted.accession_no,
        persisted.filed_at,
        persisted.source_url,
        persisted.content_hash,
    )
    if persisted_metadata != expected_metadata:
        raise ValueError("content hash is already associated with different filing metadata")
    return IngestSummary(
        ticker=persisted.ticker,
        requested_forms=requested_forms,
        selected_form=persisted.form,
        accession_no=persisted.accession_no,
        filed_at=persisted.filed_at,
        source_url=persisted.source_url,
        content_hash=persisted.content_hash,
        corpus_version=persisted.corpus_version,
        chunk_count=len(repository.list_chunks(persisted.ticker, persisted.corpus_version)),
    )


def _metadata(soup: BeautifulSoup, name: str) -> str:
    tag = soup.find("meta", attrs={"name": name})
    if tag is None or not tag.get("content"):
        raise ValueError(f"fixture is missing {name} metadata")
    return str(tag["content"])
