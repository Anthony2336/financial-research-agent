# Financial Research Agent Engineering Highlights

## Project Overview

Financial Evidence Audit Agent is a command-line research system for public US companies. It converts a ticker and a research question into a source-grounded report that separates verified facts, analytical inferences, counterevidence, and unresolved questions.

The product is designed as an evidence-audit system rather than an investment recommender. It does not generate trade instructions, price targets, portfolio actions, or personalized financial advice. Market information is presented with source and freshness metadata, while causal claims remain conservative when evidence is incomplete.

## System Architecture

The application uses one controlled research agent rather than an autonomous agent swarm. A deterministic safety layer handles prohibited advice, prompt injection, malformed identifiers, and privacy-sensitive content before model or tool execution. Supported requests are mapped to versioned research recipes with explicit tools, sources, budgets, schemas, and output guards.

LangGraph coordinates the workflow from request normalization and question planning through retrieval, analysis, citation validation, rendering, persistence, and memory updates. FastMCP provides a typed boundary around company resolution, SEC filing access, hybrid retrieval, source-span lookup, allowlisted web evidence, and market data.

The primary flow is:

1. Validate safety, privacy, request mode, and locally supported ticker metadata.
2. Select a versioned research recipe and freeze its tools, sources, and budgets.
3. Plan a small set of support and challenge questions.
4. Retrieve evidence from the current immutable company corpus.
5. Rerank and compress evidence without changing citation identity or financial values.
6. Generate a structured research memo from the approved evidence only.
7. Apply deterministic citation, provenance, privacy, and output guards.
8. Persist the guarded report, source relationships, trace identity, and eligible memory.

## Technology Stack

| Area | Technology | Engineering Role |
|---|---|---|
| Runtime | Python 3.11, uv | Reproducible dependency and command environment |
| CLI | Typer, Rich | Research, ingestion, evaluation, and readable terminal output |
| Validation | Pydantic v2 | Strict request, tool, state, evidence, and report schemas |
| Orchestration | LangGraph | Stateful nodes, bounded retries, conditional routing, and termination |
| Model integration | LangChain, provider SDKs | Structured outputs, tool binding, and model abstraction |
| Tool protocol | FastMCP, langchain-mcp-adapters | Typed, least-privilege financial research tools |
| Dense retrieval | BGE-M3, pgvector | Semantic retrieval scoped to ticker and corpus version |
| Sparse retrieval | BM25 | Exact matching for metrics, risk language, dates, and identifiers |
| Fusion and reranking | Reciprocal Rank Fusion, FlashRank 0.2.10 | Candidate fusion and local relevance ranking |
| Context compression | LLMLingua | Evidence compression while preserving citation boundaries |
| Persistence | PostgreSQL 16, SQLAlchemy 2, Alembic | Durable filings, facts, reports, claims, provenance, and memory |
| Cache and session state | Redis 7 | TTL caches and bounded short-term conversation memory |
| External access | httpx, tenacity, provider adapters | Timeouts, retries, rate limits, and normalized provider responses |
| Market data | Alpaca Basic adapter | IEX-only snapshots and bars with freshness metadata |
| Observability | Langfuse | Traces, prompt/model versions, cost, datasets, and evaluation scores |
| Delivery | Docker Compose, GitHub Actions | Local services, offline CI, and protected integration workflows |
| Quality | pytest, Ruff | Behavioral, integration, migration, and static-analysis gates |

## Core Engineering Design

### Safety and Model Routing

Deterministic rules reject prohibited financial advice and prompt-injection patterns without consuming model tokens. A fast model is used only when intent or question planning is ambiguous. The analyst model runs only after controlled evidence has been collected and cannot make safety decisions or freely access tools.

Ticker values are resolved against locally persisted company metadata before they enter durable execution. Unresolved or privacy-sensitive values are represented with fixed identifiers and cryptographic digests rather than raw user input.

### Versioned Research Recipes

The internal Skill Registry defines a small set of versioned research recipes for company research, management and governance, earnings review, financial verification, industry research, and research-quality screening.

Each recipe declares accepted intents, allowed MCP tools, source policy, token and tool budgets, input/output schemas, and guard profile. This keeps model behavior configurable without turning recipes into unrestricted plugins.

### Agentic Hybrid RAG

SEC filings are parsed into immutable, versioned corpora with exact raw spans and bounded chunk overlap. Retrieval combines BGE-M3 dense search with BM25 sparse search, then merges candidates through Reciprocal Rank Fusion.

Every non-empty fused candidate set is reranked locally with FlashRank. Support queries prioritize evidence that substantiates a thesis, while challenge queries reserve same-scope risk evidence when the initial top results contain no meaningful counterevidence.

Retrieval quality is evaluated with deterministic coverage rules. One bounded query rewrite is allowed when evidence is incomplete; only then may the workflow use one allowlisted web fallback. Failure produces an explicit insufficient-evidence result rather than an answer based on model memory.

### Context Engineering

Prompt construction separates control instructions, task metadata, memory hints, evidence blocks, and output constraints. Retrieved documents are always treated as untrusted data rather than instructions.

LLMLingua compresses evidence text only. Evidence IDs, tickers, dates, values, URLs, filing identifiers, sections, and citation boundaries remain unchanged. Token pressure removes low-value evidence and memory hints before any control instruction or provenance metadata is truncated.

### MCP and Least-Privilege Tools

FastMCP exposes narrow, typed tools for company resolution, filing lookup, hybrid retrieval, source-span access, allowlisted web evidence, and market data. LangGraph decides which tools are available at each node; the model can select only from that restricted set.

Database queries, cache operations, reranking, context compression, citation validation, and budget accounting remain deterministic internal functions and are never exposed as model-callable tools.

### Provenance and Citation Guard

Filing, XBRL, web, and market evidence use separate typed provenance models. Financial observations retain period, currency, unit, definition, source identity, discrepancy, and limitation metadata. Incompatible values are reported as non-comparable rather than averaged or silently normalized.

Every factual claim must reference evidence from the active ticker and corpus. The citation guard validates source existence, scope, original spans, filing metadata, URLs, timestamps, and allowed output fields. A single bounded repair may remove or downgrade unsupported content but cannot introduce new facts or reduce previously guarded evidence quality.

### Four-Layer Memory Model

The system separates memory by responsibility:

- Versioned system prompts define immutable control policy.
- LangGraph state stores current-run working data and expires after execution.
- Redis stores a bounded five-turn session summary with a 24-hour TTL.
- PostgreSQL and pgvector store citation-bound research summaries, counterevidence, open questions, and source pointers.

Memory is used only for query understanding and retrieval planning. It never becomes factual evidence directly; every remembered claim must be revalidated against the current corpus before use.

### Persistence and Data Integrity

PostgreSQL is the source of truth for companies, filings, chunks, XBRL facts, research runs, claims, source fetches, market snapshots, web evidence, skill runs, and long-term memory. Redis loss affects performance and conversational continuity but not citation correctness.

Ingestion validates all documents and Company Facts before database writes. PostgreSQL advisory locks and transaction-scoped writes prevent partial or competing corpus updates. Financial and market values use Decimal-based storage to avoid binary floating-point drift.

### Market and Web Evidence

External evidence passes through provider-neutral gateways with allowlists, timeouts, rate limits, bounded result counts, immutable snapshots, and machine-readable failures. The model never receives arbitrary URL access.

Market snapshots preserve provider, symbol, exchange, currency, price, market status, as-of time, fetch time, and delay information. Event context is constrained to a time window around the quote and uses neutral wording when causality cannot be established.

### Reliability, Privacy, and Cost Control

Each run has hierarchical limits for model tokens, tool calls, retrieval rounds, web fallback, and market operations. Redis and corpus-versioned embeddings avoid repeated work. Missing or corrupt local model assets fail closed instead of downloading during normal runtime.

Privacy checks cover credentials, account identifiers, personal holdings, risk profiles, trade intent, and named financial relationships. Guarded output, traces, logs, memory, source references, and persisted reports use fixed redaction contracts and never echo rejected private values.

Typed error codes distinguish unsupported tickers, missing filings, exhausted budgets, stale market data, denied sources, and unavailable dependencies. Recoverable operations have tightly bounded retries; other failures become explicit partial or unavailable results.

## Engineering Quality

The repository uses deterministic offline evaluation, unit and integration testing, real SQLite persistence paths, PostgreSQL/pgvector migration tests, Redis cache checks, CLI and MCP contract tests, and protected provider workflows.

The current offline suite contains more than 3,300 passing tests. Canonical evaluation covers normal research, insufficient evidence, prohibited advice, cross-ticker attacks, market freshness, peer comparison, and quality screening with zero recorded source-policy, budget, or cross-ticker leakage violations.

The package is built from a frozen uv lockfile, checked with Ruff, and exercised through Docker Compose and GitHub Actions. External provider operations remain explicitly opt-in so default development and CI stay deterministic and credential-free.
