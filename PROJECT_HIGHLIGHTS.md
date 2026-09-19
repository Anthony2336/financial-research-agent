# Financial-Research-Agent Engineering Highlights

## Project Overview

Financial-Research-Agent turns questions about US public companies into reports backed by SEC filings and other approved sources. It supports thesis analysis, company and governance research, earnings review, industry and peer comparison, research-quality screening, and IEX market snapshots.

Reports distinguish verified facts, interpretations, counterevidence and missing information. Research-quality screening evaluates whether the evidence justifies further investigation; market context reports time-adjacent events without asserting causation.

## System Architecture

LangGraph coordinates planning, retrieval, analysis and citation validation through bounded state transitions. A fast model handles ambiguous routing and question planning; an analyst model produces structured reports from the collected evidence. FastMCP exposes typed research tools, while PostgreSQL with pgvector stores versioned corpora and report provenance, and Redis supplies optional caching and session memory.

Every factual claim is checked against its source and company scope before publication. Retrieval retries, web fallback and report repair have explicit budgets, so incomplete evidence produces a visible limitation instead of an unsupported answer.

## Technology Stack

| Area | Technology | Role |
|---|---|---|
| Runtime and CLI | Python, uv, Typer, Rich | Commands and reproducible environments |
| Orchestration | LangGraph, FastMCP | Workflow state and typed tools |
| Models and schemas | LangChain Core, OpenAI, Pydantic | Model calls and structured output |
| Retrieval | BGE-M3, pgvector, BM25, RRF | Dense and sparse search |
| Evidence preparation | FlashRank, LLMLingua | Reranking and context compression |
| Storage | PostgreSQL, SQLAlchemy, Alembic | Transactions, provenance and migrations |
| Cache and memory | Redis | TTL caching and session continuity |
| External data | SEC, Alpaca, Tavily | Filings, market data and approved web sources |
| Observability | Langfuse | Traces, cost tracking and evaluations |
| Delivery and testing | Docker Compose, GitHub Actions, pytest, Ruff | Packaging and automated checks |

## Core Engineering Design

### Evidence retrieval

Filings form immutable, versioned corpora with citations mapped to exact source spans. BGE-M3 and BM25 rankings are combined with Reciprocal Rank Fusion, then reranked with FlashRank; support and challenge queries seek both confirming and conflicting evidence.

Coverage checks allow one query rewrite and one allowlisted web fallback. BM25 corpus statistics are cached across queries, while evidence IDs and provenance always come from the current company and corpus scope.

### Citation and financial integrity

Filing, web, XBRL and market records retain separate source identities. Citation checks verify scope, original spans, source metadata and allowed report fields; a bounded repair can remove or downgrade unsupported content but cannot add new facts.

Financial observations preserve Decimal values, periods, currencies, units and definitions. Peer comparisons expose missing, conflicting and non-comparable values rather than silently converting or averaging them.

### Controlled research workflows

Versioned recipes specify required research sections, permitted tools, sources, output schemas and budgets. Company research combines business analysis, governance review and financial verification; earnings and industry research use their own evidence requirements.

Rules reject prohibited advice and prompt-injection requests before model execution. Tool access, cost accounting and source validation remain application-controlled, with retrieved documents treated as data rather than instructions.

### Context and memory

LLMLingua compresses evidence text while preserving citation boundaries and financial metadata. Token budgets prioritize control instructions and source identity, removing lower-value evidence and memory hints when needed.

Redis holds up to five session turns for 24 hours, and PostgreSQL stores citation-bound research summaries. Memory guides question interpretation and retrieval planning; remembered claims must be checked against the current corpus before use.

### Persistence and external data

PostgreSQL transactions and advisory locks coordinate corpus updates, and Alembic manages schema changes. Company Facts are written in batches; reports, claims and source-fetch records retain the identifiers needed to trace a result back to its inputs.

Web requests enforce source policy on every redirect. Alpaca observations retain IEX coverage, exchange, currency, observation time and delay, with unavailable or stale data reported explicitly.

### Observability and resource control

Langfuse traces connect routing, retrieval, model calls and persistence to one research run. Token, tool, retry and web-search limits bound execution, while local model assets avoid downloads during research startup.

Provider failures use typed errors, and optional caches degrade to misses. Privacy checks apply to outputs, traces and stored memory as well as the initial request.

## Engineering Quality

More than 3,300 automated tests cover routing, retrieval, citations, privacy, persistence, CLI behavior and packaging. PostgreSQL integration tests exercise migrations and vector retrieval; deterministic evaluation suites cover research, peer comparisons, freshness and evidence-quality decisions.

CI installs locked dependencies, runs offline checks and PostgreSQL acceptance, and builds the package. Live SEC, Alpaca, Tavily and Langfuse checks run separately with explicit credentials.
