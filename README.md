# Financial-Research-Agent

Financial-Research-Agent is a command-line research assistant for public US companies. It ingests supported SEC filings, retrieves attributable evidence, and renders guarded reports. It is not investment advice and does not recommend trades, predict prices, or manage portfolios.

See [Engineering Highlights](PROJECT_HIGHLIGHTS.md) for the architecture, technology stack, and major design decisions.

## Supported operator modes

Research reports separate cited facts, interpretations, counterevidence and unanswered questions. Filing-based modes use an ingested company corpus; market snapshots use Alpaca data. All modes resolve the ticker against local company metadata, populated during filing ingestion.

Before running the examples, complete [Local setup](#local-setup), configure
`OPENAI_API_KEY`, `FAST_MODEL` and `ANALYST_MODEL` as described in
[Model-backed research](#model-backed-research), and [ingest the company's filings](#live-sec-ingestion).
Market snapshots use Alpaca credentials instead. To try the project without a
model API key, use the explicit thesis command in [Offline demo](#offline-demo).

| Mode | Purpose | Input |
|---|---|---|
| `thesis` | Test a specific business hypothesis against disclosures | `--thesis` |
| `auto` | Route a research question to the appropriate workflow | `--question` |
| `company-profile` | Review the business, management, governance and financial evidence | `--question` |
| `earnings-review` | Examine reported changes, guidance and risks | `--question` |
| `industry-research` | Study an industry's structure and compare explicitly selected peers | `--question` |
| `quality-screen` | Assess whether the available evidence supports further research | `--question` |
| `market-snapshot` | Retrieve an IEX snapshot, recent daily bars and optional event context | `--question` |

### Thesis analysis

Use this mode to test a claim such as whether demand supports revenue growth. The report presents supporting disclosures, counterevidence and evidence gaps, with citations for retained facts.

```bash
uv run research NVDA --mode thesis \
  --thesis "Does data center demand support revenue growth despite deployment risks?"
```

### Automatic routing

Use `auto` when you have a question but do not want to choose a workflow. Rules resolve clear requests, and a fast model classifies ambiguous ones; the selected workflow still needs its own data and credentials. This earnings example requires configured model access and reviews the latest eligible disclosure already ingested into the database.

```bash
uv run research NVDA --mode auto \
  --question "What changed in the latest earnings disclosure?"
```

### Company research

Build a company overview covering its business model, competitive position, management, governance and capital allocation. The workflow also checks financial claims against available sources and identifies missing evidence.

```bash
uv run research NVDA --mode company-profile \
  --question "Explain the business model, governance, financial position and key risks."
```

### Earnings review

Review changes in reported performance, management guidance and disclosed risks. The report distinguishes supported figures from interpretations and flags values that cannot be verified or compared.

```bash
uv run research NVDA --mode earnings-review \
  --question "What changed in revenue, margins, guidance and risks in the latest filing?"
```

### Industry and peer research

Examine the value chain, supply and demand, competition, regulation and industry risks around a company. Add `--peer-ticker` and `--peer-scope` for an explicit comparison; incompatible periods, units or definitions remain marked as non-comparable.

```bash
uv run research NVDA --mode industry-research \
  --question "Describe the semiconductor value chain, competition and demand drivers."
```

See [Peer comparison](#peer-comparison) for a multi-company example.

### Research quality screening

Assess source coverage, freshness, financial comparability and unresolved gaps before spending more time on a company. The result is `worth_further_research`, `insufficient_information` or `out_of_scope`; it assesses evidence quality, not investment attractiveness.

```bash
uv run research NVDA --mode quality-screen \
  --question "Is the available evidence sufficient for further company research?"
```

### Market snapshots

Retrieve price, daily range, previous close and recent daily bars with provider and timestamp metadata. Add `--with-context` to search for authoritative events near the observation time; nearby events are not treated as proven causes of a price move.

```bash
uv run research NVDA --mode market-snapshot \
  --question "Show the current IEX snapshot and recent daily bars."
```

Market data requires Alpaca Basic credentials and covers IEX only, not the consolidated US market. Event context also requires Tavily; see [Market data](#market-data) for setup.

## Local setup

### Prerequisites

- Python 3.11
- `uv`
- Docker with Compose for PostgreSQL 16/pgvector and Redis 7
- provider credentials only for explicitly selected live/model-backed commands

Install the locked environment:

```bash
uv sync --frozen --all-groups
cp .env.example .env
```

PostgreSQL 16 with pgvector stores filings, embeddings, reports and research memory. Redis provides optional caching and session continuity. Start the services and apply database migrations:

```bash
docker compose up -d --wait postgres redis
uv run alembic upgrade head
```

The application rejects non-empty unversioned schemas with `DATABASE_UNVERSIONED_SCHEMA_REJECTED`; it never guesses or stamps their version.

## Offline demo

After database setup, run the bundled example without model or market-provider credentials. It uses a fixed filing fixture and deterministic analysis to demonstrate the report flow, rather than generating a live research answer.

Ingest the bundled fixture:

```bash
uv run ingest \
  --fixture src/fra/resources/nvda_10q.html \
  --ticker NVDA \
  --form 10-Q
```

Run the deterministic demo explicitly:

```bash
OFFLINE_DEMO=true uv run research NVDA \
  --mode thesis \
  --thesis "Does data center demand support revenue growth despite deployment risks?"
```

Safety still applies in demo mode. For example, `uv run research NVDA --thesis "Should I buy NVDA now?"` returns a fixed refusal before research dependencies are called.

## Model-backed research

Set the following values in `.env` or export them in your shell. `OPENAI_API_KEY`
must be a valid API key; `FAST_MODEL` and `ANALYST_MODEL` must be model IDs available
to that key. Replace the `...` values below before running the commands.

```bash
export OPENAI_API_KEY=...
export FAST_MODEL=...
export ANALYST_MODEL=...
# Optional: shared retrieval cache and session continuity
export REDIS_URL=redis://localhost:6379/0
export EMBEDDING_CACHE_DIR=/absolute/path/to/bge-cache
export TOKENIZER_CACHE_DIR=/absolute/path/to/tokenizer-cache
export CONTEXT_COMPRESSOR_CACHE_DIR=/absolute/path/to/llmlingua-cache
export RERANKER_CACHE_DIR=/absolute/path/to/flashrank-cache
```

Download the local model assets before running model-backed research:

```bash
uv run prefetch-model-assets
```

The command downloads BGE-M3, the configured models' tokenizers, LLMLingua and FlashRank assets. Research loads these assets locally; missing or corrupt files produce a configuration error.

For filing-based research, `--forms 10-K,10-Q,8-K` selects eligible filing types and `--as-of-date YYYY-MM-DD` sets the latest filing date. Ingest the relevant company filings before running one of the modes above.

Use the same `--session-id` for follow-up questions. With Redis enabled, the session keeps up to five turns for 24 hours; prior answers help interpret the question, while factual claims still require current evidence.

```bash
uv run research NVDA --mode auto --session-id nvda-review \
  --question "What drove data center revenue growth?"

uv run research NVDA --mode auto --session-id nvda-review \
  --question "What risks could affect that growth?"
```

## Live SEC ingestion

Set an identifying SEC user agent and ingest a bounded corpus:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"

uv run ingest \
  --ticker NVDA \
  --forms 10-K,10-Q,8-K \
  --as-of-date 2026-08-31
```

Supported filing types are `10-K`, `10-Q` and `8-K`. Ingestion also stores SEC Company Facts with their original periods, units and source metadata in a separate table; those records are not automatically used as report evidence.

## Allowlisted web fallback

Install the optional provider extra only when using Tavily:

```bash
uv sync --frozen --all-groups --extra web-search
export TAVILY_API_KEY=...
```

Configure verified issuer domains through `WEB_ISSUER_DOMAINS`. A research request cannot add an arbitrary URL or domain. The application reports insufficient evidence when no acceptable source survives validation.

## Market data

Configure Alpaca Basic:

```bash
export ALPACA_API_KEY_ID=...
export ALPACA_API_SECRET_KEY=...
export ALPACA_DATA_FEED=iex
export ALPACA_TRADING_ENVIRONMENT=paper
```

Run a snapshot or request neutral event context:

```bash
uv run research NVDA \
  --mode market-snapshot \
  --question "Show the current IEX snapshot and recent daily bars."

uv run research NVDA \
  --mode market-snapshot \
  --question "Show authoritative events near the IEX observation time." \
  --with-context
```

Reports identify provider, feed, exchange, currency, timestamps, market status, and delay. Missing, stale, or unavailable data is shown as such; event proximity is not presented as causation.

## Peer comparison

Supply peers and scope explicitly:

```bash
uv run research NVDA \
  --mode industry-research \
  --question "Compare exact reported operating metrics." \
  --peer-ticker AMD \
  --peer-ticker AVGO \
  --peer-scope "US data center accelerators"
```

At most three `--peer-ticker` values are accepted. `--peer-scope` is required; tickers must be distinct and have ingested corpora. Partial peer failures remain visible.

## Docker setup

Build the one-shot image, start dependencies, and migrate:

```bash
docker compose --profile cli build cli
docker compose up -d --wait postgres redis
docker compose --profile cli run --rm cli db-upgrade
```

Run the packaged fixture and offline gates:

```bash
docker compose --profile cli run --rm cli ingest \
  --fixture /app/src/fra/resources/nvda_10q.html \
  --ticker NVDA --form 10-Q

OFFLINE_DEMO=true docker compose --profile cli run --rm cli research NVDA \
  --thesis "Does data center demand support revenue growth despite deployment risks?"

docker compose --profile cli run --rm cli eval --suite p2
```

The container runs as the unprivileged `app` user. Stop services without deleting volumes:

```bash
docker compose down
```

Compose bind-mounts the four configured host cache directories read-only beneath
`/opt/financial-evidence-agent/models/`. Empty caches are sufficient for the deterministic
offline demo; model-backed container runs require `prefetch-model-assets` on the host first.

### Recovering from an unversioned legacy database

Back up or export the original database first. The following sequence preserves the original project and uses a separate project for recovery:

```bash
docker compose -p financial-evidence-agent stop postgres redis
docker compose -p financial-evidence-agent-legacy-recovery up -d postgres redis
docker compose -p financial-evidence-agent-legacy-recovery --profile cli build cli
docker compose -p financial-evidence-agent-legacy-recovery --profile cli run --rm cli db-upgrade
docker compose -p financial-evidence-agent-legacy-recovery --profile cli run --rm cli ingest --fixture /app/src/fra/resources/nvda_10q.html --ticker NVDA --form 10-Q
OFFLINE_DEMO=true docker compose -p financial-evidence-agent-legacy-recovery --profile cli run --rm cli research NVDA --thesis "Do cited disclosures support sustained data center demand?"
docker compose -p financial-evidence-agent-legacy-recovery --profile cli run --rm cli eval --suite p2
docker compose -p financial-evidence-agent-legacy-recovery down
docker compose -p financial-evidence-agent up -d postgres redis
```

The recovery `down` omits `-v`, so its volumes remain recoverable. Delete volumes only for an exact disposable project you created:

```bash
docker compose -p financial-evidence-agent-release-gate-20260831 down -v
```

Never apply that command to a regular or unknown project.

## Configuration reference

`.env.example` is the complete settings template. Common operator settings are summarized here.

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL/SQLAlchemy connection used by the app and Alembic |
| `REDIS_URL` | Optional provider/retrieval cache and session backend |
| `SEC_USER_AGENT` | Required identifying header for live SEC requests |
| `SEC_CACHE_TTL_SECONDS` | Validated SEC response TTL; default `86400` |
| `XBRL_MAX_RESPONSE_BYTES` | Company Facts response cap; default 25 MiB |
| `EMBEDDING_MODEL` / `EMBEDDING_CACHE_DIR` | BGE model identity and local assets |
| `TOKENIZER_CACHE_DIR` | Verified local tiktoken assets for exact provider token counting |
| `CONTEXT_COMPRESSOR_MODEL` / `CONTEXT_COMPRESSOR_CACHE_DIR` | Local-only LLMLingua model snapshot |
| `RERANKER_MODEL` / `RERANKER_CACHE_DIR` | Locked FlashRank 0.2.10 identity, assets, and integrity manifest |
| `OPENAI_API_KEY` / `FAST_MODEL` / `ANALYST_MODEL` | Model-backed research configuration |
| `TAVILY_API_KEY` | Optional allowlisted web provider |
| `WEB_ISSUER_DOMAINS` | Administrator-verified ticker-to-domain mapping |
| `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` | Market provider credentials |
| `ALPACA_DATA_FEED` | Fixed to `iex` |
| `MARKET_DATA_MAX_STALENESS_SECONDS` | Open-market freshness threshold |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Optional trace/dataset configuration |

Keep secrets in environment or secret storage, never in tracked files, CLI output, or trace metadata.

## Langfuse operations

Set all three Langfuse variables before an explicit remote operation:

```bash
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

Synchronize and run a P2 experiment:

```bash
uv run eval --suite p2 \
  --langfuse-dataset financial-research-agent-p2 \
  --sync-langfuse-dataset

uv run eval --suite p2 \
  --langfuse-dataset financial-research-agent-p2 \
  --langfuse-experiment p2-local
```

Dataset synchronization and experiments send data to the configured Langfuse instance. Evaluation without these options runs locally.

## Evaluation

```bash
uv run eval
uv run eval --suite p2
```

External datasets may be selected explicitly:

```bash
uv run eval --dataset /absolute/path/to/p0-p1-dataset.jsonl
uv run eval --suite p2 --p2-dataset /absolute/path/to/p2-dataset.jsonl
```

Evaluation exits non-zero if required cases, evaluators, source-policy checks, budget checks, or final leakage gates fail.

## Testing and CI

Run the default credential-free suite and package gates:

```bash
uv run pytest -m "not live_sec and not live_provider" -q
uv run ruff check .
uv build --offline --no-build-isolation
```

PostgreSQL acceptance creates and removes isolated UUID-named test databases:

```bash
RUN_POSTGRES_INTEGRATION=1 \
DATABASE_URL=postgresql+psycopg://financial_evidence:financial_evidence@localhost:5432/financial_evidence \
uv run pytest \
  tests/integration/test_database_migrations.py \
  tests/integration/test_postgres_fixture_ingest.py \
  tests/integration/test_postgres_vector_retrieval.py -q
```

Default CI runs locked offline checks, PostgreSQL acceptance, both eval commands, package build, and the isolated one-shot fixture/cache gate. It does not receive provider secrets.

### Protected live tests

Live tests require provider credentials and network access:

```bash
uv run pytest -m live_provider tests/integration/test_live_sec_opt_in.py -k live_p1_application_smoke -v
uv run pytest -m live_provider tests/integration/test_live_alpaca_opt_in.py -v
uv run pytest -m live_provider tests/integration/test_live_langfuse_opt_in.py -v
uv run pytest -m live_provider tests/integration/test_live_sec_opt_in.py -k real_tavily_provider_smoke -v
uv run pytest -m live_sec tests/integration/test_live_sec_opt_in.py -k real_nvda_sec_ingest_smoke -v
```

`.github/workflows/live-smoke.yml` runs provider checks manually with repository secrets. These checks are separate from the default offline suite.

## Troubleshooting

### `DATABASE_UNVERSIONED_SCHEMA_REJECTED`

Back up the database and use the non-destructive recovery sequence above. Do not stamp or rewrite it automatically.

### `EMBEDDING_MODEL_UNAVAILABLE`

Prefetch BGE-M3 and point `EMBEDDING_CACHE_DIR` to the same directory. Runtime startup does not download it.

### `RERANKER_MODEL_UNAVAILABLE`

Run `uv run prefetch-model-assets` with network access and point
`RERANKER_CACHE_DIR` at the resulting cache. Research validates the model manifest
before loading the reranker.

### `encoding_unavailable` or `model_unavailable`

Verify `TOKENIZER_CACHE_DIR` and `CONTEXT_COMPRESSOR_CACHE_DIR` point to the caches produced
by `prefetch-model-assets`. Missing or corrupt assets fail locally; runtime does not fetch a
replacement.

### `WEB_SEARCH_DEPENDENCY_UNAVAILABLE`

Install the `web-search` extra, or unset `TAVILY_API_KEY` for the labelled local-only path.

### `THESIS_CONFIGURATION_MISSING` or `P1_MODEL_CONFIGURATION_MISSING`

Set `OPENAI_API_KEY`, `FAST_MODEL`, and `ANALYST_MODEL`. Redis is optional in every
research mode; leave `REDIS_URL` empty to disable it. Use `OFFLINE_DEMO=true` only intentionally.

### `UNSUPPORTED_TICKER`

Ingest a supported filing corpus first. Model-backed modes do not auto-load demo data.

### `MARKET_DATA_CONFIGURATION_MISSING`

Set both Alpaca credential variables and keep `ALPACA_DATA_FEED=iex`.

### `STALE_MARKET_DATA`

The observation failed freshness/time validation. It is not replaced by an unlabelled cached price.

### `LANGFUSE_EXPORT_FAILED`

Check all three Langfuse values, dataset name, network access, and permissions. Offline eval remains available.

### Redis is unavailable

Cache operations degrade to misses. Research may be slower, but citation rules do not change.

### Docker ports are already in use

Identify the exact project using ports 5432/6379. Do not stop unknown projects or remove unknown volumes.

### P2 evaluation is red

Inspect the failing case and `cross_ticker_leakage_count`, source-policy violations, and budget violations. A red case invalidates acceptance until corrected and rerun.

## MCP integration

The CLI uses an in-process FastMCP server/client boundary. For MCP Inspector or another client, start the standalone stdio server:

```bash
uv run python -m fra.mcp_server
```

Tools expose company lookup, filing search, source spans, allowlisted web evidence and market data. Configure the same database, local model assets and provider credentials used by the CLI.
