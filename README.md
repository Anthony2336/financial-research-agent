# Financial Research Agent

Financial Research Agent is a command-line research assistant for public US companies. It ingests supported SEC filings, retrieves attributable evidence, and renders guarded reports. It is not investment advice and does not recommend trades, predict prices, or manage portfolios.

See [Engineering Highlights](PROJECT_HIGHLIGHTS.md) for the architecture, technology stack, and major design decisions.

The host CLI uses an in-process FastMCP server/client boundary. A standalone stdio entry point is available for MCP Inspector or external clients:

```bash
uv run python -m financial_evidence_agent.mcp_server
```

## Supported operator modes

| Mode | CLI input | Required runtime |
|---|---|---|
| `thesis` | `--thesis` | Ingested corpus, Redis, local model assets, and OpenAI; or explicit `OFFLINE_DEMO=true` |
| `auto` | `--question` | OpenAI model configuration |
| `company-profile` | `--question` | Ingested corpus, local model assets, and OpenAI |
| `earnings-review` | `--question` | Ingested corpus, local model assets, and OpenAI |
| `industry-research` | `--question` | Ingested corpus, local model assets, and OpenAI |
| `quality-screen` | `--question` | Ingested corpus, local model assets, and OpenAI |
| `market-snapshot` | `--question` | Alpaca Basic credentials; Tavily only for requested event context |

Only `10-K`, `10-Q`, and `8-K` filings are supported. Market output is labelled Alpaca Basic `IEX-only`, never consolidated. Users may supply at most three peers; the application does not discover or rank securities. For the architecture, source policy, provenance, memory, budgets, and guard behavior, see the engineering highlights above.

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

Start dependencies and migrate the database:

```bash
docker compose up -d --wait postgres redis
uv run alembic upgrade head
```

The application rejects non-empty unversioned schemas with `DATABASE_UNVERSIONED_SCHEMA_REJECTED`; it never guesses or stamps their version.

## Offline demo

Ingest the bundled fixture:

```bash
uv run ingest \
  --fixture src/financial_evidence_agent/resources/nvda_10q.html \
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

Configure the model path:

```bash
export OPENAI_API_KEY=...
export FAST_MODEL=...
export ANALYST_MODEL=...
export REDIS_URL=redis://localhost:6379/0
export EMBEDDING_CACHE_DIR=/absolute/path/to/bge-cache
export TOKENIZER_CACHE_DIR=/absolute/path/to/tokenizer-cache
export CONTEXT_COMPRESSOR_CACHE_DIR=/absolute/path/to/llmlingua-cache
export RERANKER_CACHE_DIR=/absolute/path/to/flashrank-cache
```

From an explicitly network-enabled operator environment, prefetch every approved model asset
before production research:

```bash
uv run prefetch-model-assets
```

This explicit command downloads BGE-M3, the exact FAST/ANALYST tokenizer encodings, the
LLMLingua snapshot, and the locked FlashRank 0.2.10 model. It also writes the FlashRank
integrity manifest required at runtime. Normal CLI startup and default/offline CI never invoke
the provisioner: tokenizer and LLMLingua load with local-only flags, and FlashRank validates
its local manifest before its constructor can reach the package download boundary.

Examples:

```bash
uv run research NVDA \
  --mode company-profile \
  --question "Summarize the business, governance, risks, and evidence gaps."

uv run research NVDA \
  --mode earnings-review \
  --question "What changed in the latest earnings disclosure?"

uv run research NVDA \
  --mode industry-research \
  --question "Describe industry structure, competition, and regulation."

uv run research NVDA \
  --mode quality-screen \
  --question "Is this guarded evidence sufficient for further research?"
```

Use `--session-id` only for bounded research continuity. Session and research memory are planning hints, never factual evidence; the isolated two-turn acceptance remains grounded and cache-backed in the executed matrix.

## Live SEC ingestion

Set an identifying SEC user agent and ingest a bounded corpus:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"

uv run ingest \
  --ticker NVDA \
  --forms 10-K,10-Q,8-K \
  --as-of-date 2026-08-31
```

Live ingestion also stores exact SEC Company Facts in a separate table. They are not silently promoted into filing/web evidence or automatically treated as live P1 reconciliation inputs. Protected SEC connectivity remains unverified unless the live workflow produces a passing run record.

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
  --fixture /app/src/financial_evidence_agent/resources/nvda_10q.html \
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
docker compose -p financial-evidence-agent-legacy-recovery --profile cli run --rm cli ingest --fixture /app/src/financial_evidence_agent/resources/nvda_10q.html --ticker NVDA --form 10-Q
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

These commands are external side effects. Ordinary offline evaluation does not contact Langfuse.

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

Run these only inside the authorized protected workflow:

```bash
uv run pytest -m live_provider tests/integration/test_live_sec_opt_in.py -k live_p1_application_smoke -v
uv run pytest -m live_provider tests/integration/test_live_alpaca_opt_in.py -v
uv run pytest -m live_provider tests/integration/test_live_langfuse_opt_in.py -v
uv run pytest -m live_provider tests/integration/test_live_sec_opt_in.py -k real_tavily_provider_smoke -v
uv run pytest -m live_sec tests/integration/test_live_sec_opt_in.py -k real_nvda_sec_ingest_smoke -v
```

`.github/workflows/live-smoke.yml` is manual and secret-scoped. No local passing run ID is recorded for these checks as of 2026-09-04, so provider connectivity remains unverified.

## Troubleshooting

### `DATABASE_UNVERSIONED_SCHEMA_REJECTED`

Back up the database and use the non-destructive recovery sequence above. Do not stamp or rewrite it automatically.

### `EMBEDDING_MODEL_UNAVAILABLE`

Prefetch BGE-M3 and point `EMBEDDING_CACHE_DIR` to the same directory. Runtime startup does not download it.

### `RERANKER_MODEL_UNAVAILABLE`

Run `uv run prefetch-model-assets` in the authorized network-enabled environment and keep
`RERANKER_CACHE_DIR` pointed at that cache. Production validates the manifest and never
substitutes identity reranking or downloads from the constructor.

### `encoding_unavailable` or `model_unavailable`

Verify `TOKENIZER_CACHE_DIR` and `CONTEXT_COMPRESSOR_CACHE_DIR` point to the caches produced
by `prefetch-model-assets`. Missing or corrupt assets fail locally; runtime does not fetch a
replacement.

### `WEB_SEARCH_DEPENDENCY_UNAVAILABLE`

Install the `web-search` extra, or unset `TAVILY_API_KEY` for the labelled local-only path.

### `THESIS_CONFIGURATION_MISSING` or `P1_MODEL_CONFIGURATION_MISSING`

Set `OPENAI_API_KEY`, `FAST_MODEL`, and `ANALYST_MODEL`; thesis production also requires Redis. Use `OFFLINE_DEMO=true` only intentionally.

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
