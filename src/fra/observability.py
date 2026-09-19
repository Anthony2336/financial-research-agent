"""Optional tracing adapters that remain inert without complete credentials."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from importlib import import_module
from typing import Literal, Protocol

from fra.config import Settings

logger = logging.getLogger(__name__)

ObservationKind = Literal[
    "agent",
    "chain",
    "generation",
    "tool",
    "retriever",
    "guardrail",
    "evaluator",
]


class ObservationHandle(Protocol):
    """One updateable observation inside a research run."""

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Attach safe result metadata to the observation."""


class TraceRun(Protocol):
    """One application run represented by a single observation tree."""

    @property
    def trace_id(self) -> str | None:
        """Return the exporter trace identifier when one exists."""

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: ObservationKind,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[ObservationHandle]:
        """Create one typed child observation."""

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Attach safe result metadata to the root observation."""


class TraceSink(Protocol):
    """Application tracing boundary with an explicit exporter lifecycle."""

    @contextmanager
    def run(
        self,
        *,
        run_id: str,
        input: object,
        metadata: Mapping[str, object],
    ) -> Iterator[TraceRun]:
        """Create one root agent observation for an application run."""

    def flush(self) -> None:
        """Flush pending exports without changing the research result."""

    def record(self, node: str, attributes: Mapping[str, object]) -> None:
        """Compatibility seam for direct graph and evaluation callers."""


class ExperimentCase(Protocol):
    """One canonical local case exportable to a Langfuse dataset item."""

    id: str

    def input_payload(self) -> dict[str, object]:
        """Return the strict JSON-safe task input payload."""

    def expected_payload(self) -> dict[str, object]:
        """Return the strict JSON-safe expected output payload."""

    def metadata_payload(self) -> dict[str, object]:
        """Return the strict JSON-safe experiment metadata payload."""


class LangfuseOperationError(RuntimeError):
    """Fail-closed explicit dataset/experiment operation error."""

    code = "LANGFUSE_EXPORT_FAILED"

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


@dataclass(slots=True)
class RunAccounting:
    """Aggregate only safe, provider-reported values from child observations."""

    input_tokens: int = 0
    output_tokens: int = 0
    provider_cost_usd: Decimal = Decimal(0)
    model_calls: int = 0
    cache_hits: int = 0
    model_ids: list[str] = field(default_factory=list)

    def begin_observation(
        self,
        *,
        kind: ObservationKind,
        metadata: Mapping[str, object] | None,
    ) -> _AccountingObservation:
        observation = _AccountingObservation(self)
        if kind == "generation":
            self.model_calls += 1
        observation.update_metadata(metadata)
        return observation

    def metadata(self, *, budget: object, status: str) -> dict[str, object]:
        state = getattr(budget, "state", None) if getattr(budget, "configured", False) else None
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "provider_cost_usd": float(self.provider_cost_usd),
            "model_ids": list(self.model_ids),
            "model_calls": self.model_calls,
            "tool_calls": int(getattr(state, "tool_calls", 0)),
            "retrieval_rounds": int(getattr(state, "retrieval_rounds", 0)),
            "web_calls": int(getattr(state, "web_calls", 0)),
            "cache_hits": self.cache_hits,
            "final_status": status,
        }


class _AccountingObservation:
    """Per-child de-duplication for cumulative provider response metadata."""

    def __init__(self, accounting: RunAccounting) -> None:
        self._accounting = accounting
        self._input_tokens = 0
        self._output_tokens = 0
        self._provider_cost_usd = Decimal(0)
        self._cache_hit = False

    def update_metadata(self, metadata: Mapping[str, object] | None) -> None:
        if metadata is None:
            return
        model = metadata.get("model")
        if isinstance(model, str) and model.strip() and len(model) <= 200:
            normalized = model.strip()
            if normalized not in self._accounting.model_ids:
                self._accounting.model_ids.append(normalized)
        usage = metadata.get("usage")
        if isinstance(usage, Mapping):
            self._update_int("input_tokens", usage, "_input_tokens")
            self._update_int("output_tokens", usage, "_output_tokens")
        cost = metadata.get("cost")
        value = _safe_cost(cost)
        if value is not None:
            if value >= self._provider_cost_usd:
                self._accounting.provider_cost_usd += value - self._provider_cost_usd
                self._provider_cost_usd = value
        if metadata.get("cache_hit") is True and not self._cache_hit:
            self._accounting.cache_hits += 1
            self._cache_hit = True

    def _update_int(self, field_name: str, usage: Mapping[object, object], attr: str) -> None:
        value = _safe_non_negative_int(usage.get(field_name))
        if value is None:
            return
        previous = getattr(self, attr)
        if value >= previous:
            total = getattr(self._accounting, field_name) + value - previous
            setattr(self._accounting, field_name, total)
            setattr(self, attr, value)


class _AccountedObservation:
    def __init__(self, observation: ObservationHandle, accounting: _AccountingObservation) -> None:
        self._observation = observation
        self._accounting = accounting

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._accounting.update_metadata(metadata)
        self._observation.update(output=output, metadata=metadata)


class AccountingTraceRun:
    """Trace wrapper that observes child metadata without changing exporter behavior."""

    def __init__(self, trace: TraceRun, accounting: RunAccounting) -> None:
        self._trace = trace
        self._accounting = accounting

    @property
    def trace_id(self) -> str | None:
        return self._trace.trace_id

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: ObservationKind,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[ObservationHandle]:
        accounting_observation = self._accounting.begin_observation(
            kind=kind,
            metadata=metadata,
        )
        with self._trace.observation(
            name=name,
            kind=kind,
            input=input,
            metadata=metadata,
        ) as observation:
            yield _AccountedObservation(observation, accounting_observation)

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self._trace.update(output=output, metadata=metadata)


_FINAL_ROOT_METADATA_KEYS = frozenset(
    {
        "effective_intent",
        "recipe_names",
        "recipe_versions",
        "corpus_scope",
        "prompt_versions",
        "requested_as_of_dates",
        "evidence_cutoff_dates",
        "source_policy_version",
        "source_policy_versions",
        "source_refs",
        "information_sufficiency",
        "guard_errors",
        "cross_ticker_leakage_count",
        "cross_ticker_rejection_count",
        "quality_decision",
        "quality_effective_date",
        "quality_max_source_age_days",
        "primary_ticker",
        "peer_tickers",
        "peer_scope",
        "missing_tickers",
        "guard_notes",
        "error_codes",
        "comparison_count",
        "provider",
        "feed",
        "coverage",
        "exchange",
        "currency",
    }
)


def complete_root_metadata(
    metadata: Mapping[str, object],
    *,
    accounting: RunAccounting,
    budget: object,
    status: str,
) -> dict[str, object]:
    """Return the strict final root allowlist plus application-owned totals."""

    safe_metadata = _safe_root_metadata(metadata)
    return {**safe_metadata, **accounting.metadata(budget=budget, status=status)}


def _safe_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_cost(value: object) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not numeric.is_finite() or numeric < 0 or numeric > Decimal("1e100"):
        return None
    try:
        exported = float(numeric)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(exported) or (numeric != 0 and exported == 0):
        return None
    return numeric


_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,200}\Z")
_CODE = re.compile(r"[A-Z0-9_.:-]{1,200}\Z")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_SAFE_PEER_SCOPE = "Removed by output safety guard."


def _safe_root_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    safe: dict[str, object] = {}
    for key, value in metadata.items():
        if key not in _FINAL_ROOT_METADATA_KEYS:
            continue
        if key in {
            "effective_intent", "information_sufficiency", "provider", "feed", "coverage",
            "exchange", "currency", "source_policy_version", "quality_decision", "primary_ticker",
        }:
            if _safe_identifier(value):
                safe[key] = value
        elif key in {
            "recipe_names", "recipe_versions", "corpus_scope", "prompt_versions",
            "source_policy_versions", "source_refs", "peer_tickers", "missing_tickers",
        }:
            values = _safe_identifier_list(value)
            if values is not None:
                safe[key] = values
        elif key in {"requested_as_of_dates", "evidence_cutoff_dates"}:
            values = _safe_date_list(value)
            if values is not None:
                safe[key] = values
        elif key in {"guard_errors", "error_codes"}:
            values = _safe_code_list(value)
            if values is not None:
                safe[key] = values
        elif key in {
            "cross_ticker_leakage_count", "cross_ticker_rejection_count",
            "quality_max_source_age_days", "comparison_count",
        }:
            if _safe_non_negative_int(value) is not None:
                safe[key] = value
        elif key == "quality_effective_date" and isinstance(value, str) and _DATE.fullmatch(value):
            safe[key] = value
        elif key == "peer_scope" and value == _SAFE_PEER_SCOPE:
            safe[key] = value
    return safe


def _safe_identifier(value: object) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def _safe_identifier_list(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)) or len(value) > 50:
        return None
    values = list(value)
    return values if all(_safe_identifier(item) for item in values) else None


def _safe_date_list(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)) or len(value) > 50:
        return None
    values = list(value)
    return values if all(
        isinstance(item, str) and _DATE.fullmatch(item) for item in values
    ) else None


def _safe_code_list(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)) or len(value) > 50:
        return None
    values = list(value)
    return values if all(
        isinstance(item, str) and _CODE.fullmatch(item) for item in values
    ) else None


class _NoopObservation:
    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        del output, metadata


class NoopTraceRun(_NoopObservation):
    """No-network run handle with the same control flow as an exporter."""

    @property
    def trace_id(self) -> str | None:
        return None

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: ObservationKind,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[ObservationHandle]:
        del name, kind, input, metadata
        yield _NoopObservation()


_NOOP_RUN = NoopTraceRun()
_CURRENT_RUN: ContextVar[TraceRun] = ContextVar(
    "fra_trace_run",
    default=_NOOP_RUN,
)


@contextmanager
def bind_trace_run(run: TraceRun) -> Iterator[None]:
    """Expose the application-owned run to provider and tool adapters."""
    token: Token[TraceRun] = _CURRENT_RUN.set(run)
    try:
        yield
    finally:
        _CURRENT_RUN.reset(token)


def current_trace_run() -> TraceRun:
    """Return the active run or the explicit no-op context."""
    return _CURRENT_RUN.get()


@contextmanager
def observe(
    *,
    name: str,
    kind: ObservationKind,
    input: object | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Iterator[ObservationHandle]:
    """Create a child on the active root, swallowing exporter-only failures."""
    try:
        manager = current_trace_run().observation(
            name=name,
            kind=kind,
            input=input,
            metadata=metadata,
        )
        observation = manager.__enter__()
    except Exception:
        logger.warning("trace observation export failed: %s", name)
        yield _NoopObservation()
        return

    body_error: BaseException | None = None
    try:
        yield observation
    except BaseException as error:
        body_error = error
        raise
    finally:
        try:
            manager.__exit__(
                type(body_error) if body_error is not None else None,
                body_error,
                body_error.__traceback__ if body_error is not None else None,
            )
        except Exception:
            logger.warning("trace observation close failed: %s", name)


class NoopTraceSink:
    """Default sink for offline and no-key execution."""

    @contextmanager
    def run(
        self,
        *,
        run_id: str,
        input: object,
        metadata: Mapping[str, object],
    ) -> Iterator[TraceRun]:
        del run_id, input, metadata
        yield _NOOP_RUN

    def flush(self) -> None:
        return None

    def record(self, node: str, attributes: Mapping[str, object]) -> None:
        del node, attributes


class _LangfuseObservation:
    def __init__(self, observation: object) -> None:
        self._observation = observation

    def update(
        self,
        *,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        values: dict[str, object] = {}
        if output is not None:
            values["output"] = output
        if metadata is not None:
            values["metadata"] = dict(metadata)
        if not values:
            return
        try:
            getattr(self._observation, "update")(**values)
        except Exception:
            logger.warning("trace observation update failed")


class _LangfuseRun(_LangfuseObservation):
    def __init__(self, client: object, root: object) -> None:
        super().__init__(root)
        self._client = client
        self._root = root

    @property
    def trace_id(self) -> str | None:
        value = getattr(self._root, "trace_id", None)
        return value if isinstance(value, str) and value else None

    @contextmanager
    def observation(
        self,
        *,
        name: str,
        kind: ObservationKind,
        input: object | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[ObservationHandle]:
        values: dict[str, object] = {"as_type": kind, "name": name}
        if input is not None:
            values["input"] = input
        if metadata is not None:
            values["metadata"] = dict(metadata)
        try:
            manager = getattr(self._client, "start_as_current_observation")(**values)
            observation = manager.__enter__()
        except Exception:
            logger.warning("trace observation export failed: %s", name)
            yield _NoopObservation()
            return
        body_error: BaseException | None = None
        try:
            yield _LangfuseObservation(observation)
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                manager.__exit__(
                    type(body_error) if body_error is not None else None,
                    body_error,
                    body_error.__traceback__ if body_error is not None else None,
                )
            except Exception:
                logger.warning("trace observation close failed: %s", name)


class LangfuseTraceSink:
    """Safe Langfuse v4 adapter around an already-constructed client."""

    def __init__(self, client: object) -> None:
        self._client = client

    @contextmanager
    def run(
        self,
        *,
        run_id: str,
        input: object,
        metadata: Mapping[str, object],
    ) -> Iterator[TraceRun]:
        values = {
            "as_type": "agent",
            "name": "financial-research-agent.run",
            "input": input,
            "metadata": {**dict(metadata), "run_id": run_id},
        }
        try:
            manager = getattr(self._client, "start_as_current_observation")(**values)
            root = manager.__enter__()
        except Exception:
            logger.warning("trace root export failed")
            yield _NOOP_RUN
            return

        body_error: BaseException | None = None
        try:
            yield _LangfuseRun(self._client, root)
        except BaseException as error:
            body_error = error
            raise
        finally:
            try:
                manager.__exit__(
                    type(body_error) if body_error is not None else None,
                    body_error,
                    body_error.__traceback__ if body_error is not None else None,
                )
            except Exception:
                logger.warning("trace root close failed")

    def flush(self) -> None:
        try:
            getattr(self._client, "flush")()
        except Exception:
            logger.warning("trace flush failed")

    def record(self, node: str, attributes: Mapping[str, object]) -> None:
        metadata = _safe_legacy_metadata(attributes)
        if current_trace_run() is not _NOOP_RUN:
            with observe(name=node, kind="chain", metadata=metadata):
                pass
            return
        try:
            manager = getattr(self._client, "start_as_current_observation")(
                as_type="chain",
                name=node,
                metadata=metadata,
            )
            manager.__enter__()
            manager.__exit__(None, None, None)
        except Exception:
            logger.warning("standalone trace observation export failed: %s", node)


def build_trace_sink(settings: Settings) -> TraceSink:
    """Build Langfuse only when public key, secret key, and host are all present."""

    if not settings.has_complete_langfuse_credentials:
        return NoopTraceSink()

    try:
        langfuse_module = import_module("langfuse")
        langfuse_type = getattr(langfuse_module, "Langfuse")
        client = langfuse_type(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
    except Exception:
        logger.warning("trace client construction failed")
        return NoopTraceSink()
    return LangfuseTraceSink(client)


def _safe_legacy_metadata(attributes: Mapping[str, object]) -> dict[str, object]:
    allowed = {
        "case_count",
        "case_id",
        "citation_validity",
        "claim_supported",
        "cost_source",
        "cost_usd",
        "coverage_reasons",
        "pass_rate",
        "passed",
        "passed_count",
        "recipe_name",
        "recipe_names",
        "recipe_version",
        "recipe_versions",
        "retrieval_rounds",
        "source_ids",
        "status",
        "ticker",
        "web_calls",
    }
    return {
        key: value
        for key, value in attributes.items()
        if key.casefold() in allowed
    }

def build_langfuse_operation_client(settings: Settings) -> object:
    """Construct the explicit Langfuse client used by dataset sync and experiments."""

    if not settings.has_complete_langfuse_credentials:
        raise LangfuseOperationError(
            "complete Langfuse credentials are required for dataset synchronization and experiments"
        )
    try:
        langfuse_module = import_module("langfuse")
        langfuse_type = getattr(langfuse_module, "Langfuse")
        return langfuse_type(
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
        )
    except Exception as error:
        raise LangfuseOperationError("failed to construct Langfuse client") from error


def sync_langfuse_dataset(
    client: object,
    name: str,
    cases: Sequence[ExperimentCase],
) -> None:
    """Synchronize the canonical local cases into one Langfuse dataset."""

    _ensure_dataset_exists(client, name)
    for case in cases:
        try:
            getattr(client, "create_dataset_item")(
                dataset_name=name,
                id=case.id,
                input=_json_safe_mapping(case.input_payload(), label=f"{case.id} input"),
                expected_output=_json_safe_mapping(
                    case.expected_payload(),
                    label=f"{case.id} expected output",
                ),
                metadata=_json_safe_mapping(
                    case.metadata_payload(),
                    label=f"{case.id} metadata",
                ),
            )
        except LangfuseOperationError:
            raise
        except Exception as error:
            raise LangfuseOperationError(
                f"failed to upsert Langfuse dataset item '{case.id}'"
            ) from error
    try:
        getattr(client, "flush")()
    except Exception as error:
        raise LangfuseOperationError(
            f"failed to flush Langfuse dataset synchronization for '{name}'"
        ) from error


def _ensure_dataset_exists(client: object, name: str) -> None:
    try:
        getattr(client, "get_dataset")(name)
        return
    except Exception as error:
        if not _is_not_found_error(error):
            raise LangfuseOperationError(
                f"failed to fetch Langfuse dataset '{name}'"
            ) from error
    try:
        getattr(client, "create_dataset")(name=name)
    except Exception as error:
        raise LangfuseOperationError(
            f"failed to create Langfuse dataset '{name}'"
        ) from error


def _is_not_found_error(error: Exception) -> bool:
    return getattr(error, "status_code", None) == 404


def _json_safe_mapping(payload: Mapping[str, object], *, label: str) -> dict[str, object]:
    try:
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    except TypeError as error:
        raise LangfuseOperationError(f"{label} must be JSON-serializable") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise LangfuseOperationError(f"{label} must serialize to a JSON object")
    return decoded
