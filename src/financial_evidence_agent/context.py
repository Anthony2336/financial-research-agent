"""Focused deterministic context allocation, evidence compression, and run budgets."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha1, sha256
from html import escape
from os import environ
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from financial_evidence_agent.prompts import STRICT_GROUNDING_SYSTEM


class ContextCompressor(Protocol):
    """Compress evidence body text without receiving citation metadata."""

    def compress(self, body: str, *, max_tokens: int) -> str:
        """Return a body no larger than the requested evidence-token allocation."""


class ContextCompressionErrorCode(StrEnum):
    """Stable production compression failure categories."""

    PACKAGE_UNAVAILABLE = "package_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    INVALID_OUTPUT = "invalid_output"


class ContextCompressionError(RuntimeError):
    """Typed actionable compressor failure safe to expose at a runtime boundary."""

    def __init__(self, code: ContextCompressionErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


class LazyLLMLinguaCompressor:
    """Load LLMLingua and its configured model only when oversized evidence needs it."""

    def __init__(
        self,
        *,
        model_name: str,
        device_map: str | None = "cpu",
        model_cache_dir: str | None = None,
        tokenizer_cache_dir: str | None = None,
        compressor_factory: Callable[[str, str | None], object] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device_map = device_map
        self._model_cache_dir = model_cache_dir
        self._tokenizer_cache_dir = tokenizer_cache_dir
        self._factory = compressor_factory
        self._compressor: object | None = None

    def compress(self, body: str, *, max_tokens: int) -> str:
        compressor = self._load()
        compress_prompt = getattr(compressor, "compress_prompt", None)
        if not callable(compress_prompt):
            raise ContextCompressionError(
                ContextCompressionErrorCode.INVALID_OUTPUT,
                "LLMLingua compressor does not expose compress_prompt",
            )
        try:
            output = compress_prompt(
                [body],
                target_token=max_tokens,
                force_reserve_digit=True,
            )
        except OSError:
            raise ContextCompressionError(
                ContextCompressionErrorCode.MODEL_UNAVAILABLE,
                "The configured LLMLingua model could not compress evidence; "
                "install or cache the configured model before production research",
            ) from None
        if not isinstance(output, Mapping) or not isinstance(output.get("compressed_prompt"), str):
            raise ContextCompressionError(
                ContextCompressionErrorCode.INVALID_OUTPUT,
                "LLMLingua returned an invalid compressed_prompt payload",
            )
        return output["compressed_prompt"].strip()

    def _load(self) -> object:
        if self._compressor is not None:
            return self._compressor
        if self._factory is not None:
            try:
                self._compressor = self._factory(self._model_name, self._device_map)
            except ModuleNotFoundError:
                raise ContextCompressionError(
                    ContextCompressionErrorCode.PACKAGE_UNAVAILABLE,
                    "LLMLingua support is unavailable; install the 'llmlingua' project "
                    "dependency before running production evidence compression",
                ) from None
            except OSError:
                raise ContextCompressionError(
                    ContextCompressionErrorCode.MODEL_UNAVAILABLE,
                    "The configured LLMLingua model is unavailable; install or cache it "
                    "before production research",
                ) from None
            return self._compressor
        try:
            local_model_path = _resolve_local_hf_snapshot(
                self._model_name,
                self._model_cache_dir,
            )
            _validated_tokenizer_cache_path(
                "gpt-3.5-turbo",
                self._tokenizer_cache_dir,
            )
            self._compressor = _build_llmlingua_compressor(
                local_model_path,
                self._device_map,
                self._tokenizer_cache_dir,
            )
        except ModuleNotFoundError:
            raise ContextCompressionError(
                ContextCompressionErrorCode.PACKAGE_UNAVAILABLE,
                "LLMLingua support is unavailable; install the 'llmlingua' project "
                "dependency before running production evidence compression",
            ) from None
        except (OSError, TokenCounterError, ValueError):
            raise ContextCompressionError(
                ContextCompressionErrorCode.MODEL_UNAVAILABLE,
                "The configured LLMLingua model is unavailable; install or cache it "
                "before production research",
            ) from None
        return self._compressor


_LOCAL_MODEL_LOAD_LOCK = RLock()


def _resolve_local_hf_snapshot(model_name: str, cache_dir: str | None) -> Path:
    if not cache_dir:
        raise OSError("local model cache is not configured")
    from huggingface_hub import snapshot_download

    cache_root = Path(cache_dir).resolve(strict=True)
    snapshot = Path(
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_root),
            local_files_only=True,
        )
    ).resolve(strict=True)
    snapshot.relative_to(cache_root)
    config = json.loads((snapshot / "config.json").read_text("utf-8"))
    if not isinstance(config, dict) or not any(
        path.is_file() and path.stat().st_size > 0
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in snapshot.glob(pattern)
    ):
        raise OSError("local model snapshot is incomplete")
    return snapshot


def _build_llmlingua_compressor(
    model_path: Path,
    device_map: str | None,
    tokenizer_cache_dir: str | None,
) -> object:
    from llmlingua import PromptCompressor

    if not tokenizer_cache_dir:
        raise OSError("local tokenizer cache is not configured")
    with _LOCAL_MODEL_LOAD_LOCK:
        previous = environ.get("TIKTOKEN_CACHE_DIR")
        environ["TIKTOKEN_CACHE_DIR"] = tokenizer_cache_dir
        try:
            return PromptCompressor(
                model_name=str(model_path),
                device_map=device_map,
                model_config={
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
                use_llmlingua2=True,
            )
        finally:
            if previous is None:
                environ.pop("TIKTOKEN_CACHE_DIR", None)
            else:
                environ["TIKTOKEN_CACHE_DIR"] = previous


class ContextLimits(BaseModel):
    """Frozen allocations from the numbered component design."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_control_tokens: int = Field(default=500, ge=1)
    max_task_tokens: int = Field(default=500, ge=1)
    max_memory_tokens: int = Field(default=300, ge=0)
    max_evidence_tokens: int = Field(default=3_000, ge=0)
    max_total_tokens: int = Field(default=4_300, ge=1)


class MemoryHint(BaseModel):
    """Future-compatible source-pointer hint; never a factual evidence source."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1)
    score: float = 0.0
    identity: Literal["session", "research"] | None = None
    pointer_id: str | None = Field(default=None, min_length=1, max_length=128)


class ContextEvidence(BaseModel):
    """One canonical evidence body plus protected citation metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    evidence_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    ticker: str = Field(min_length=1, max_length=10)
    body: str = Field(min_length=1)
    score: float
    source_url: str = Field(min_length=1)
    date: str | None = None
    accession_no: str | None = None
    section: str | None = None
    raw_start: int | None = Field(default=None, ge=0)
    raw_end: int | None = Field(default=None, gt=0)
    form: str | None = None
    title: str | None = None
    source_kind: str | None = None
    source_tier: str | None = None
    fetched_at: str | None = None
    citation_bindings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_raw_boundary(self) -> ContextEvidence:
        if (self.raw_start is None) is not (self.raw_end is None):
            raise ValueError("raw citation boundaries must be present together")
        if self.raw_start is not None and self.raw_end is not None:
            if self.raw_end <= self.raw_start:
                raise ValueError("raw_end must be greater than raw_start")
        return self


class BoundedEvidence(ContextEvidence):
    """Provider body with its untouched canonical citation source retained separately."""

    original_body: str = Field(min_length=1, exclude=True)


class BoundedContext(BaseModel):
    """Allocated provider context with deterministic drop provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    control: str
    task: str
    memory_hints: tuple[MemoryHint, ...]
    evidence: tuple[BoundedEvidence, ...]
    dropped_evidence_ids: tuple[str, ...] = ()
    dropped_memory_hints: tuple[str, ...] = ()
    control_tokens: int = Field(ge=0)
    task_tokens: int = Field(ge=0)
    memory_tokens: int = Field(ge=0)
    evidence_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

    def render_task(self) -> str:
        """Render the complete task allocation, including its container boundary."""
        return f"<task>\n{self.task}\n</task>\n"

    def render_memory(self) -> str:
        """Render the complete memory allocation, including labels and boundaries."""
        memory = "\n".join(_render_memory_hint(hint) for hint in self.memory_hints)
        return f'<memory_hints untrusted="true">\n{memory}\n</memory_hints>\n'

    def render_evidence(self) -> str:
        """Render the complete evidence allocation and protected citation metadata."""
        evidence = "\n".join(_render_evidence_block(item) for item in self.evidence)
        return (
            '<evidence_blocks untrusted="true" instruction_authority="none">\n'
            f"{evidence}\n"
            "</evidence_blocks>"
        )

    def render(self) -> str:
        """Render only lower-priority HumanMessage data with explicit trust boundaries."""
        return self.render_task() + self.render_memory() + self.render_evidence()


class ContextBudgetError(RuntimeError):
    """Raised when non-truncatable control or task data exceeds its allocation."""


TokenCounter = Callable[[str], int]


class TokenCounterErrorCode(StrEnum):
    """Stable configured-provider tokenizer failure categories."""

    PACKAGE_UNAVAILABLE = "package_unavailable"
    ENCODING_UNAVAILABLE = "encoding_unavailable"


class TokenCounterError(RuntimeError):
    """Typed local tokenizer/configuration failure without provider-name disclosure."""

    def __init__(self, code: TokenCounterErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}: {detail}")


@dataclass(frozen=True, slots=True)
class TokenEncodingAsset:
    """One approved local tiktoken BPE asset and immutable encoding metadata."""

    encoding_name: str
    cache_url: str
    expected_hash: str
    pattern: str
    special_tokens: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "special_tokens",
            MappingProxyType(dict(self.special_tokens)),
        )


_CL100K_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| "
    r"?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)
_O200K_PATTERN = "|".join(
    (
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    )
)
_CL100K_ASSET = TokenEncodingAsset(
    encoding_name="cl100k_base",
    cache_url="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    pattern=_CL100K_PATTERN,
    special_tokens={
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    },
)
_O200K_ASSET = TokenEncodingAsset(
    encoding_name="o200k_base",
    cache_url="https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    expected_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    pattern=_O200K_PATTERN,
    special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
)
_MODEL_ASSETS: dict[str, TokenEncodingAsset] = {
    "gpt-5": _O200K_ASSET,
    "gpt-5-mini": _O200K_ASSET,
    "gpt-5-nano": _O200K_ASSET,
    "gpt-4.1": _O200K_ASSET,
    "gpt-4.1-mini": _O200K_ASSET,
    "gpt-4.1-nano": _O200K_ASSET,
    "gpt-4o": _O200K_ASSET,
    "gpt-4o-mini": _O200K_ASSET,
    "o1": _O200K_ASSET,
    "o1-mini": _O200K_ASSET,
    "o3": _O200K_ASSET,
    "o3-mini": _O200K_ASSET,
    "o4-mini": _O200K_ASSET,
    "gpt-4": _CL100K_ASSET,
    "gpt-3.5-turbo": _CL100K_ASSET,
    "gpt-35-turbo": _CL100K_ASSET,
}
_VERSIONED_MODEL_ASSETS: tuple[tuple[str, TokenEncodingAsset], ...] = (
    ("gpt-5", _O200K_ASSET),
    ("gpt-5-mini", _O200K_ASSET),
    ("gpt-5-nano", _O200K_ASSET),
    ("gpt-4.1", _O200K_ASSET),
    ("gpt-4.1-mini", _O200K_ASSET),
    ("gpt-4.1-nano", _O200K_ASSET),
    ("gpt-4o", _O200K_ASSET),
    ("gpt-4o-mini", _O200K_ASSET),
    ("o1", _O200K_ASSET),
    ("o3", _O200K_ASSET),
    ("o4-mini", _O200K_ASSET),
    ("gpt-4", _CL100K_ASSET),
    ("gpt-3.5-turbo", _CL100K_ASSET),
    ("gpt-35-turbo", _CL100K_ASSET),
)


class LazyTiktokenTokenCounter:
    """Construct an exact tokenizer only from a verified approved local cache asset."""

    def __init__(
        self,
        *,
        model_name: str,
        cache_dir: str | None = None,
        model_assets: Mapping[str, TokenEncodingAsset] | None = None,
        encoding_factory: Callable[[TokenEncodingAsset, Path], object] | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model_assets = dict(model_assets or {})
        self._factory = encoding_factory or _build_local_tiktoken_encoding
        self._encoding: object | None = None

    def __call__(self, value: str) -> int:
        encoding = self._load()
        encode = getattr(encoding, "encode", None)
        if not callable(encode):
            raise TokenCounterError(
                TokenCounterErrorCode.ENCODING_UNAVAILABLE,
                "The configured provider model tokenizer has no encode operation",
            )
        tokens = encode(value)
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise TokenCounterError(
                TokenCounterErrorCode.ENCODING_UNAVAILABLE,
                "The configured provider model tokenizer returned an invalid token sequence",
            )
        return len(tokens)

    def _load(self) -> object:
        if self._encoding is not None:
            return self._encoding
        cache_dir = (
            self._cache_dir
            or environ.get("TOKENIZER_CACHE_DIR")
            or environ.get("TIKTOKEN_CACHE_DIR")
            or environ.get("DATA_GYM_CACHE_DIR")
        )
        asset, cache_path = _validated_tokenizer_cache_path(
            self._model_name,
            cache_dir,
            model_assets=self._model_assets,
        )
        try:
            self._encoding = self._factory(asset, cache_path)
        except ModuleNotFoundError:
            raise TokenCounterError(
                TokenCounterErrorCode.PACKAGE_UNAVAILABLE,
                "Exact provider token counting requires the installed tiktoken dependency",
            ) from None
        except Exception:
            raise TokenCounterError(
                TokenCounterErrorCode.ENCODING_UNAVAILABLE,
                "The verified local provider-model encoding could not be constructed",
            ) from None
        return self._encoding


def _validated_tokenizer_cache_path(
    model_name: str,
    cache_dir: str | None,
    *,
    model_assets: Mapping[str, TokenEncodingAsset] | None = None,
) -> tuple[TokenEncodingAsset, Path]:
    asset = (model_assets or {}).get(model_name) or _asset_for_model(model_name)
    if asset is None:
        raise TokenCounterError(
            TokenCounterErrorCode.ENCODING_UNAVAILABLE,
            "Exact token counting requires a recognized OpenAI model name",
        )
    if not cache_dir:
        raise TokenCounterError(
            TokenCounterErrorCode.ENCODING_UNAVAILABLE,
            "Set TOKENIZER_CACHE_DIR or TIKTOKEN_CACHE_DIR and prefetch the exact "
            "configured-model encoding asset",
        )
    cache_path = Path(cache_dir) / sha1(asset.cache_url.encode()).hexdigest()
    try:
        payload = cache_path.read_bytes()
    except OSError:
        raise TokenCounterError(
            TokenCounterErrorCode.ENCODING_UNAVAILABLE,
            "Prefetch the exact configured-model encoding asset into the approved "
            "tokenizer cache before production research",
        ) from None
    if sha256(payload).hexdigest() != asset.expected_hash:
        raise TokenCounterError(
            TokenCounterErrorCode.ENCODING_UNAVAILABLE,
            "The approved tokenizer cache asset failed hash verification; prefetch a "
            "verified replacement",
        )
    return asset, cache_path


def _asset_for_model(model_name: str) -> TokenEncodingAsset | None:
    if asset := _MODEL_ASSETS.get(model_name):
        return asset
    return next(
        (
            asset
            for base_name, asset in _VERSIONED_MODEL_ASSETS
            if re.fullmatch(rf"{re.escape(base_name)}-\d{{4}}-\d{{2}}-\d{{2}}", model_name)
        ),
        None,
    )


def _build_local_tiktoken_encoding(
    asset: TokenEncodingAsset,
    cache_path: Path,
) -> object:
    from tiktoken import Encoding
    from tiktoken.load import load_tiktoken_bpe

    mergeable_ranks = load_tiktoken_bpe(
        str(cache_path),
        expected_hash=asset.expected_hash,
    )
    return Encoding(
        name=asset.encoding_name,
        pat_str=asset.pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=dict(asset.special_tokens),
    )


class ContextBuilder:
    """Allocate control/task/memory/evidence without modifying citation metadata."""

    def __init__(
        self,
        *,
        control: str = STRICT_GROUNDING_SYSTEM,
        limits: ContextLimits | None = None,
        compressor: ContextCompressor | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._control = control
        self._limits = limits or ContextLimits()
        self._compressor = compressor
        self._count = token_counter or _utf8_byte_upper_bound
        control_tokens = self._count(control)
        if control_tokens > self._limits.max_control_tokens:
            raise ContextBudgetError(
                f"control requires {control_tokens} tokens; "
                f"limit is {self._limits.max_control_tokens}; control is never truncated"
            )
        self._control_tokens = control_tokens

    def build(
        self,
        task: str,
        memory_hints: Sequence[MemoryHint],
        evidence: Sequence[ContextEvidence],
    ) -> BoundedContext:
        """Build one bounded context for a fixed task string."""
        return self.build_dynamic(
            task_factory=lambda retained: task,
            memory_hints=memory_hints,
            evidence=evidence,
        )

    def build_dynamic(
        self,
        *,
        task_factory: Callable[[tuple[BoundedEvidence, ...]], str],
        memory_hints: Sequence[MemoryHint],
        evidence: Sequence[ContextEvidence],
    ) -> BoundedContext:
        """Select evidence before rendering retained-ID-dependent final task data."""

        retained_hints = list(memory_hints)
        dropped_hints: list[str] = []
        while (
            _memory_area_tokens(retained_hints, self._count)
            > self._limits.max_memory_tokens
        ):
            if not retained_hints:
                raise ContextBudgetError("memory container exceeds its token allocation")
            dropped = _pop_lowest_score(retained_hints)
            dropped_hints.append(dropped.text)

        retained_evidence = [
            BoundedEvidence(
                **item.model_dump(),
                original_body=item.body,
            )
            for item in evidence
        ]
        dropped_evidence: list[str] = []
        if (
            _evidence_area_tokens(retained_evidence, self._count)
            > self._limits.max_evidence_tokens
            and self._compressor is not None
        ):
            retained_evidence, invalid_ids = self._compress_evidence(retained_evidence)
            dropped_evidence.extend(invalid_ids)
        while (
            _evidence_area_tokens(retained_evidence, self._count)
            > self._limits.max_evidence_tokens
            and retained_evidence
        ):
            dropped = _pop_lowest_score(retained_evidence)
            dropped_evidence.append(dropped.evidence_id)

        while True:
            task = task_factory(tuple(retained_evidence))
            candidate = _bounded_context(
                control=self._control,
                task=task,
                memory=retained_hints,
                evidence=retained_evidence,
                dropped_evidence=dropped_evidence,
                dropped_hints=dropped_hints,
                count=self._count,
            )
            if candidate.task_tokens > self._limits.max_task_tokens:
                if retained_evidence:
                    dropped = _pop_lowest_score(retained_evidence)
                    dropped_evidence.append(dropped.evidence_id)
                    continue
                raise ContextBudgetError(
                    f"task requires {candidate.task_tokens} tokens; "
                    f"limit is {self._limits.max_task_tokens}; task is never truncated"
                )
            if candidate.total_tokens <= self._limits.max_total_tokens:
                self._assert_final_limits(candidate)
                return candidate
            if retained_evidence:
                dropped = _pop_lowest_score(retained_evidence)
                dropped_evidence.append(dropped.evidence_id)
            elif retained_hints:
                dropped = _pop_lowest_score(retained_hints)
                dropped_hints.append(dropped.text)
            else:
                raise ContextBudgetError(
                    "control and task exceed the total context allocation and are never truncated"
                )

    def _compress_evidence(
        self,
        evidence: list[BoundedEvidence],
    ) -> tuple[list[BoundedEvidence], list[str]]:
        metadata_tokens = sum(
            self._count(_render_evidence_block(item.model_copy(update={"body": ""})))
            for item in evidence
        )
        available_body_tokens = max(
            0,
            self._limits.max_evidence_tokens - metadata_tokens,
        )
        body_tokens = [max(1, self._count(item.body)) for item in evidence]
        total_body_tokens = sum(body_tokens)
        retained: list[BoundedEvidence] = []
        dropped: list[str] = []
        for item, item_tokens in zip(evidence, body_tokens, strict=True):
            target = max(
                1,
                round(available_body_tokens * item_tokens / total_body_tokens),
            )
            if item_tokens <= target:
                retained.append(item)
                continue
            assert self._compressor is not None
            compressed = self._compressor.compress(item.body, max_tokens=target).strip()
            if not _compression_preserves_support(item.body, compressed):
                dropped.append(item.evidence_id)
                continue
            retained.append(item.model_copy(update={"body": compressed}))
        return retained, dropped

    def _assert_final_limits(self, context: BoundedContext) -> None:
        actual = _bounded_context(
            control=context.control,
            task=context.task,
            memory=context.memory_hints,
            evidence=context.evidence,
            dropped_evidence=list(context.dropped_evidence_ids),
            dropped_hints=list(context.dropped_memory_hints),
            count=self._count,
        )
        if actual != context:
            raise AssertionError("final rendered context accounting changed after allocation")
        for area, used, limit in (
            ("control", context.control_tokens, self._limits.max_control_tokens),
            ("task", context.task_tokens, self._limits.max_task_tokens),
            ("memory", context.memory_tokens, self._limits.max_memory_tokens),
            ("evidence", context.evidence_tokens, self._limits.max_evidence_tokens),
            ("total", context.total_tokens, self._limits.max_total_tokens),
        ):
            if used > limit:
                raise AssertionError(f"final {area} payload exceeds its token allocation")


def _render_evidence_block(item: BoundedEvidence) -> str:
    attributes: list[tuple[str, object | None]] = [
        ("evidence_id", item.evidence_id),
        ("source_type", item.source_type),
        ("ticker", item.ticker),
        ("source_url", item.source_url),
        ("date", item.date),
        ("accession_no", item.accession_no),
        ("section", item.section),
        ("raw_start", item.raw_start),
        ("raw_end", item.raw_end),
        ("form", item.form),
        ("title", item.title),
        ("source_kind", item.source_kind),
        ("source_tier", item.source_tier),
        ("fetched_at", item.fetched_at),
        ("citation_bindings", "|".join(item.citation_bindings)),
        ("untrusted", "true"),
    ]
    rendered_attributes = " ".join(
        f'{name}="{escape(str(value), quote=True)}"'
        for name, value in attributes
        if value is not None
    )
    return f"<evidence {rendered_attributes}>\n{escape(item.body)}\n</evidence>"


def _render_task(task: str) -> str:
    return f"<task>\n{task}\n</task>\n"


def _render_memory(values: Sequence[MemoryHint]) -> str:
    memory = "\n".join(_render_memory_hint(hint) for hint in values)
    return f'<memory_hints untrusted="true">\n{memory}\n</memory_hints>\n'


def _render_memory_hint(hint: MemoryHint) -> str:
    attributes = ['untrusted="true"']
    if hint.identity is not None:
        attributes.append(f'identity="{hint.identity}"')
    if hint.pointer_id is not None:
        attributes.append(f'pointer_id="{escape(hint.pointer_id, quote=True)}"')
    return (
        f"<memory_hint {' '.join(attributes)}>"
        f"{escape(hint.text)}</memory_hint>"
    )


def _render_evidence_area(values: Sequence[BoundedEvidence]) -> str:
    evidence = "\n".join(_render_evidence_block(value) for value in values)
    return (
        '<evidence_blocks untrusted="true" instruction_authority="none">\n'
        f"{evidence}\n"
        "</evidence_blocks>"
    )


def _memory_area_tokens(values: Sequence[MemoryHint], count: TokenCounter) -> int:
    return count(_render_memory(values))


def _evidence_area_tokens(values: Sequence[BoundedEvidence], count: TokenCounter) -> int:
    return count(_render_evidence_area(values))


def _bounded_context(
    *,
    control: str,
    task: str,
    memory: Sequence[MemoryHint],
    evidence: Sequence[BoundedEvidence],
    dropped_evidence: list[str],
    dropped_hints: list[str],
    count: TokenCounter,
) -> BoundedContext:
    task_area = _render_task(task)
    memory_area = _render_memory(memory)
    evidence_area = _render_evidence_area(evidence)
    human = task_area + memory_area + evidence_area
    return BoundedContext(
        control=control,
        task=task,
        memory_hints=tuple(memory),
        evidence=tuple(evidence),
        dropped_evidence_ids=tuple(dict.fromkeys(dropped_evidence)),
        dropped_memory_hints=tuple(dict.fromkeys(dropped_hints)),
        control_tokens=count(control),
        task_tokens=count(task_area),
        memory_tokens=count(memory_area),
        evidence_tokens=count(evidence_area),
        total_tokens=count(control) + count(human),
    )


def _pop_lowest_score(values: list[Any]) -> Any:
    index = min(range(len(values)), key=lambda position: (values[position].score, -position))
    return values.pop(index)


_READABLE_TERM = re.compile(r"[A-Za-z]{3,}|[\u3400-\u9fff]")
_EXTRACTIVE_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _compression_preserves_support(original: str, compressed: str) -> bool:
    if not compressed.strip() or _READABLE_TERM.search(compressed) is None:
        return False
    original_tokens = _EXTRACTIVE_TOKEN.findall(original)
    compressed_tokens = _EXTRACTIVE_TOKEN.findall(compressed)
    cursor = 0
    for token in compressed_tokens:
        try:
            cursor = original_tokens.index(token, cursor) + 1
        except ValueError:
            return False
    original_numeric = [token for token in original_tokens if any(char.isdigit() for char in token)]
    compressed_numeric = [
        token for token in compressed_tokens if any(char.isdigit() for char in token)
    ]
    if compressed_numeric != original_numeric:
        return False
    return True


def _utf8_byte_upper_bound(value: str) -> int:
    """Return a provider-independent upper bound: no token encodes less than one byte."""
    return len(value.encode("utf-8"))


class BudgetLimits(BaseModel):
    """Call-count authority for one research run or frozen recipe execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_planner_calls: int = Field(default=1, ge=0)
    max_analysis_calls: int = Field(default=1, ge=0)
    max_repair_calls: int = Field(default=1, ge=0)
    max_tool_calls: int = Field(ge=0)
    max_retrieval_rounds: int = Field(ge=0)
    max_web_calls: int = Field(ge=0)

    @classmethod
    def from_policy(cls, policy: object) -> BudgetLimits:
        budget = getattr(policy, "budget")
        retrieval_rounds = int(getattr(budget, "max_retrieval_rounds"))
        web_calls = int(getattr(budget, "max_web_calls"))
        max_questions = int(getattr(budget, "max_questions"))
        facet_count = len(getattr(policy, "required_facets"))
        return cls(
            max_tool_calls=(
                max_questions * 2 * facet_count * retrieval_rounds + web_calls
            ),
            max_retrieval_rounds=retrieval_rounds,
            max_web_calls=web_calls,
        )

    @classmethod
    def aggregate(cls, policies: Sequence[object]) -> BudgetLimits:
        limits = [cls.from_policy(policy) for policy in policies]
        return cls(
            max_planner_calls=sum(limit.max_planner_calls for limit in limits),
            max_analysis_calls=sum(limit.max_analysis_calls for limit in limits),
            max_repair_calls=1 if limits else 0,
            max_tool_calls=(1 if limits else 0)
            + sum(limit.max_tool_calls for limit in limits),
            max_retrieval_rounds=sum(limit.max_retrieval_rounds for limit in limits),
            max_web_calls=sum(limit.max_web_calls for limit in limits),
        )


class BudgetExhaustedError(RuntimeError):
    """Raised before a run operation would exceed one immutable hard limit."""

    def __init__(self, dimension: str, attempted: int, limit: int) -> None:
        self.dimension = dimension
        self.attempted = attempted
        self.limit = limit
        super().__init__(f"budget exhausted for {dimension}: attempted {attempted}, limit {limit}")


class BudgetState(BaseModel):
    """Mutable run-local counters validated atomically against frozen limits."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    _lock: RLock = PrivateAttr(default_factory=RLock)

    limits: BudgetLimits
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    planner_calls: int = Field(default=0, ge=0)
    analysis_calls: int = Field(default=0, ge=0)
    repair_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    retrieval_rounds: int = Field(default=0, ge=0)
    web_calls: int = Field(default=0, ge=0)

    def consume(
        self,
        *,
        planner_calls: int = 0,
        analysis_calls: int = 0,
        repair_calls: int = 0,
        tool_calls: int = 0,
        retrieval_rounds: int = 0,
        web_calls: int = 0,
    ) -> None:
        with self._lock:
            resulting = self._check_unlocked(
                planner_calls=planner_calls,
                analysis_calls=analysis_calls,
                repair_calls=repair_calls,
                tool_calls=tool_calls,
                retrieval_rounds=retrieval_rounds,
                web_calls=web_calls,
            )
            self._apply_unlocked(resulting)

    def check(
        self,
        *,
        planner_calls: int = 0,
        analysis_calls: int = 0,
        repair_calls: int = 0,
        tool_calls: int = 0,
        retrieval_rounds: int = 0,
        web_calls: int = 0,
    ) -> dict[str, int]:
        """Return validated resulting counts without mutating this state."""
        with self._lock:
            return self._check_unlocked(
                planner_calls=planner_calls,
                analysis_calls=analysis_calls,
                repair_calls=repair_calls,
                tool_calls=tool_calls,
                retrieval_rounds=retrieval_rounds,
                web_calls=web_calls,
            )

    def _check_unlocked(
        self,
        *,
        planner_calls: int = 0,
        analysis_calls: int = 0,
        repair_calls: int = 0,
        tool_calls: int = 0,
        retrieval_rounds: int = 0,
        web_calls: int = 0,
    ) -> dict[str, int]:
        increments = {
            "planner_calls": planner_calls,
            "analysis_calls": analysis_calls,
            "repair_calls": repair_calls,
            "retrieval_rounds": retrieval_rounds,
            "web_calls": web_calls,
            "tool_calls": tool_calls,
        }
        resulting: dict[str, int] = {}
        for dimension, increment in increments.items():
            if increment < 0:
                raise ValueError("budget increments must be non-negative")
            attempted = int(getattr(self, dimension)) + increment
            limit = int(getattr(self.limits, f"max_{dimension}"))
            if attempted > limit:
                raise BudgetExhaustedError(dimension, attempted, limit)
            resulting[dimension] = attempted
        return resulting

    def _apply_unlocked(self, resulting: Mapping[str, int]) -> None:
        for dimension, attempted in resulting.items():
            setattr(self, dimension, attempted)


class BudgetGate(Protocol):
    """Small pre-side-effect budget interface threaded through collectors and adapters."""

    def consume(
        self,
        *,
        planner_calls: int = 0,
        analysis_calls: int = 0,
        repair_calls: int = 0,
        tool_calls: int = 0,
        retrieval_rounds: int = 0,
        web_calls: int = 0,
    ) -> None: ...


class BudgetAuthority:
    """Application-owned shared authority configured exactly once per selected run."""

    def __init__(self) -> None:
        self._state: BudgetState | None = None
        self._lock = RLock()

    @property
    def configured(self) -> bool:
        with self._lock:
            return self._state is not None

    @property
    def state(self) -> BudgetState:
        with self._lock:
            if self._state is None:
                raise RuntimeError("run budget authority is not configured")
            return self._state

    def configure(self, limits: BudgetLimits) -> None:
        with self._lock:
            if self._state is None:
                self._state = BudgetState(limits=limits)
                return
            if self._state.limits != limits:
                raise RuntimeError("run budget authority cannot be reconfigured")

    def consume(self, **increments: int) -> None:
        with self._lock:
            state = self.state
            with state._lock:
                resulting = state._check_unlocked(**increments)
                state._apply_unlocked(resulting)

    def child(self, limits: BudgetLimits) -> HierarchicalBudgetGate:
        with self._lock:
            return HierarchicalBudgetGate(self, BudgetState(limits=limits))


class HierarchicalBudgetGate:
    """Atomically enforce shared run limits and one recipe-local limit set."""

    def __init__(self, shared: BudgetAuthority, local: BudgetState) -> None:
        self.shared = shared
        self.local = local

    def consume(self, **increments: int) -> None:
        with self.shared._lock, self.shared.state._lock, self.local._lock:
            shared_state = self.shared.state
            shared_result = shared_state._check_unlocked(**increments)
            local_result = self.local._check_unlocked(**increments)
            shared_state._apply_unlocked(shared_result)
            self.local._apply_unlocked(local_result)
