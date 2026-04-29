"""
Token Usage Accumulator

Derived from OpenClaw run.ts UsageAccumulator / mergeUsage pattern
and usage.ts:normalizeUsage() for provider-agnostic field normalization.
Tracks cumulative token usage (prompt + completion + cache reads)
across all judge and miner calls in a single evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UsageSnapshot:
    """Single usage record from one LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    source: str = ""          # "judge" | "miner"
    model: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Universal Usage Normalization (from OpenClaw usage.ts)
# ──────────────────────────────────────────────────────────────────────────────

def _nested_cache(d: dict) -> int:
    """
    Extract cache from OpenAI nested prompt_tokens_details.
    Handles both dict and object forms.
    """
    details = d.get("prompt_tokens_details") or d.get("promptTokensDetails") or {}
    if isinstance(details, dict):
        return details.get("cached_tokens", 0) or details.get("cachedTokens", 0)
    return getattr(details, "cached_tokens", 0) or getattr(details, "cachedTokens", 0)


def normalize_usage(
    raw: dict | object | None,
    source: str = "",
    model: str = "",
) -> UsageSnapshot | None:
    """
    Universal usage normalizer — handles all known provider field formats.
    Derived from OpenClaw src/agents/usage.ts:normalizeUsage().

    Supports field names from:
      - OpenAI:    prompt_tokens, completion_tokens, total_tokens
      - Anthropic: input_tokens, output_tokens
      - Google:    promptTokens, completionTokens
      - Ollama:    prompt_eval_count, eval_count
      - camelCase: inputTokens, outputTokens
      - Short:     input, output

    Returns:
        UsageSnapshot or None if input is None.
    """
    if raw is None:
        return None

    if isinstance(raw, dict):
        d = raw
    elif hasattr(raw, "__dict__"):
        d = vars(raw)
    else:
        # Try attribute access for frozen objects (e.g., openai.types.CompletionUsage)
        d = {}
        for attr in ("prompt_tokens", "completion_tokens", "total_tokens",
                      "input_tokens", "output_tokens", "cache_read_input_tokens"):
            val = getattr(raw, attr, None)
            if val is not None:
                d[attr] = val

    prompt = (
        d.get("prompt_tokens")
        or d.get("promptTokens")
        or d.get("input_tokens")
        or d.get("inputTokens")
        or d.get("input")
        or d.get("prompt_eval_count")  # Ollama
        or 0
    )
    completion = (
        d.get("completion_tokens")
        or d.get("completionTokens")
        or d.get("output_tokens")
        or d.get("outputTokens")
        or d.get("output")
        or d.get("eval_count")  # Ollama
        or 0
    )
    total = d.get("total_tokens") or d.get("totalTokens") or (prompt + completion)
    cache_read = (
        d.get("cache_read_input_tokens")
        or d.get("cacheReadInputTokens")
        or _nested_cache(d)
        or 0
    )
    cache_create = (
        d.get("cache_creation_input_tokens")
        or d.get("cacheCreationInputTokens")
        or 0
    )

    return UsageSnapshot(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_create,
        source=source,
        model=model,
    )


def has_nonzero_usage(snap: UsageSnapshot | None) -> bool:
    """Check if a usage snapshot has any non-zero token counts."""
    if snap is None:
        return False
    return (snap.prompt_tokens + snap.completion_tokens + snap.total_tokens) > 0


class UsageAccumulator:
    """
    Accumulates token usage across all LLM calls in one evaluation.

    Derived from OpenClaw's UsageAccumulator pattern:
      - merge()       : add usage from an OpenAI-compat response
      - merge_ollama(): add usage from Ollama's eval_count / prompt_eval_count
      - summary()     : return aggregate stats

    Usage:
        acc = UsageAccumulator()
        # After each judge call:
        acc.merge(response.usage, source="judge", model="Qwen3-30B")
        # After each miner stream:
        acc.merge_ollama(stream_acc, source="miner", model="evolai-1.5b")
        # At end:
        print(acc.summary())
    """

    def __init__(self) -> None:
        self._records: list[UsageSnapshot] = []
        self._last_cache_read: int = 0  # Track cache hit pattern

    def merge(
        self,
        usage: object | dict | None,
        source: str = "",
        model: str = "",
    ) -> None:
        """
        Merge usage from any LLM provider response.

        Uses normalize_usage() (from OpenClaw usage.ts) for provider-agnostic
        field extraction — handles OpenAI, Anthropic, Google, Ollama, and
        camelCase variants automatically.
        """
        snap = normalize_usage(usage, source=source, model=model)
        if snap is None:
            return
        self._last_cache_read = snap.cache_read_tokens
        self._records.append(snap)

    def merge_ollama(
        self,
        stream_acc: object,
        source: str = "miner",
        model: str = "",
    ) -> None:
        """
        Merge usage from an Ollama StreamAccumulator.

        Ollama reports prompt_eval_count (input tokens) and
        eval_count (output tokens). Now delegates to normalize_usage()
        which handles these field names natively.
        """
        snap = normalize_usage(
            {
                "prompt_eval_count": getattr(stream_acc, "prompt_eval_count", 0),
                "eval_count": getattr(stream_acc, "eval_count", 0),
            },
            source=source,
            model=model,
        )
        if snap is not None:
            self._records.append(snap)

    @property
    def last_cache_read(self) -> int:
        """Tokens served from KV cache in the most recent call."""
        return self._last_cache_read

    def summary(self) -> dict:
        """
        Return aggregate usage summary.

        Returns:
            {
                "total_prompt_tokens": int,
                "total_completion_tokens": int,
                "total_tokens": int,
                "total_cache_read_tokens": int,
                "total_cache_creation_tokens": int,
                "last_cache_read": int,       # OVERWRITE, not summed (per design)
                "call_count": int,
                "by_source": {"judge": {...}, "miner": {...}},
            }

        Note: last_cache_read uses OVERWRITE semantics (not sum) per design doc
        §Token Management. Cache read reflects how much of the CURRENT context
        was served from KV cache — summing across calls would inflate this metric.
        """
        total_prompt = sum(r.prompt_tokens for r in self._records)
        total_completion = sum(r.completion_tokens for r in self._records)
        total_all = sum(r.total_tokens for r in self._records)
        total_cache_read = sum(r.cache_read_tokens for r in self._records)
        total_cache_create = sum(r.cache_creation_tokens for r in self._records)

        by_source: dict[str, dict] = {}
        for r in self._records:
            key = r.source or "unknown"
            if key not in by_source:
                by_source[key] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls": 0,
                }
            by_source[key]["prompt_tokens"] += r.prompt_tokens
            by_source[key]["completion_tokens"] += r.completion_tokens
            by_source[key]["total_tokens"] += r.total_tokens
            by_source[key]["calls"] += 1

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_all,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_creation_tokens": total_cache_create,
            "last_cache_read": self._last_cache_read,  # OVERWRITE per design
            "call_count": len(self._records),
            "by_source": by_source,
        }

    def reset(self) -> None:
        """Clear all records."""
        self._records.clear()
        self._last_cache_read = 0
