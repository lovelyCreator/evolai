"""
Session Cost & Latency Tracker

Derived from OpenClaw src/infra/session-cost-usage.ts.
Tracks per-model pricing, latency percentiles, and call volumes
for each evaluation run. Enables cost visibility and budget management.

Usage:
    tracker = CostTracker()
    tracker.record("Qwen/Qwen3-30B", prompt_tokens=5000,
                   completion_tokens=200, latency_ms=850.0, source="judge")
    print(tracker.summary())
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Single cost record from one LLM call."""
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    source: str  # "judge" | "miner"
    ts: float = field(default_factory=time.time)


class CostTracker:
    """
    Per-evaluation cost and latency tracking.
    Derived from OpenClaw src/infra/session-cost-usage.ts.

    Tracks:
      - Cost per model (GPU amortization pricing for local, API pricing for cloud)
      - Latency: average, P50, P95, per-model breakdown
      - Token volume: prompt + completion by source
      - Call count: total and by model/source
    """

    # Approximate $/1M tokens for known models
    # Local vLLM pricing = GPU amortization (H100 ~$2/hr → ~$0.15/1M input tokens)
    MODEL_COST_PER_1M: dict[str, dict[str, float]] = {
        "Qwen/Qwen3-30B-A3B-Instruct-2507": {"input": 0.15, "output": 0.60},
        # Cloud fallback pricing (if ever added)
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    }

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        source: str = "judge",
    ) -> None:
        """
        Record a single LLM call's cost and latency.

        Args:
            model: Model name (must match MODEL_COST_PER_1M keys for pricing).
            prompt_tokens: Input tokens consumed.
            completion_tokens: Output tokens generated.
            latency_ms: Wall-clock latency in milliseconds.
            source: "judge" or "miner".
        """
        cost_map = self.MODEL_COST_PER_1M.get(model, {"input": 0.0, "output": 0.0})
        cost = (
            prompt_tokens * cost_map["input"]
            + completion_tokens * cost_map["output"]
        ) / 1_000_000

        entry = CostEntry(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            source=source,
        )
        self._entries.append(entry)

    def summary(self) -> dict[str, Any]:
        """
        Return aggregate cost and latency summary.

        Returns:
            {
                "total_cost_usd": float,
                "total_calls": int,
                "total_prompt_tokens": int,
                "total_completion_tokens": int,
                "avg_latency_ms": float,
                "p50_latency_ms": float,
                "p95_latency_ms": float,
                "by_model": { model: { calls, cost, avg_latency } },
                "by_source": { source: { calls, cost, tokens } },
            }
        """
        if not self._entries:
            return {
                "total_cost_usd": 0.0,
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "by_model": {},
                "by_source": {},
            }

        total_cost = sum(e.cost_usd for e in self._entries)
        total_prompt = sum(e.prompt_tokens for e in self._entries)
        total_completion = sum(e.completion_tokens for e in self._entries)
        latencies = sorted(e.latency_ms for e in self._entries)
        n = len(latencies)

        by_model = self._group_by("model")
        by_source = self._group_by("source")

        return {
            "total_cost_usd": round(total_cost, 6),
            "total_calls": n,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "avg_latency_ms": round(sum(latencies) / n, 1),
            "p50_latency_ms": round(latencies[n // 2], 1),
            "p95_latency_ms": round(latencies[int(n * 0.95)], 1),
            "by_model": by_model,
            "by_source": by_source,
        }

    def _group_by(self, field: str) -> dict[str, dict[str, Any]]:
        """Group entries by a field and aggregate."""
        groups: dict[str, list[CostEntry]] = {}
        for e in self._entries:
            key = getattr(e, field)
            groups.setdefault(key, []).append(e)

        result: dict[str, dict[str, Any]] = {}
        for key, entries in groups.items():
            lats = [e.latency_ms for e in entries]
            result[key] = {
                "calls": len(entries),
                "cost_usd": round(sum(e.cost_usd for e in entries), 6),
                "prompt_tokens": sum(e.prompt_tokens for e in entries),
                "completion_tokens": sum(e.completion_tokens for e in entries),
                "avg_latency_ms": round(sum(lats) / len(lats), 1),
            }
        return result

    @property
    def total_cost_usd(self) -> float:
        """Quick accessor for total cost."""
        return sum(e.cost_usd for e in self._entries)

    @property
    def call_count(self) -> int:
        """Total number of recorded calls."""
        return len(self._entries)

    def reset(self) -> None:
        """Clear all entries."""
        self._entries.clear()

    def __repr__(self) -> str:
        return (
            f"CostTracker(calls={self.call_count}, "
            f"cost=${self.total_cost_usd:.4f})"
        )
