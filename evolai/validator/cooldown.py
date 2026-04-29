"""
Cooldown-Aware Judge Rotation

Derived from OpenClaw src/agents/model-fallback.ts and
auth-profiles/usage.ts:markAuthProfileFailure/isProfileInCooldown.

Tracks per-model cooldown state and rotates through the judge pool
when the primary judge is in cooldown.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import torch

from .config import (
    JUDGE_COOLDOWN_OOM_MS,
    JUDGE_COOLDOWN_ERROR_MS,
    PROBE_INTERVAL_MS,
    PROBE_MARGIN_MS,
    JUDGE_MODELS,
)

logger = logging.getLogger(__name__)


class JudgeCooldownStore:
    """
    Tracks per-model cooldown state across evaluation rounds.
    Derived from OpenClaw auth-profiles/usage.ts:
      markAuthProfileFailure / isProfileInCooldown.
    """

    def __init__(self) -> None:
        # model_name → expiry epoch in milliseconds
        self._cooldowns: dict[str, float] = {}
        # model_name → last probe epoch in milliseconds
        self._last_probe: dict[str, float] = {}

    def mark_failure(self, model_name: str, error: Exception) -> None:
        """
        Mark a judge model as failed and start its cooldown period.
        OOM → 5 min cooldown; other errors → 60 s cooldown.
        """
        now_ms = time.monotonic() * 1000
        if isinstance(error, torch.cuda.OutOfMemoryError):
            cooldown_ms = JUDGE_COOLDOWN_OOM_MS
            logger.warning(
                f"[cooldown] {model_name} OOM → cooldown {cooldown_ms / 1000:.0f}s"
            )
        else:
            cooldown_ms = JUDGE_COOLDOWN_ERROR_MS
            logger.warning(
                f"[cooldown] {model_name} error → cooldown {cooldown_ms / 1000:.0f}s: "
                f"{str(error)[:120]}"
            )
        self._cooldowns[model_name] = now_ms + cooldown_ms

    def is_in_cooldown(self, model_name: str) -> bool:
        """True if model is currently in cooldown."""
        expiry = self._cooldowns.get(model_name)
        if expiry is None:
            return False
        return time.monotonic() * 1000 < expiry

    def cooldown_remaining_ms(self, model_name: str) -> float:
        """Milliseconds remaining in cooldown, or 0 if not in cooldown."""
        expiry = self._cooldowns.get(model_name)
        if expiry is None:
            return 0
        remaining = expiry - time.monotonic() * 1000
        return max(0, remaining)

    def should_probe_primary(self, model_name: str) -> bool:
        """
        Derived from OpenClaw shouldProbePrimaryDuringCooldown().
        Returns True if:
          - model IS in cooldown
          - cooldown expires within PROBE_MARGIN_MS
          - at least PROBE_INTERVAL_MS since last probe
        """
        if not self.is_in_cooldown(model_name):
            return False  # Not in cooldown → just use it normally
        remaining = self.cooldown_remaining_ms(model_name)
        if remaining > PROBE_MARGIN_MS:
            return False  # Still deep in cooldown
        now_ms = time.monotonic() * 1000
        last_probe = self._last_probe.get(model_name, 0)
        if now_ms - last_probe < PROBE_INTERVAL_MS:
            return False  # Probed too recently
        return True

    def record_probe(self, model_name: str) -> None:
        """Record that we just probed this model."""
        self._last_probe[model_name] = time.monotonic() * 1000

    def clear_cooldown(self, model_name: str) -> None:
        """Remove cooldown on successful probe / recovery."""
        self._cooldowns.pop(model_name, None)
        logger.info(f"[cooldown] {model_name} recovered — cooldown cleared")


# Module-level singleton (shared across one validator process)
_cooldown_store = JudgeCooldownStore()


def get_cooldown_store() -> JudgeCooldownStore:
    """Return the process-global cooldown store."""
    return _cooldown_store


class FallbackAttempt:
    """
    Tracks which judges have been tried in the current call.
    Ensures we don't retry the same judge in one invocation.
    """

    def __init__(self, pool: list[str]) -> None:
        self._pool = list(pool)
        self._tried: set[str] = set()
        self._primary = pool[0] if pool else None

    def next_available(self) -> Optional[str]:
        """
        Return the next judge to try, respecting cooldowns.
        Priority: primary (if not in cooldown) → fallbacks in order.
        If primary should be probed, try it even during cooldown.
        """
        store = get_cooldown_store()

        # Check if primary should be probed (near end of cooldown)
        if (
            self._primary
            and self._primary not in self._tried
            and store.should_probe_primary(self._primary)
        ):
            store.record_probe(self._primary)
            self._tried.add(self._primary)
            logger.info(f"[fallback] probing primary {self._primary} (cooldown expiring)")
            return self._primary

        # Try each model in pool order
        for model in self._pool:
            if model in self._tried:
                continue
            if store.is_in_cooldown(model):
                logger.debug(
                    f"[fallback] skipping {model} "
                    f"(cooldown {store.cooldown_remaining_ms(model):.0f}ms remaining)"
                )
                continue
            self._tried.add(model)
            return model

        # All models tried or in cooldown — try any untried model regardless
        for model in self._pool:
            if model not in self._tried:
                self._tried.add(model)
                logger.warning(
                    f"[fallback] all preferred judges in cooldown, "
                    f"forcing {model}"
                )
                return model

        return None  # All exhausted

    @property
    def tried_count(self) -> int:
        return len(self._tried)
