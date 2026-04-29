"""
Fixed-Window Rate Limiter

Derived from OpenClaw src/infra/fixed-window-rate-limit.ts.
Prevents judge/miner call spikes from overloading local vLLM servers
during batch evaluation of multiple miners.

Usage:
    limiter = FixedWindowRateLimiter(max_requests=30, window_ms=60_000)
    result = limiter.consume()
    if not result["allowed"]:
        time.sleep(result["retry_after_ms"] / 1000)
"""

from __future__ import annotations

import time
import logging

from .config import (
    JUDGE_RATE_LIMIT_MAX_REQUESTS,
    JUDGE_RATE_LIMIT_WINDOW_MS,
    MINER_RATE_LIMIT_MAX_REQUESTS,
    MINER_RATE_LIMIT_WINDOW_MS,
)

logger = logging.getLogger(__name__)


class FixedWindowRateLimiter:
    """
    Fixed-window rate limiter.
    Derived from OpenClaw src/infra/fixed-window-rate-limit.ts:
        createFixedWindowRateLimiter({ maxRequests, windowMs })
        → { consume(), reset() }

    consume() returns { allowed, retry_after_ms, remaining }.
    """

    def __init__(self, max_requests: int, window_ms: int) -> None:
        self.max_requests = max_requests
        self.window_ms = window_ms
        self._count: int = 0
        self._window_start_ms: int = 0

    def consume(self) -> dict:
        """
        Attempt to consume a request slot.

        Returns:
            {
                "allowed": bool,
                "retry_after_ms": int,  # 0 if allowed, else ms to wait
                "remaining": int,       # remaining slots in this window
            }
        """
        now_ms = int(time.time() * 1000)

        # Reset window if expired
        if now_ms - self._window_start_ms >= self.window_ms:
            self._window_start_ms = now_ms
            self._count = 0

        if self._count >= self.max_requests:
            retry_after = self.window_ms - (now_ms - self._window_start_ms)
            logger.debug(
                f"[rate-limiter] Request denied — "
                f"{self._count}/{self.max_requests} used, "
                f"retry in {max(0, retry_after)}ms"
            )
            return {
                "allowed": False,
                "retry_after_ms": max(0, retry_after),
                "remaining": 0,
            }

        self._count += 1
        remaining = self.max_requests - self._count
        return {
            "allowed": True,
            "retry_after_ms": 0,
            "remaining": remaining,
        }

    def reset(self) -> None:
        """Reset the rate limiter (e.g., after server restart)."""
        self._count = 0
        self._window_start_ms = 0

    def __repr__(self) -> str:
        return (
            f"FixedWindowRateLimiter("
            f"max={self.max_requests}, window={self.window_ms}ms, "
            f"used={self._count})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton instances (from OpenClaw pattern)
# ──────────────────────────────────────────────────────────────────────────────

_judge_limiter: FixedWindowRateLimiter | None = None
_miner_limiter: FixedWindowRateLimiter | None = None


def get_judge_rate_limiter() -> FixedWindowRateLimiter:
    """Return the module-level judge rate limiter singleton."""
    global _judge_limiter
    if _judge_limiter is None:
        _judge_limiter = FixedWindowRateLimiter(
            max_requests=JUDGE_RATE_LIMIT_MAX_REQUESTS,
            window_ms=JUDGE_RATE_LIMIT_WINDOW_MS,
        )
    return _judge_limiter


def get_miner_rate_limiter() -> FixedWindowRateLimiter:
    """Return the module-level miner rate limiter singleton."""
    global _miner_limiter
    if _miner_limiter is None:
        _miner_limiter = FixedWindowRateLimiter(
            max_requests=MINER_RATE_LIMIT_MAX_REQUESTS,
            window_ms=MINER_RATE_LIMIT_WINDOW_MS,
        )
    return _miner_limiter


def wait_for_judge_slot() -> None:
    """
    Blocking helper: waits until a judge call slot is available.
    Used inside call_judge_with_fallback() before issuing each call.
    """
    limiter = get_judge_rate_limiter()
    while True:
        result = limiter.consume()
        if result["allowed"]:
            return
        wait_s = result["retry_after_ms"] / 1000
        logger.info(f"[rate-limiter] Judge rate limit hit, waiting {wait_s:.1f}s")
        time.sleep(wait_s)


def wait_for_miner_slot() -> None:
    """
    Blocking helper: waits until a miner call slot is available.
    Used inside _get_miner_response() before streaming.
    """
    limiter = get_miner_rate_limiter()
    while True:
        result = limiter.consume()
        if result["allowed"]:
            return
        wait_s = result["retry_after_ms"] / 1000
        logger.info(f"[rate-limiter] Miner rate limit hit, waiting {wait_s:.1f}s")
        time.sleep(wait_s)
