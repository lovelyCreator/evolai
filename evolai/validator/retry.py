"""
Retry with Exponential Backoff + Jitter

Derived from OpenClaw src/infra/retry.ts:retryAsync().
All judge calls pass through retry_judge_call() — no raw calls allowed.
"""

from __future__ import annotations

import time
import random
import logging
from typing import Callable, TypeVar, Any

import torch

from .config import (
    JUDGE_RETRY_ATTEMPTS,
    JUDGE_MIN_DELAY_MS,
    JUDGE_MAX_DELAY_MS,
    JUDGE_JITTER,
)
from .error_handling import (
    RateLimitError,
    ContextOverflowError,
    is_likely_context_overflow,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errors that must NEVER be retried (mirroring OpenClaw shouldRethrowAbort)
NON_RETRYABLE: set[type] = {
    torch.cuda.OutOfMemoryError,
    KeyboardInterrupt,
    SystemExit,
    ContextOverflowError,
}


def apply_jitter(delay_ms: float, jitter: float) -> float:
    """
    Apply ±jitter% random offset to a delay.
    Derived from OpenClaw retry.ts jitter calculation.
    
    Args:
        delay_ms: Base delay in milliseconds
        jitter: Jitter fraction (0.2 = ±20%)
    
    Returns:
        Adjusted delay in milliseconds (≥ 0)
    """
    offset = (random.random() * 2 - 1) * jitter
    return max(0, round(delay_ms * (1 + offset)))


def parse_retry_after_header(error: Exception) -> int | None:
    """
    Extract retry-after value from rate-limit errors.
    Returns milliseconds, or None if not available.
    """
    if isinstance(error, RateLimitError) and error.retry_after_ms is not None:
        return error.retry_after_ms
    # Try to read from httpx/requests response headers
    resp = getattr(error, "response", None)
    if resp is not None:
        retry_after = None
        if hasattr(resp, "headers"):
            retry_after = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return int(float(retry_after) * 1000)
            except (ValueError, TypeError):
                pass
    return None


def retry_judge_call(
    fn: Callable[..., T],
    *,
    attempts: int = JUDGE_RETRY_ATTEMPTS,
    min_delay_ms: int = JUDGE_MIN_DELAY_MS,
    max_delay_ms: int = JUDGE_MAX_DELAY_MS,
    jitter: float = JUDGE_JITTER,
    label: str = "judge",
) -> T:
    """
    Exponential-backoff retry with jitter for judge calls.
    
    Derived from OpenClaw src/infra/retry.ts:retryAsync():
      - base_delay = min_delay_ms × 2^(attempt−1)
      - jitter_offset = base_delay × jitter × random(−1, 1)
      - actual_delay = clamp(base_delay + jitter_offset, 0, max_delay_ms)
      - Never retries NON_RETRYABLE errors.
      - Respects retry-after header (analogous to retryAfterMs callback).
    
    Args:
        fn: Zero-argument callable that performs the judge call.
        attempts: Maximum number of attempts.
        min_delay_ms: Initial delay in milliseconds.
        max_delay_ms: Maximum delay cap in milliseconds.
        jitter: Jitter fraction (±).
        label: Label for log messages.
    
    Returns:
        Whatever fn() returns on success.
    
    Raises:
        The last caught exception after all attempts exhausted.
    """
    last_err: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except tuple(NON_RETRYABLE) as e:
            # Never retry OOM, abort, or context overflow
            raise
        except RateLimitError as e:
            retry_after = parse_retry_after_header(e) or min_delay_ms
            delay = apply_jitter(max(retry_after, min_delay_ms), jitter)
            last_err = e
        except Exception as e:
            # Check if this is actually a context overflow in disguise
            if is_likely_context_overflow(e):
                raise ContextOverflowError(str(e), original_error=e)
            base = min_delay_ms * (2 ** (attempt - 1))
            delay = apply_jitter(min(base, max_delay_ms), jitter)
            last_err = e

        if attempt < attempts:
            logger.warning(
                f"[{label}] attempt {attempt}/{attempts} failed, "
                f"retrying in {delay}ms: {last_err}"
            )
            time.sleep(delay / 1000)
        else:
            logger.error(
                f"[{label}] all {attempts} attempts exhausted. "
                f"Last error: {last_err}"
            )

    assert last_err is not None
    raise last_err
