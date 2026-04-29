"""
Context Window Guard

Derived from OpenClaw src/agents/context-window-guard.ts.
Pre-flight check before judge calls: ensures enough tokens remain
in the judge model's context window for a meaningful response.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import (
    CONTEXT_WINDOW_HARD_MIN_TOKENS,
    CONTEXT_WINDOW_WARN_BELOW_TOKENS,
    JUDGE_PROMPT_OVERHEAD_TOKENS,
    SAFETY_MARGIN,
)

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Intentional overestimate: chars/3 × SAFETY_MARGIN.
    GPT/Claude tokenizers average ~3 chars/token for English prose;
    code and special chars can be denser.  The 1.2× safety margin
    ensures we never silently exceed context budgets.
    """
    return int((len(text) / 3) * SAFETY_MARGIN)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total token count for a list of chat messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
        total += 4  # role / delimiters overhead per message
    return total


def resolve_judge_max_tokens(
    model_name: str,
    model_context_length: Optional[int] = None,
) -> int:
    """
    Return the context window size for a given judge model.
    If not provided, use conservative defaults per known model family.
    """
    if model_context_length is not None:
        return model_context_length

    # Conservative defaults for known model families
    name_lower = model_name.lower()
    if "qwen3-30b" in name_lower:
        return 32768   # VLLM_JUDGE_MAX_MODEL_LEN; actual calls peak at ~8K
    if "qwen" in name_lower:
        return 32768
    if "gpt-4" in name_lower:
        return 128000
    if "claude" in name_lower:
        return 200000
    if "mistral" in name_lower:
        return 32768
    if "llama" in name_lower:
        return 8192

    # Safe fallback
    return 16384


def check_judge_context_window(
    messages: list[dict],
    model_name: str,
    max_response_tokens: int = 512,
    model_context_length: Optional[int] = None,
) -> dict:
    """
    Pre-flight context window check before issuing a judge call.
    Derived from OpenClaw context-window-guard.ts.

    Returns:
        {
            "ok": bool,           # True if call can proceed
            "action": str,        # "proceed" | "warn" | "block"
            "available_tokens": int,
            "message": str,
        }
    """
    context_limit = resolve_judge_max_tokens(model_name, model_context_length)
    used_tokens = estimate_messages_tokens(messages)
    reserved = JUDGE_PROMPT_OVERHEAD_TOKENS + max_response_tokens
    available = context_limit - used_tokens - reserved

    if available < CONTEXT_WINDOW_HARD_MIN_TOKENS:
        msg = (
            f"[context-guard] BLOCKED: {model_name} has only ~{available} tokens "
            f"available (need ≥{CONTEXT_WINDOW_HARD_MIN_TOKENS}). "
            f"Context: {context_limit}, used: ~{used_tokens}, reserved: {reserved}."
        )
        logger.error(msg)
        return {
            "ok": False,
            "action": "block",
            "available_tokens": available,
            "message": msg,
        }

    if available < CONTEXT_WINDOW_WARN_BELOW_TOKENS:
        msg = (
            f"[context-guard] WARNING: {model_name} has ~{available} tokens "
            f"available (warn threshold={CONTEXT_WINDOW_WARN_BELOW_TOKENS}). "
            f"Consider compaction."
        )
        logger.warning(msg)
        return {
            "ok": True,
            "action": "warn",
            "available_tokens": available,
            "message": msg,
        }

    return {
        "ok": True,
        "action": "proceed",
        "available_tokens": available,
        "message": f"Context OK: ~{available} tokens available.",
    }
