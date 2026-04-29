"""
Staged Context Compaction

Derived from OpenClaw src/agents/compaction.ts.
Replaces hard truncation (conversation_history[-6:]) with staged
summarization that preserves interview quality.

Three stages (fallback chain):
  Stage 1 — Full: Summarize older turns via judge call → keep recent N verbatim
  Stage 2 — Partial: Strip verbose fields from older turns
  Stage 3 — Hard: Keep only SUMMARY_KEEP_RECENT most recent turns
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import (
    SAFETY_MARGIN,
    COMPACTION_TRIGGER_K,
    SUMMARY_KEEP_RECENT,
    COMPACTION_MAX_TOKENS,
    JUDGE_MODELS,
)
from .context_guard import estimate_tokens
from .prompts import build_compaction_messages

logger = logging.getLogger(__name__)


class TokenTracker:
    """
    Tracks cumulative token usage within an interview session.
    Uses SAFETY_MARGIN=1.2 overestimate to prevent silent budget overrun.
    """

    def __init__(
        self,
        response_limit: int = 4096,
        interview_limit: int = 20480,
    ) -> None:
        self.response_limit = response_limit
        self.interview_limit = interview_limit
        self.total_tokens: int = 0

    def estimate(self, text: str) -> int:
        """Overestimate token count with SAFETY_MARGIN."""
        return estimate_tokens(text)

    def add(self, text: str) -> int:
        """Add text to running total, return estimated tokens for this text."""
        n = self.estimate(text)
        self.total_tokens += n
        return n

    def can_continue(self) -> bool:
        """True if we haven't exceeded the interview token budget."""
        return self.total_tokens < self.interview_limit

    def would_exceed_response(self, text: str) -> bool:
        """True if this text alone exceeds per-response limit."""
        return self.estimate(text) > self.response_limit

    def reset(self) -> None:
        self.total_tokens = 0


def should_compact(history: list[dict]) -> bool:
    """True if conversation history exceeds the compaction trigger threshold."""
    return len(history) > COMPACTION_TRIGGER_K


def compact_conversation_history(
    history: list[dict],
    judge_model_pool: Optional[list[str]] = None,
) -> list[dict]:
    """
    Staged context compaction.
    Derived from OpenClaw compaction.ts:compactConversationHistory().

    Stage 1 — Full summarization:
      Summarize turns [0 .. -SUMMARY_KEEP_RECENT] into a single summary block.
    Stage 2 — Partial compaction:
      Strip verbose fields from older turns, truncate long content.
    Stage 3 — Fallback:
      Keep only the most recent SUMMARY_KEEP_RECENT turns.

    The SUMMARY_KEEP_RECENT newest turns are ALWAYS kept verbatim so the
    judge has full context for the next follow-up question.
    """
    if len(history) <= SUMMARY_KEEP_RECENT:
        return history  # Nothing to compact

    older = history[:-SUMMARY_KEEP_RECENT]
    recent = history[-SUMMARY_KEEP_RECENT:]
    pool = judge_model_pool or JUDGE_MODELS

    # Stage 1: Full summarization via judge call
    try:
        summary_text = _summarize_older_turns(older, pool)
        logger.info(
            f"[compaction] stage=full older_turns={len(older)} → summary block"
        )
        return [
            {
                "role": "system",
                "content": f"[INTERVIEW SUMMARY]\n{summary_text}",
            }
        ] + recent
    except Exception as e:
        logger.warning(f"[compaction] stage=full failed: {e} — trying partial")

    # Stage 2: Partial — strip verbose fields
    stripped = []
    for turn in older:
        t = dict(turn)
        # Drop bulky debug/metadata fields
        for key in ("raw_logprobs", "tool_calls", "function_call"):
            t.pop(key, None)
        # Truncate long content
        content = t.get("content", "")
        if len(content) > 400:
            t["content"] = content[:400] + " [truncated]"
        stripped.append(t)
    logger.info(f"[compaction] stage=partial stripped {len(older)} older turns")
    return stripped + recent


def _summarize_older_turns(
    older: list[dict],
    judge_pool: list[str],
) -> str:
    """
    Call judge to produce a concise summary of older interview turns.
    Uses the compaction prompt template from prompts.py.
    """
    # Lazy import to avoid circular dependency
    from .judge_client import call_judge_with_fallback

    history_text = "\n".join(
        f"{t['role'].upper()}: {t.get('content', '')[:300]}"
        for t in older
    )
    messages = build_compaction_messages(history_text)

    raw, _, _ = call_judge_with_fallback(
        messages=messages,
        judge_model_pool=judge_pool,
        max_tokens=COMPACTION_MAX_TOKENS,
        temperature=0.0,
        label="compaction",
    )
    return raw.strip()


def resolve_interview_turns(
    base_turns: int,
    running_avg: float,
    scored_questions: int,
) -> int:
    """
    Scale interview turn budget based on miner's running performance.

    Strong performers (avg >= 0.75): extend by +2 to probe mastery edge cases.
    Weak performers  (avg <  0.30): cut by -3 after at least 2 questions
                                    (fast-fail confirmed low performers).
    Others: use base_turns unchanged.

    Derived from OpenClaw run.ts:resolveMaxRunRetryIterations().
    """
    from .config import INTERVIEW_MAX_TURNS, INTERVIEW_MIN_TURNS

    if running_avg >= 0.75:
        turns = min(base_turns + 2, INTERVIEW_MAX_TURNS)
    elif running_avg < 0.30 and scored_questions >= 2:
        turns = max(base_turns - 3, INTERVIEW_MIN_TURNS)
    else:
        turns = base_turns

    logger.debug(
        f"[interview] adaptive turns={turns} "
        f"(avg={running_avg:.2f}, scored={scored_questions})"
    )
    return turns
