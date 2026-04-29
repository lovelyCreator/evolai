"""
Judge Client — HTTP Call Layer

Central entry point for all LLM judge invocations.
No judge call is issued directly — everything passes through
call_judge_with_fallback() which wires together:
  - retry.py           (exponential backoff + jitter)
  - cooldown.py        (per-model cooldown + rotation)
  - context_guard.py   (pre-flight context window check)
  - rate_limiter.py    (fixed-window rate limiting — from OpenClaw)
  - payload_trace.py   (diagnostic tracing — from OpenClaw)

Derived from OpenClaw model-fallback.ts + retry.ts + context-window-guard.ts +
fixed-window-rate-limit.ts + cache-trace.ts + anthropic-payload-log.ts.
"""

from __future__ import annotations

import json
import re
import time
import logging
from typing import Any, Optional

from openai import OpenAI

from .config import (
    JUDGE_MODELS,
    JUDGE_TEMPERATURE,
    JUDGE_TIMEOUT_S,
    LOCAL_JUDGE_ENDPOINTS,
    LOCAL_API_KEY,
    VLLM_BASE_URL,
)
from .retry import retry_judge_call
from .cooldown import FallbackAttempt, get_cooldown_store
from .context_guard import check_judge_context_window
from .rate_limiter import wait_for_judge_slot
from .payload_trace import get_tracer, STAGE_PROMPT_BEFORE, STAGE_RESPONSE_RECEIVED
from .error_handling import FailoverError, FailoverReason, classify_judge_error

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Client cache  (one OpenAI client per base_url, reused across calls)
# ──────────────────────────────────────────────────────────────────────────────

_client_cache: dict[str, OpenAI] = {}


def _get_judge_client(model_name: str) -> tuple[OpenAI, str]:
    """
    Return (OpenAI client, base_url) for the given judge model.
    Uses the OpenAI SDK pointed at a local vLLM / Ollama server.
    Derived from OpenClaw's provider-config layer (models-config.providers.ts).
    """
    base_url = LOCAL_JUDGE_ENDPOINTS.get(model_name, VLLM_BASE_URL)
    if base_url not in _client_cache:
        _client_cache[base_url] = OpenAI(
            base_url=base_url,
            api_key=LOCAL_API_KEY,
            timeout=JUDGE_TIMEOUT_S,
        )
    return _client_cache[base_url], base_url


def _invoke_judge(
    client: OpenAI,
    model_name: str,
    messages: list[dict],
    *,
    max_tokens: int = 512,
    temperature: float = JUDGE_TEMPERATURE,
) -> str:
    """
    Single judge call via OpenAI-compat /v1/chat/completions.
    Returns the assistant content string.

    Handles Qwen3 reasoning models: if content is empty, falls back to
    the reasoning field (analogous to OpenClaw's Qwen3 empty-content fallback).
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = response.choices[0]
    content = (choice.message.content or "").strip()

    # Qwen3 reasoning fallback: model may emit reasoning instead of content
    if not content and hasattr(choice.message, "reasoning"):
        content = (choice.message.reasoning or "").strip()
        if content:
            logger.debug(f"[judge] {model_name}: used reasoning fallback")

    if not content:
        raise ValueError(f"Judge {model_name} returned empty response")

    # Strip <think>...</think> reasoning preamble from thinking models (Qwen3, etc.)
    # Must happen before returning so callers never receive raw <think> blocks.
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if cleaned != content:
        logger.debug(f"[judge] {model_name}: stripped <think> block ({len(content)-len(cleaned)} chars removed)")
        content = cleaned

    return content


# ──────────────────────────────────────────────────────────────────────────────
# Transcript session  (high-level per-miner conversation log)
# ──────────────────────────────────────────────────────────────────────────────

class TranscriptSession:
    """
    Per-miner evaluation transcript.  Emits a human-readable, visually
    structured log at DEBUG level (``evolcli validator run --debug``).

    Layout::

        ╔══ EVALUATION TRANSCRIPT … ══╗

        ━━━ PHASE 1 · SANITY CHECK ━━━
          ▶ QUESTION    │ …
          ▶ MINER       │ …
          ▶ EVALUATION  │ valid: yes  |  correctness: 0.92

        ━━━ Q1 · SCORING ━━━
          ▶ QUESTION    │ …
          ▶ MINER       │ …
          ▶ EVALUATION  │ score: 0.720  |  confidence: 0.85

        ━━━ Q1 · INTERVIEW ━━━
          ▶ INITIAL ANALYSIS │ {…}
          ┄┄┄ Turn 1 ┄┄┄
          ▶ FOLLOW-UP  │ …
          ▶ MINER      │ …
          ┄┄┄ Turn 2 ┄┄┄
          …

        ╚══ END OF TRANSCRIPT ══╝
    """

    W = 72      # full-width banner
    INDENT = "  "

    def __init__(self, miner_uid: int, model_name: str) -> None:
        self.miner_uid  = miner_uid
        self.model_name = model_name
        self._active    = False

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "TranscriptSession":
        import sys as _sys
        self._active = logger.isEnabledFor(logging.DEBUG)
        if self._active:
            border = "═" * self.W
            title  = f"  EVALUATION TRANSCRIPT  ·  uid={self.miner_uid}  ·  {self.model_name}"
            logger.debug(f"\n╔{border}╗\n║{title:<{self.W}}║\n╚{border}╝")
            _sys.stdout.flush()
        return self

    def __exit__(self, *_) -> None:
        if self._active:
            border = "═" * self.W
            footer = f"  END OF TRANSCRIPT  ·  uid={self.miner_uid}"
            logger.debug(f"\n╔{border}╗\n║{footer:<{self.W}}║\n╚{border}╝\n")
        self._active = False

    # ── internal helpers ──────────────────────────────────────────────────

    def _banner(self, title: str) -> str:
        """━━━ TITLE ━━━ full-width banner."""
        body  = f"  {title}  "
        sides = max(0, self.W - len(body))
        left  = sides // 2
        right = sides - left
        return "━" * left + body + "━" * right

    def _block(self, label: str, text: str) -> list[str]:
        """
        Render one labelled block:
            ▶ LABEL
              │ line 1
              │ line 2
        Long content is wrapped after 66 chars to keep it readable.
        """
        lines  = [f"{self.INDENT}▶ {label}"]
        prefix = f"{self.INDENT}  │ "
        max_w  = 66
        for raw_line in text.splitlines():
            raw_line = raw_line.rstrip()
            if not raw_line:
                lines.append(f"{self.INDENT}  │")
                continue
            # simple word-wrap
            while len(raw_line) > max_w:
                lines.append(prefix + raw_line[:max_w])
                raw_line = raw_line[max_w:]
            lines.append(prefix + raw_line)
        return lines

    def _turn_divider(self, turn: int) -> str:
        body  = f"  Turn {turn}  "
        sides = max(0, self.W - len(body) - 4)
        return f"{self.INDENT}┄" + "┄" * (sides // 2) + body + "┄" * (sides - sides // 2)

    def _emit(self, *parts: str) -> None:
        if self._active:
            logger.debug("\n" + "\n".join(parts))

    # ── public API ────────────────────────────────────────────────────────

    def log_sanity_check(
        self,
        question: str,
        miner_answer: str,
        verdict: dict,
    ) -> None:
        """Phase 1 — validity / sanity-check round."""
        valid       = verdict.get("valid", "?")
        correctness = verdict.get("correctness", "?")
        reason      = verdict.get("reason", "")
        eval_text   = f"valid: {valid}  |  correctness: {correctness}"
        if reason:
            eval_text += f"\n{self.INDENT}  │ reason: {reason}"

        lines = [
            "",
            self._banner("PHASE 1 · SANITY CHECK"),
            "",
            *self._block("QUESTION",   question),
            "",
            *self._block("MINER",      miner_answer),
            "",
            *self._block("EVALUATION", eval_text),
        ]
        self._emit(*lines)

    def log_question_score(
        self,
        q_num: int,
        question: str,
        miner_answer: str,
        score: float,
        confidence: float,
        consensus: bool = False,
    ) -> None:
        """Phase 2 — knowledge-assessment scoring."""
        eval_text = f"score: {score:.3f}  |  confidence: {confidence:.2f}"
        if consensus:
            eval_text += "  |  consensus"

        lines = [
            "",
            self._banner(f"Q{q_num} · SCORING"),
            "",
            *self._block("QUESTION",   question),
            "",
            *self._block("MINER",      miner_answer),
            "",
            *self._block("EVALUATION", eval_text),
        ]
        self._emit(*lines)

    def log_interview_init(
        self,
        q_num: int,
        initial_analysis: dict,
    ) -> None:
        """Phase 3 — judge's initial analysis before follow-up turns."""
        import json as _json
        text = _json.dumps(initial_analysis, indent=2) if initial_analysis else "(none)"

        lines = [
            "",
            self._banner(f"Q{q_num} · INTERVIEW"),
            "",
            *self._block("INITIAL ANALYSIS", text),
        ]
        self._emit(*lines)

    def log_interview_turn(
        self,
        q_num: int,
        turn: int,
        follow_up_question: str,
        miner_answer: str,
    ) -> None:
        """Phase 3 — one follow-up/answer exchange."""
        lines = [
            "",
            self._turn_divider(turn),
            "",
            *self._block("FOLLOW-UP", follow_up_question),
            "",
            *self._block("MINER",     miner_answer),
        ]
        self._emit(*lines)


def call_judge_with_fallback(
    *,
    messages: list[dict],
    judge_model_pool: list[str] | None = None,
    max_tokens: int = 512,
    temperature: float = JUDGE_TEMPERATURE,
    label: str = "judge",
) -> tuple[str, str, int]:
    """
    Call a judge model with retry + cooldown-aware fallback.

    This is the ONLY function that should be used to issue judge calls.

    Args:
        messages: Chat messages (system + user).
        judge_model_pool: Ordered list of model names to try.
        max_tokens: Max completion tokens.
        temperature: Sampling temperature.
        label: Descriptive label for logging.

    Returns:
        (response_text, model_name_used, total_attempts)

    Raises:
        RuntimeError: All judges exhausted.
        ContextOverflowError: Context window too small to proceed.
    """
    pool = judge_model_pool or JUDGE_MODELS
    fallback = FallbackAttempt(pool)
    store = get_cooldown_store()
    tracer = get_tracer()
    total_attempts = 0

    while True:
        model_name = fallback.next_available()
        if model_name is None:
            raise RuntimeError(
                f"[{label}] All {len(pool)} judge models exhausted or in cooldown"
            )

        # Pre-flight: context window guard
        guard = check_judge_context_window(
            messages, model_name, max_response_tokens=max_tokens
        )
        if not guard["ok"]:
            from .error_handling import ContextOverflowError
            raise ContextOverflowError(guard["message"])

        # Rate limiting (from OpenClaw fixed-window-rate-limit.ts)
        wait_for_judge_slot()

        # Payload tracing (from OpenClaw cache-trace.ts)
        tracer.trace(STAGE_PROMPT_BEFORE, messages, {"model": model_name, "label": label})

        client, base_url = _get_judge_client(model_name)

        try:
            t0 = time.monotonic()
            result = retry_judge_call(
                lambda: _invoke_judge(
                    client, model_name, messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                label=f"{label}/{model_name}",
            )
            latency_ms = (time.monotonic() - t0) * 1000

            # Trace successful response
            tracer.trace_response(
                STAGE_RESPONSE_RECEIVED, result, model=model_name,
                metadata={"latency_ms": round(latency_ms, 1), "label": label},
            )

            # Success — clear any residual cooldown
            store.clear_cooldown(model_name)
            total_attempts += fallback.tried_count
            return result, model_name, total_attempts

        except Exception as e:
            total_attempts += 1
            # Classify using structured FailoverReason (from OpenClaw failover-error.ts)
            reason = classify_judge_error(e)
            store.mark_failure(model_name, e)
            logger.warning(
                f"[{label}] {model_name} failed (reason={reason.value}) "
                f"after retries, rotating to next judge: {e}"
            )
            # Continue loop → FallbackAttempt picks next model


def parse_judge_json(raw: str) -> dict[str, Any]:
    """
    Best-effort JSON extraction from judge response.
    Handles models that wrap JSON in markdown code fences, prose,
    or <think>...</think> reasoning blocks (Qwen3 / thinking models).
    """
    text = raw.strip()

    # Strip <think>...</think> reasoning blocks — must be first so subsequent
    # { } scanning doesn't pick up JSON fragments inside the think block.
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if stripped != text:
        logger.debug(f"[parse_judge_json] stripped <think> block ({len(text)-len(stripped)} chars)")
        text = stripped

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        inner = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    # Extract first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: return raw as a dict
    logger.warning(f"[parse_judge_json] Could not parse JSON, returning raw text")
    return {"raw": text}
