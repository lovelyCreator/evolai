"""
Miner Answer Sanitizer — Prompt Injection Defense

Derived from OpenClaw's scrubAnthropicRefusalMagic() pattern (run.ts)
and sanitizeForPromptLiteral() (sanitize-for-prompt.ts).
All miner output passes through sanitize_miner_answer_for_judge() before
reaching any judge prompt.

Defence layers:
  1. Unicode sanitization (strip invisible control/format chars)
  2. Proportional truncation (30% of judge context, not flat limit)
  3. Scrub known injection patterns
  4. Scrub Anthropic refusal magic tokens
  5. Wrap in XML delimiters (prevents role confusion)
"""

from __future__ import annotations

import re
import logging
import unicodedata

from .config import MINER_ANSWER_MAX_CHARS

logger = logging.getLogger(__name__)


# Known injection strings — analogous to OpenClaw's ANTHROPIC_MAGIC_STRING scrubbing
INJECTION_SCRUB_PATTERNS: list[str] = [
    "IGNORE PREVIOUS INSTRUCTIONS",
    "IGNORE ALL PREVIOUS",
    "you are now",
    "system: ",
    "```system",
    "<|system|>",
    "<|im_start|>system",
    "<|im_start|>",
    "<|im_end|>",
    "Human:",
    "Assistant:",
    "<<SYS>>",
    "<</SYS>>",
    "[INST]",
    "[/INST]",
]


def scrub_anthropic_refusal_magic(text: str) -> str:
    """
    Remove special tokens that can hijack model behavior.
    Direct analog of OpenClaw's scrubAnthropicRefusalMagic().
    """
    # Anthropic Human/Assistant turn markers
    text = re.sub(r"\n\nHuman:\s*", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\nAssistant:\s*", "\n\n", text, flags=re.IGNORECASE)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Unicode Sanitization (from OpenClaw sanitize-for-prompt.ts)
# ──────────────────────────────────────────────────────────────────────────────

# Characters to preserve even though they are technically control chars
_WHITESPACE_KEEP = frozenset({"\n", "\r", "\t"})


def sanitize_unicode_for_prompt(value: str) -> str:
    """
    Strip invisible Unicode characters that can break LLM prompt parsing.
    Derived from OpenClaw src/agents/sanitize-for-prompt.ts:sanitizeForPromptLiteral().

    Removes:
      - Cc (control chars) except \\n, \\r, \\t
      - Cf (format chars) — includes zero-width joiners, bidi overrides
      - Zl (U+2028 line separator)
      - Zp (U+2029 paragraph separator)

    These invisible characters bypass regex-based injection scrubbing and can
    confuse model tokenizers, cause inconsistent prompt hashing, or enable
    bidi text injection attacks.
    """
    return "".join(
        ch for ch in value
        if ch in _WHITESPACE_KEEP
        or unicodedata.category(ch) not in ("Cc", "Cf", "Zl", "Zp")
    )


# ──────────────────────────────────────────────────────────────────────────────
# Proportional Truncation (from OpenClaw tool-result-truncation.ts)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_max_response_chars(context_window_tokens: int) -> int:
    """
    Dynamic response size limit based on model context window.
    Derived from OpenClaw tool-result-truncation.ts:calculateMaxToolResultChars().

    No single miner answer should exceed 30% of the judge's context window.
    """
    from .config import MAX_RESPONSE_CONTEXT_SHARE, HARD_MAX_RESPONSE_CHARS, MIN_KEEP_CHARS
    proportional = int(context_window_tokens * MAX_RESPONSE_CONTEXT_SHARE * 3)  # tokens → chars
    return max(MIN_KEEP_CHARS, min(proportional, HARD_MAX_RESPONSE_CHARS))


def truncate_response_text(text: str, max_chars: int) -> str:
    """
    Truncate at newline boundary, preserving readability.
    Derived from OpenClaw tool-result-truncation.ts:truncateToolResultText().
    """
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Try to break at a newline boundary for cleaner truncation
    last_nl = cut.rfind("\n")
    if last_nl > int(max_chars * 0.8):
        cut = cut[:last_nl]
    return cut + f"\n[... truncated at {len(cut)} / {len(text)} chars]"


def sanitize_miner_answer_for_judge(
    answer: str,
    context_window_tokens: int | None = None,
) -> str:
    """
    Prepare miner output for inclusion in a judge prompt.
    
    1. Strip invisible Unicode control/format characters (sanitize-for-prompt.ts).
    2. Proportional truncation — 30% of judge context window (tool-result-truncation.ts).
       Falls back to MINER_ANSWER_MAX_CHARS if context_window_tokens not provided.
    3. Replace known prompt-injection strings with safe placeholders.
    4. Remove Anthropic refusal magic tokens.
    5. Wrap in explicit XML delimiters so the judge can't be confused about
       where miner output ends and evaluation instructions begin.
    
    Returns:
        Sanitized string wrapped in <miner_answer>…</miner_answer>.
    """
    # 1. Unicode sanitization (new — from OpenClaw sanitize-for-prompt.ts)
    text = sanitize_unicode_for_prompt(answer)

    # 2. Proportional truncation (new — from OpenClaw tool-result-truncation.ts)
    if context_window_tokens is not None:
        max_chars = calculate_max_response_chars(context_window_tokens)
    else:
        max_chars = MINER_ANSWER_MAX_CHARS
    text = truncate_response_text(text, max_chars)

    # 3. Scrub injection patterns (case-insensitive)
    for pattern in INJECTION_SCRUB_PATTERNS:
        text = re.sub(
            re.escape(pattern),
            f"[{pattern[:10].upper()}_REDACTED]",
            text,
            flags=re.IGNORECASE,
        )

    # 4. Anthropic refusal scrub
    text = scrub_anthropic_refusal_magic(text)

    # 5. Delimiter-wrap (prevents role confusion in judge)
    return f"<miner_answer>\n{text}\n</miner_answer>"
