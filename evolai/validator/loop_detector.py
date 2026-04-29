"""
Answer Loop Detector — SHA-256 Gaming Detection

Derived from OpenClaw src/agents/tool-loop-detection.ts.
Detects three loop patterns:
  - generic_repeat : same answer hash seen N times in rolling window
  - ping_pong      : alternating A-B-A-B answer pattern
  - no_progress    : identical answer for different questions

The detector is initialized per-miner and shared across Phase 2 and Phase 3.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from .config import (
    ANSWER_LOOP_HISTORY_SIZE,
    LOOP_WARNING_THRESHOLD,
    LOOP_CIRCUIT_BREAKER,
)

logger = logging.getLogger(__name__)


def hash_answer(answer: str) -> str:
    """
    SHA-256 of first 500 chars.
    Derived from OpenClaw hashToolCall / digestStable pattern.
    Focuses on semantic content, ignoring trailing whitespace or formatting.
    """
    normalized = answer[:500].strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class AnswerLoopDetector:
    """
    Detects three loop patterns from OpenClaw's LoopDetectorKind:
      - generic_repeat  : same answer hash seen N times in rolling window
      - ping_pong       : alternating A-B-A-B answer pattern
      - no_progress     : same answer hash returned for different follow-up questions

    Usage:
        detector = AnswerLoopDetector()
        # For each turn:
        detector.record(question, answer)
        result = detector.detect(answer)
        if result["stuck"]:
            ...
    """

    def __init__(self, history_size: int = ANSWER_LOOP_HISTORY_SIZE) -> None:
        self._history: list[dict] = []   # [{q_hash, a_hash}]
        self._hash_counts: dict[str, int] = {}
        self._history_size = history_size

    def record(self, question: str, answer: str) -> None:
        """Record a question-answer pair in the rolling window."""
        q_hash = hash_answer(question)
        a_hash = hash_answer(answer)
        self._history.append({"q_hash": q_hash, "a_hash": a_hash})

        # Maintain rolling window
        if len(self._history) > self._history_size:
            old = self._history.pop(0)
            old_a = old["a_hash"]
            self._hash_counts[old_a] = max(
                0, self._hash_counts.get(old_a, 0) - 1
            )

        self._hash_counts[a_hash] = self._hash_counts.get(a_hash, 0) + 1

    def detect(self, answer: str) -> dict:
        """
        Check if the given answer triggers any loop pattern.

        Returns:
            {"stuck": False}  or
            {"stuck": True, "level": "warning"|"critical",
             "detector": str, "count": int, "message": str}
        """
        a_hash = hash_answer(answer)
        count = self._hash_counts.get(a_hash, 0)

        # ── generic_repeat: same content across different questions ──
        if count >= LOOP_CIRCUIT_BREAKER:
            return {
                "stuck": True,
                "level": "critical",
                "detector": "generic_repeat",
                "count": count,
                "message": (
                    f"Identical answer hash seen {count} times "
                    f"(circuit breaker ≥ {LOOP_CIRCUIT_BREAKER}) — "
                    f"likely copy-paste evasion"
                ),
            }
        if count >= LOOP_WARNING_THRESHOLD:
            return {
                "stuck": True,
                "level": "warning",
                "detector": "generic_repeat",
                "count": count,
                "message": (
                    f"Identical answer hash seen {count} times "
                    f"(warning ≥ {LOOP_WARNING_THRESHOLD})"
                ),
            }

        # ── ping_pong: A-B-A-B alternation in last 6 entries ──
        if len(self._history) >= 6:
            tail = [h["a_hash"] for h in self._history[-6:]]
            if self._is_ping_pong(tail):
                return {
                    "stuck": True,
                    "level": "warning",
                    "detector": "ping_pong",
                    "count": 3,
                    "message": "Alternating answer pattern detected (ping-pong)",
                }

        # ── no_progress: same answer hash for different questions ──
        if len(self._history) >= 3:
            recent = self._history[-3:]
            q_hashes = {r["q_hash"] for r in recent}
            a_hashes = {r["a_hash"] for r in recent}
            if len(q_hashes) >= 2 and len(a_hashes) == 1:
                return {
                    "stuck": True,
                    "level": "warning",
                    "detector": "no_progress",
                    "count": 3,
                    "message": (
                        "Same answer for 3 different questions — "
                        "model may not be processing input"
                    ),
                }

        return {"stuck": False}

    @staticmethod
    def _is_ping_pong(hashes: list[str]) -> bool:
        """True if pattern is strictly A-B-A-B-A-B (two distinct values alternating)."""
        if len(set(hashes)) != 2:
            return False
        return all(hashes[i] != hashes[i + 1] for i in range(len(hashes) - 1))

    def reset(self) -> None:
        """Clear all state (e.g. between miners)."""
        self._history.clear()
        self._hash_counts.clear()

    @property
    def total_recorded(self) -> int:
        return len(self._history)


def apply_loop_detection_result(
    loop_result: dict,
    miner_uid: int,
) -> Optional[float]:
    """
    Act on a loop detection result.
    
    Returns:
        0.0 to circuit-break (critical loop) — caller should end evaluation.
        None to continue normally (no loop or warning-only).
    """
    if not loop_result.get("stuck"):
        return None

    level = loop_result["level"]
    detector = loop_result["detector"]
    msg = loop_result["message"]

    logger.warning(
        f"[loop-detect] uid={miner_uid} level={level} "
        f"detector={detector}: {msg}"
    )

    if level == "critical":
        logger.error(
            f"[loop-detect] CIRCUIT BREAK uid={miner_uid} — score=0.0"
        )
        return 0.0

    return None  # Warning only — penalty applied in scoring, evaluation continues
