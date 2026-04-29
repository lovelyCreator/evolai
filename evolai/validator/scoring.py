"""
Score Calculation & Normalisation

Implements the scoring formula from VALIDATOR_INTERVIEW_EVALUATION.md §Scoring Formula.

Total = Validity + ΣQuestions + InterviewBonus − LoopPenalties
Max   = 0.2 + 20.0 + 2.0 = 22.2

Anti-Gaming Reasons (score forced to 0 before/during evaluation)
─────────────────────────────────────────────────────────────────
"answer_loop_circuit_break"  — miner repeated identical response ≥ 5 times
"copy_gaming"                — model fingerprint matches another UID's model;
                               detected by model_fingerprint.py during validation
                               (exact weight hash, structural clone, or near-copy)
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import (
    VALIDITY_SCORE,
    VALIDITY_CORRECTNESS_THRESH,
    LOOP_WARNING_PENALTY,
    MAX_RAW_SCORE,
    CONTESTED_SCORE_LOW,
    CONTESTED_SCORE_HIGH,
    LOW_CONFIDENCE_DISCOUNT,
)

logger = logging.getLogger(__name__)


def calculate_final_score(
    initial_correctness: float,
    question_scores: list[float],
    interview_metrics: list[dict],
    loop_events: list[dict],
) -> dict:
    """
    Compute the final score.

    question_scores are already blended (Phase2 × 0.4 + Interview × 0.6)
    so no separate interview bonus is needed here.

    Args:
        initial_correctness: [0,1] from validity judge.
        question_scores: blended [0,1] per question (max 20 questions).
        interview_metrics: list of {"rounds_completed": int, ...} per question.
        loop_events: list of {"level": "warning"|"critical", ...}.

    Returns:
        {
            "final_score": float,
            "validity_score": float,
            "question_score": float,
            "loop_penalty": float,
            "initial_correctness": float,
            "reason": str | None,
        }
    """
    # Validity component
    validity = VALIDITY_SCORE if initial_correctness > VALIDITY_CORRECTNESS_THRESH else 0.0

    # Question component — sum of blended scores (Phase 2 + Phase 3 verdict)
    q_score = sum(question_scores)  # max = 20.0

    # Loop detection penalties
    loop_penalty = 0.0
    for evt in loop_events:
        if evt.get("level") == "critical":
            return {
                "final_score": 0.0,
                "validity_score": validity,
                "question_score": round(q_score, 4),
                "loop_penalty": 0.0,
                "initial_correctness": round(initial_correctness, 4),
                "reason": "answer_loop_circuit_break",
                "detector": evt.get("detector"),
            }
        elif evt.get("level") == "warning":
            loop_penalty += LOOP_WARNING_PENALTY

    total = validity + q_score - loop_penalty
    total = max(0.0, total)

    return {
        "final_score": round(total, 4),
        "validity_score": validity,
        "question_score": round(q_score, 4),
        "loop_penalty": round(loop_penalty, 4),
        "initial_correctness": round(initial_correctness, 4),
        "reason": None,
    }


def normalize_score(raw_score: float, max_score: float = MAX_RAW_SCORE) -> float:
    """
    Normalise raw score to [0, 1] for subnet consensus.

    Args:
        raw_score: Raw evaluation score from calculate_final_score().
        max_score: Maximum possible raw score (default 22.2).

    Returns:
        Score in [0, 1].
    """
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, raw_score / max_score))


def apply_confidence_discount(
    score: float,
    confidence: float,
    threshold: float = 0.3,
    discount: float = LOW_CONFIDENCE_DISCOUNT,
) -> float:
    """
    Apply low-confidence discount to a question score.
    When judge confidence < threshold, reduce score by (1 − discount).
    """
    if confidence < threshold:
        discounted = score * discount
        logger.debug(
            f"[score] low-confidence discount: {score:.2f} → {discounted:.2f} "
            f"(conf={confidence:.2f})"
        )
        return discounted
    return score


def is_contested(score: float) -> bool:
    """True if score falls in the contested range requiring two-judge consensus."""
    return CONTESTED_SCORE_LOW <= score <= CONTESTED_SCORE_HIGH


def score_zero_copy_gaming(
    uid: int,
    model_name: str,
    owner_uid: Optional[int],
    reason: str,
) -> dict:
    """
    Build a zero-score record for a miner caught copying another model.

    This is a convenience builder used for audit logging.  The actual
    enforcement (setting the EMA score to 0 and skipping evaluation) happens
    in the CLI round loop immediately after ``ModelValidator.validate_model``
    returns ``copy_gaming=True`` in its ``info`` dict.

    Args:
        uid:        Miner UID that was caught gaming.
        model_name: Model repo that was flagged as a copy.
        owner_uid:  UID of the original model owner (None if unknown).
        reason:     Collision description from ``fingerprints_collide()``.

    Returns:
        Score dict compatible with ``calculate_final_score()`` output shape.
    """
    logger.warning(
        f"[scoring] copy_gaming score=0: uid={uid} model={model_name!r} "
        f"copied from uid={owner_uid} reason={reason!r}"
    )
    return {
        "final_score": 0.0,
        "validity_score": 0.0,
        "question_score": 0.0,
        "loop_penalty": 0.0,
        "initial_correctness": 0.0,
        "reason": "copy_gaming",
        "copy_owner_uid": owner_uid,
        "copy_reason": reason,
    }
