"""
Lifecycle Events — W&B Audit Trail

Derived from OpenClaw pi-embedded-subscribe.handlers.lifecycle.ts:emitAgentEvent().
Fire-and-forget: never blocks the main evaluation path.

Event types:
  - evaluation_start   : new miner evaluation begins
  - evaluation_end     : miner evaluation complete
  - validity_check     : Phase 1 result
  - question_start     : Phase 2 question begins
  - question_end       : Phase 2 question scored
  - interview_start    : Phase 3 interview begins
  - turn_start         : interview turn begins
  - turn_end           : interview turn complete
  - auto_compaction    : context compaction triggered
  - interview_end      : Phase 3 interview complete
  - loop_detected      : answer loop warning or circuit break
  - judge_fallback     : judge model rotation occurred
  - server_event       : vLLM server start/stop/swap
  - round_start        : evaluation round begins (judge sampled)
  - round_end          : evaluation round complete (GPU freed)
  - miner_round_start  : single miner evaluation begins within round
  - miner_round_end    : single miner evaluation complete (miner server stopped)
"""

from __future__ import annotations

import time
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try to import wandb; if not available, events are logged only
_wandb_available = False
try:
    import wandb
    _wandb_available = True
except ImportError:
    pass


def emit_event(
    event_type: str,
    miner_uid: int = -1,
    turn: int = 0,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """
    Emit a structured lifecycle event.
    
    Fire-and-forget: logs the event and attempts W&B logging.
    Never raises — evaluation must not be blocked by telemetry failures.
    
    Args:
        event_type: One of the documented event types.
        miner_uid: Miner UID being evaluated (-1 for global events).
        turn: Current interview turn (0 for non-interview events).
        metadata: Additional key-value pairs to include.
    """
    payload: dict[str, Any] = {
        "event": event_type,
        "miner_uid": miner_uid,
        "turn": turn,
        "ts": time.time(),
    }
    if metadata:
        payload.update(metadata)

    # Always log locally
    logger.info(f"[lifecycle] {event_type} uid={miner_uid} turn={turn}")
    logger.debug(f"[lifecycle] payload: {payload}")

    # Attempt W&B logging (fire-and-forget)
    if _wandb_available:
        try:
            if wandb.run is not None:
                wandb.log(payload)
        except Exception as e:
            logger.debug(f"[lifecycle] wandb.log failed (non-fatal): {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience wrappers
# ──────────────────────────────────────────────────────────────────────────────

def emit_evaluation_start(miner_uid: int, model_name: str, **kwargs: Any) -> None:
    emit_event("evaluation_start", miner_uid, metadata={"model": model_name, **kwargs})

def emit_evaluation_end(miner_uid: int, score: float, **kwargs: Any) -> None:
    emit_event("evaluation_end", miner_uid, metadata={"score": score, **kwargs})

def emit_validity_check(miner_uid: int, passed: bool, correctness: float, **kwargs: Any) -> None:
    emit_event("validity_check", miner_uid, metadata={"passed": passed, "correctness": correctness, **kwargs})

def emit_question_start(miner_uid: int, question_idx: int, **kwargs: Any) -> None:
    emit_event("question_start", miner_uid, metadata={"question_idx": question_idx, **kwargs})

def emit_question_end(miner_uid: int, question_idx: int, score: float, **kwargs: Any) -> None:
    emit_event("question_end", miner_uid, metadata={"question_idx": question_idx, "score": score, **kwargs})

def emit_interview_start(miner_uid: int, question_idx: int, adaptive_turns: int, **kwargs: Any) -> None:
    emit_event("interview_start", miner_uid, metadata={"question_idx": question_idx, "adaptive_turns": adaptive_turns, **kwargs})

def emit_turn_start(miner_uid: int, turn: int, **kwargs: Any) -> None:
    emit_event("turn_start", miner_uid, turn, metadata=kwargs)

def emit_turn_end(miner_uid: int, turn: int, **kwargs: Any) -> None:
    emit_event("turn_end", miner_uid, turn, metadata=kwargs)

def emit_auto_compaction(miner_uid: int, turn: int, **kwargs: Any) -> None:
    emit_event("auto_compaction", miner_uid, turn, metadata=kwargs)

def emit_interview_end(miner_uid: int, total_turns: int, total_tokens: int, **kwargs: Any) -> None:
    emit_event("interview_end", miner_uid, metadata={"total_turns": total_turns, "total_tokens": total_tokens, **kwargs})

def emit_loop_detected(miner_uid: int, turn: int, level: str, detector: str, **kwargs: Any) -> None:
    emit_event("loop_detected", miner_uid, turn, metadata={"level": level, "detector": detector, **kwargs})

def emit_judge_fallback(miner_uid: int, from_model: str, to_model: str, **kwargs: Any) -> None:
    emit_event("judge_fallback", miner_uid, metadata={"from": from_model, "to": to_model, **kwargs})
