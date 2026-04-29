"""
Evaluation Orchestrator — Async Main Pipeline

Single entry point that wires together ALL modular components into the
three-phase evaluation pipeline described in VALIDATOR_INTERVIEW_EVALUATION.md.

Architecture (from §Agent Design & Orchestration):
  Orchestrator
    ├─ Phase 1: Validity Check
    ├─ Phase 2: Knowledge Assessment (20 questions)
    └─ Phase 3: Adaptive Interview (per question)

Two-Tier Run Loop (from OpenClaw run.ts):
  Outer loop:  judge model rotation — handles OOM, auth, rate-limit
  Inner loop:  interview turns per question — handles context overflow via compaction

All judge calls route through judge_client.call_judge_with_fallback().
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import (
    JUDGE_MODELS,
    JUDGE_TEMPERATURE,
    NUM_QUESTIONS,
    VALIDITY_SCORE,
    VALIDITY_CORRECTNESS_THRESH,
    INTERVIEW_BASE_TURNS,
    PER_RESPONSE_TOKEN_LIMIT,
    TOTAL_INTERVIEW_TOKEN_LIMIT,
    PHASE2_WEIGHT,
    INTERVIEW_WEIGHT,
    MINER_ANSWER_MAX_CHARS,
    COMPACTION_TRIGGER_K,
    MAX_OVERFLOW_COMPACTION_ATTEMPTS,
    BASE_RUN_RETRY_ITERATIONS,
    RUN_RETRY_PER_PROFILE,
    CONTESTED_SCORE_LOW,
    CONTESTED_SCORE_HIGH,
    PHASE2_SKIP_INTERVIEW_THRESHOLD,
    LOW_CONFIDENCE_DISCOUNT,
    OLLAMA_CHAT_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    VLLM_MINER_BASE_URL,
)
from .error_handling import ContextOverflowError
from .judge_client import call_judge_with_fallback, parse_judge_json, TranscriptSession


async def _call_judge_async(
    **kwargs,
) -> tuple[str, str, int]:
    """
    Run the synchronous call_judge_with_fallback in a thread pool
    so it doesn't block the async event loop.
    """
    return await asyncio.to_thread(
        call_judge_with_fallback, **kwargs
    )
from .prompts import (
    build_validity_messages,
    build_scoring_messages,
    build_initial_interview_messages,
    build_followup_interview_messages,
    build_final_interview_verdict_messages,
)
from .sanitizer import sanitize_miner_answer_for_judge
from .loop_detector import AnswerLoopDetector, apply_loop_detection_result
from .compaction import (
    TokenTracker,
    should_compact,
    compact_conversation_history,
    resolve_interview_turns,
)
from .scoring import (
    calculate_final_score,
    normalize_score,
    apply_confidence_discount,
    is_contested,
)
from .usage import UsageAccumulator
from .streaming import stream_miner_response, stream_miner_response_vllm, stream_miner_response_hf, StreamAccumulator
from .lifecycle import (
    emit_evaluation_start,
    emit_evaluation_end,
    emit_validity_check,
    emit_question_start,
    emit_question_end,
    emit_interview_start,
    emit_interview_end,
    emit_turn_start,
    emit_turn_end,
    emit_auto_compaction,
    emit_loop_detected,
    emit_judge_fallback,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Run-loop helpers  (from OpenClaw run.ts)
# ──────────────────────────────────────────────────────────────────────────────

def resolve_max_judge_call_attempts(judge_pool_size: int) -> int:
    """
    Compute max judge call attempts based on pool size.
    Derived from OpenClaw run.ts resolveMaxRunRetryIterations():
      result = BASE + max(1, pool_size) * PER_PROFILE
      clamped to [32, 160]
    """
    raw = BASE_RUN_RETRY_ITERATIONS + max(1, judge_pool_size) * RUN_RETRY_PER_PROFILE
    return max(32, min(160, raw))


# ──────────────────────────────────────────────────────────────────────────────
# Result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """Result from Phase 2 scoring of a single question."""
    question_text: str = ""
    reference_answer: str = ""
    miner_answer: str = ""
    score: float = 0.0
    confidence: float = 0.5
    judge_used: str = ""
    consensus_used: bool = False
    loop_event: Optional[dict] = None


@dataclass
class InterviewResult:
    """Result from Phase 3 adaptive interview for a single question."""
    rounds_completed: int = 0
    total_tokens: int = 0
    compaction_events: int = 0
    loop_events: list = field(default_factory=list)
    final_analysis: str = ""
    interview_score: float = 0.0   # Judge's final verdict on genuine understanding [0,1]


@dataclass
class EvaluationOutput:
    """Complete output of one miner evaluation."""
    miner_uid: int = -1
    model_name: str = ""
    final_score: float = 0.0
    normalized_score: float = 0.0
    validity_passed: bool = False
    initial_correctness: float = 0.0
    question_results: list[QuestionResult] = field(default_factory=list)
    interview_results: list[InterviewResult] = field(default_factory=list)
    loop_events: list[dict] = field(default_factory=list)
    score_breakdown: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationOrchestrator:
    """
    Async evaluation orchestrator.

    Coordinates all three phases and the two-tier run loop.
    Derived from §Agent Design & Orchestration in
    VALIDATOR_INTERVIEW_EVALUATION.md.

    Usage:
        orch = EvaluationOrchestrator(
            judge_model_pool=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
            miner_backend="ollama",          # or "vllm"
            miner_model_id="evolai-1.5b",
        )
        output = await orch.evaluate_miner(
            miner_uid=42,
            model_name="user/evolai-model",
            questions=sampled_questions,
        )
    """

    def __init__(
        self,
        *,
        judge_model_pool: list[str] | None = None,
        miner_backend: str = "ollama",       # "ollama" or "vllm"
        miner_model_id: str = "",
        miner_vllm_base_url: str = VLLM_MINER_BASE_URL,
        ollama_url: str = OLLAMA_CHAT_URL,
        ollama_num_ctx: int = OLLAMA_NUM_CTX,
    ) -> None:
        self.judge_pool = judge_model_pool or JUDGE_MODELS
        self.miner_backend = miner_backend
        self.miner_model_id = miner_model_id
        self.miner_vllm_base_url = miner_vllm_base_url
        self.ollama_url = ollama_url
        self.ollama_num_ctx = ollama_num_ctx

        # Set by evaluate_miner() when miner_backend="hf"
        self.miner_hf_model = None
        self.miner_hf_tokenizer = None

        # Per-evaluation state — reset in evaluate_miner()
        self._usage: UsageAccumulator = UsageAccumulator()
        self._loop_detector: AnswerLoopDetector = AnswerLoopDetector()
        self._all_loop_events: list[dict] = []
        self._run_loop_iterations: int = 0
        self._max_attempts: int = resolve_max_judge_call_attempts(len(self.judge_pool))
        self._tx: TranscriptSession = TranscriptSession(-1, "")

    # ──────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────

    async def evaluate_miner(
        self,
        miner_uid: int,
        model_name: str,
        questions: list[dict],
        *,
        instruction_for_validity: str = "",
        skip_sanity_check: bool = False,
        miner_backend: str | None = None,
        miner_hf_model=None,
        miner_hf_tokenizer=None,
    ) -> EvaluationOutput:
        """
        Run the full three-phase evaluation pipeline for one miner.

        Args:
            miner_uid: Miner UID on the subnet.
            model_name: HuggingFace model name being evaluated.
            questions: List of question dicts from DatasetSampler
                       (each has 'question', 'answer', 'dataset_id', …).
            instruction_for_validity: Distinct instruction (not from the scored
                question pool) used for Phase 1 sanity check.  If empty and
                skip_sanity_check is False, falls back to questions[0].
            skip_sanity_check: When True, bypass Phase 1 entirely and treat the
                miner as valid (initial_correctness = 1.0).

        Returns:
            EvaluationOutput with scores, breakdown, usage, etc.
        """
        t0 = time.time()
        output = EvaluationOutput(miner_uid=miner_uid, model_name=model_name)

        # Override backend/model for this specific evaluation if supplied
        if miner_backend is not None:
            self.miner_backend = miner_backend
        if miner_hf_model is not None:
            self.miner_hf_model = miner_hf_model
            self.miner_hf_tokenizer = miner_hf_tokenizer
        # Always update the model id so vLLM /v1/chat/completions gets the correct model name
        self.miner_model_id = model_name

        # Reset per-evaluation state
        self._usage = UsageAccumulator()
        self._loop_detector = AnswerLoopDetector()
        self._all_loop_events = []
        self._run_loop_iterations = 0
        self._tx = TranscriptSession(miner_uid=miner_uid, model_name=model_name)

        emit_evaluation_start(miner_uid, model_name, metadata={
            "num_questions": len(questions),
            "judge_pool": self.judge_pool,
        })

        # ── One flowing transcript per miner (active when --debug / DEBUG log) ──
        with self._tx:
            try:
                # ── Phase 1: Validity Check (sanity check) ───────────────
                if skip_sanity_check:
                    logger.info("[orch] Phase 1 skipped (skip_sanity_check=True)")
                    initial_correctness = 1.0
                else:
                    validity_instruction = (
                        instruction_for_validity
                        or self._extract_question_text(questions[0])
                    )
                    initial_correctness = await self._phase1_validity(
                        miner_uid, validity_instruction,
                    )
                output.initial_correctness = initial_correctness
                output.validity_passed = initial_correctness > VALIDITY_CORRECTNESS_THRESH

                if not output.validity_passed:
                    output.final_score = 0.0
                    output.normalized_score = 0.0
                    output.score_breakdown = {
                        "final_score": 0.0,
                        "reason": "validity_failed",
                        "initial_correctness": round(initial_correctness, 4),
                    }
                    output.elapsed_s = round(time.time() - t0, 2)
                    emit_evaluation_end(miner_uid, 0.0, metadata={
                        "reason": "validity_failed",
                    })
                    return output

                # ── Phase 2 + 3: Questions + Interviews ──────────────────
                question_scores: list[float] = []
                interview_metrics: list[dict] = []
                running_sum = 0.0

                for q_idx, q_data in enumerate(questions):
                    q_text = self._extract_question_text(q_data)
                    ref_answer = self._extract_reference_answer(q_data)
                    if not q_text:
                        logger.warning(f"[orch] Q{q_idx+1} has no text, skipping")
                        continue

                    emit_question_start(miner_uid, q_idx + 1, metadata={
                        "dataset_id": q_data.get("dataset_id", ""),
                    })

                    try:
                        # Phase 2: get miner answer + score
                        q_result = await self._phase2_score_question(
                            miner_uid, q_idx, q_text, ref_answer,
                        )
                        output.question_results.append(q_result)

                        if q_result.loop_event:
                            self._all_loop_events.append(q_result.loop_event)
                            if q_result.loop_event.get("level") == "critical":
                                # Circuit-break — terminate entire evaluation
                                logger.warning(f"[orch] Critical loop at Q{q_idx+1}, stopping evaluation")
                                break

                        # Skip interview when Phase 2 score is too low (wrong answer):
                        # the miner clearly didn't understand the question; running an
                        # interview would waste ~7 turns and likely produce noise.
                        if q_result.score < PHASE2_SKIP_INTERVIEW_THRESHOLD:
                            logger.info(
                                f"[orch] Q{q_idx+1} Phase2 score={q_result.score:.3f} "
                                f"< {PHASE2_SKIP_INTERVIEW_THRESHOLD} "
                                f"— skipping interview (wrong answer)"
                            )
                            blended = q_result.score * PHASE2_WEIGHT  # interview contribution = 0
                            question_scores.append(blended)
                            running_sum += blended
                            emit_question_end(miner_uid, q_idx + 1, blended, metadata={
                                "phase2_score": q_result.score,
                                "interview_skipped": True,
                                "skip_reason": "wrong_answer",
                                "blended_score": blended,
                            })
                            continue  # skip _phase3_interview for this question

                        # Phase 3: adaptive interview (uses running_avg from previous questions)
                        iv_result = await self._phase3_interview(
                            miner_uid=miner_uid,
                            q_idx=q_idx,
                            question=q_text,
                            reference_answer=ref_answer,
                            miner_answer=q_result.miner_answer,
                            running_avg=running_sum / max(q_idx, 1),
                            scored_questions=q_idx + 1,
                        )
                        output.interview_results.append(iv_result)
                        interview_metrics.append({
                            "rounds_completed": iv_result.rounds_completed,
                        })
                        self._all_loop_events.extend(iv_result.loop_events)

                        # Blend Phase 2 (initial answer quality) and Phase 3 (interview verdict)
                        blended = (
                            q_result.score * PHASE2_WEIGHT
                            + iv_result.interview_score * INTERVIEW_WEIGHT
                        )
                        question_scores.append(blended)
                        running_sum += blended

                        logger.info(
                            f"[orch] Q{q_idx+1} final: phase2={q_result.score:.3f} "
                            f"interview={iv_result.interview_score:.3f} blended={blended:.3f}"
                        )
                        emit_question_end(miner_uid, q_idx + 1, blended, metadata={
                            "phase2_score": q_result.score,
                            "interview_score": iv_result.interview_score,
                            "blended_score": blended,
                            "interview_rounds": iv_result.rounds_completed,
                        })

                    except Exception as q_err:
                        # Log and continue — one bad question must not abort the rest
                        logger.error(
                            f"[orch] Q{q_idx+1} failed, skipping: {q_err}",
                            exc_info=True,
                        )
                        emit_question_end(miner_uid, q_idx + 1, 0.0, metadata={
                            "error": str(q_err),
                        })

                # ── Final scoring ─────────────────────────────────────────
                output.loop_events = self._all_loop_events
                breakdown = calculate_final_score(
                    initial_correctness=initial_correctness,
                    question_scores=question_scores,
                    interview_metrics=interview_metrics,
                    loop_events=self._all_loop_events,
                )
                output.score_breakdown = breakdown
                output.final_score = breakdown["final_score"]
                output.normalized_score = normalize_score(output.final_score)
                output.usage = self._usage.summary()

            except Exception as e:
                logger.error(f"[orch] Evaluation failed for uid={miner_uid}: {e}", exc_info=True)
                output.error = str(e)
                output.final_score = 0.0
                output.normalized_score = 0.0

            output.elapsed_s = round(time.time() - t0, 2)
            emit_evaluation_end(
                miner_uid,
                output.final_score,
                normalized_score=output.normalized_score,
                elapsed_s=output.elapsed_s,
                error=output.error,
            )
            return output

    # ──────────────────────────────────────────────────────────────────────
    # Phase 1 — Validity
    # ──────────────────────────────────────────────────────────────────────

    async def _phase1_validity(
        self,
        miner_uid: int,
        instruction: str,
    ) -> float:
        """
        Phase 1: Send instruction to miner, judge the response for validity.
        Returns initial_correctness ∈ [0, 1].
        """
        # 1. Get miner response
        miner_acc = await self._get_miner_response(
            [{"role": "user", "content": instruction}],
        )
        raw_answer = miner_acc.content
        self._usage.merge_ollama(miner_acc, source="miner", model=self.miner_model_id)

        # 2. Sanitize
        sanitized = sanitize_miner_answer_for_judge(raw_answer)

        # 3. Judge via call layer
        messages = build_validity_messages(instruction, sanitized)
        raw_response, judge_used, attempts = await _call_judge_async(
            messages=messages,
            judge_model_pool=self.judge_pool,
            max_tokens=200,
            temperature=0.0,
            label="validity",
        )
        parsed = parse_judge_json(raw_response)

        # Parse correctness: prefer float "correctness" field, fall back to
        # binary derivation from "valid" string (design doc §Phase 1)
        correctness_raw = parsed.get("correctness")
        if correctness_raw is not None:
            try:
                correctness = float(correctness_raw)
                correctness = max(0.0, min(1.0, correctness))
            except (ValueError, TypeError):
                correctness = 0.0
        else:
            valid_str = str(parsed.get("valid", "no")).lower()
            correctness = 1.0 if valid_str in ("yes", "true", "1") else 0.0

        # Transcript: sanity-check round
        self._tx.log_sanity_check(instruction, raw_answer, parsed)

        emit_validity_check(miner_uid, passed=correctness > VALIDITY_CORRECTNESS_THRESH, correctness=correctness, metadata={
            "valid": parsed.get("valid", "unknown"),
            "judge_used": judge_used,
            "attempts": attempts,
        })
        logger.info(
            f"[orch] Phase 1 validity: correctness={correctness:.2f} "
            f"(judge={judge_used})"
        )
        return correctness

    # ──────────────────────────────────────────────────────────────────────
    # Phase 2 — Knowledge Assessment
    # ──────────────────────────────────────────────────────────────────────

    async def _phase2_score_question(
        self,
        miner_uid: int,
        q_idx: int,
        question: str,
        reference_answer: str,
    ) -> QuestionResult:
        """
        Phase 2: Get miner answer, run loop detection, score with judge.
        Implements two-judge consensus for contested scores.
        """
        result = QuestionResult(
            question_text=question,
            reference_answer=reference_answer,
        )

        # 1. Get miner answer
        miner_acc = await self._get_miner_response(
            [{"role": "user", "content": question}],
        )
        raw_answer = miner_acc.content
        result.miner_answer = raw_answer
        self._usage.merge_ollama(miner_acc, source="miner", model=self.miner_model_id)

        # 2. Loop detection before judge call
        self._loop_detector.record(question, raw_answer)
        loop_result = self._loop_detector.detect(raw_answer)
        circuit = apply_loop_detection_result(loop_result, miner_uid)

        if loop_result.get("stuck"):
            result.loop_event = loop_result
            emit_loop_detected(miner_uid, turn=q_idx + 1, metadata=loop_result)
            if circuit is not None:
                result.score = 0.0
                return result

        # 3. Sanitize + build scoring prompt
        sanitized = sanitize_miner_answer_for_judge(raw_answer)
        messages = build_scoring_messages(question, reference_answer, sanitized)

        # 4. Primary judge call
        raw_resp, judge1, attempts = await _call_judge_async(
            messages=messages,
            judge_model_pool=self.judge_pool,
            max_tokens=300,
            temperature=0.0,
            label=f"score/Q{q_idx+1}",
        )
        parsed = parse_judge_json(raw_resp)
        score1 = float(parsed.get("score", 0.0))
        conf1 = float(parsed.get("confidence", 0.5))
        result.judge_used = judge1

        # 5. Two-judge consensus for contested scores
        if is_contested(score1) and len(self.judge_pool) > 1:
            pool2 = [m for m in self.judge_pool if m != judge1]
            if pool2:
                try:
                    raw_resp2, judge2, _ = await _call_judge_async(
                        messages=messages,
                        judge_model_pool=pool2,
                        max_tokens=300,
                        temperature=0.0,
                        label=f"score/Q{q_idx+1}/consensus",
                    )
                    parsed2 = parse_judge_json(raw_resp2)
                    score2 = float(parsed2.get("score", 0.0))
                    score1 = (score1 + score2) / 2.0
                    result.consensus_used = True
                    result.judge_used = f"{judge1}+{judge2}"
                    logger.info(
                        f"[orch] Q{q_idx+1} consensus: {score1:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"[orch] Q{q_idx+1} consensus failed: {e}")

        # 6. Low-confidence discount
        result.score = apply_confidence_discount(score1, conf1)
        result.confidence = conf1

        # Transcript: scoring round
        self._tx.log_question_score(
            q_num=q_idx + 1,
            question=question,
            miner_answer=raw_answer,
            score=result.score,
            confidence=result.confidence,
            consensus=result.consensus_used,
        )

        logger.info(
            f"[orch] Q{q_idx+1}: score={result.score:.3f} "
            f"conf={conf1:.2f} judge={result.judge_used}"
        )
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Phase 3 — Adaptive Interview
    # ──────────────────────────────────────────────────────────────────────

    async def _phase3_interview(
        self,
        *,
        miner_uid: int,
        q_idx: int,
        question: str,
        reference_answer: str,
        miner_answer: str,
        running_avg: float,
        scored_questions: int,
    ) -> InterviewResult:
        """
        Phase 3: Conduct adaptive interview for one question.
        Uses the two-tier run loop with context overflow recovery.
        """
        iv = InterviewResult()

        # Adaptive turn budget
        adaptive_turns = resolve_interview_turns(
            INTERVIEW_BASE_TURNS, running_avg, scored_questions,
        )

        # Conversation history (OpenAI message format)
        history: list[dict] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": miner_answer},
        ]
        token_tracker = TokenTracker(
            response_limit=PER_RESPONSE_TOKEN_LIMIT,
            interview_limit=TOTAL_INTERVIEW_TOKEN_LIMIT,
        )
        # Count only miner (assistant) responses toward the budget
        token_tracker.add(miner_answer)

        emit_interview_start(miner_uid, q_idx + 1, adaptive_turns, metadata={
            "question_index": q_idx,
        })

        # Initial analysis by judge
        sanitized_init = sanitize_miner_answer_for_judge(miner_answer)
        init_messages = build_initial_interview_messages(
            question, reference_answer, sanitized_init,
        )
        try:
            init_raw, judge_used, _ = await _call_judge_async(
                messages=init_messages,
                judge_model_pool=self.judge_pool,
                max_tokens=400,
                temperature=0.0,
                label=f"interview/init/Q{q_idx+1}",
            )
            analysis = parse_judge_json(init_raw)
            previous_summary = json.dumps(analysis, indent=2)
        except Exception as e:
            logger.warning(f"[orch] Interview init analysis failed: {e}")
            analysis = {}
            previous_summary = "{}"

        # Transcript: initial analysis
        self._tx.log_interview_init(q_num=q_idx + 1, initial_analysis=analysis)

        # ── Interview turn loop with overflow recovery ────────────────
        compaction_count = 0

        for turn in range(1, adaptive_turns + 1):
            if not token_tracker.can_continue():
                logger.info(f"[orch] Token budget exhausted at turn {turn}")
                break

            emit_turn_start(miner_uid, turn=turn)
            turn_start = time.time()

            # Get follow-up question from judge (returns full parsed response)
            follow_up_question, followup_parsed = await self._get_followup_question(
                history, previous_summary, q_idx, turn,
                total_turns=adaptive_turns,
            )
            if follow_up_question is None:
                # Judge decided interview is complete
                logger.info(f"[orch] Judge ended interview at turn {turn}")
                iv.rounds_completed = turn - 1
                break

            # Add question to history (not counted toward miner token budget)
            history.append({"role": "user", "content": follow_up_question})

            # Get miner answer with overflow recovery
            try:
                miner_acc = await self._get_miner_response_with_recovery(
                    history,
                    compaction_count,
                    q_idx,
                    turn,
                    miner_uid,
                )
            except ContextOverflowError:
                logger.error(f"[orch] Unrecoverable context overflow at turn {turn}")
                iv.rounds_completed = turn - 1
                break

            follow_answer = miner_acc.content
            self._usage.merge_ollama(miner_acc, source="miner", model=self.miner_model_id)

            # Add answer to history and count toward miner token budget
            history.append({"role": "assistant", "content": follow_answer})
            token_tracker.add(follow_answer)

            # Transcript: interview turn exchange
            self._tx.log_interview_turn(
                q_num=q_idx + 1,
                turn=turn,
                follow_up_question=follow_up_question,
                miner_answer=follow_answer,
            )

            # Loop detection
            self._loop_detector.record(follow_up_question, follow_answer)
            loop_result = self._loop_detector.detect(follow_answer)
            circuit = apply_loop_detection_result(loop_result, miner_uid)

            elapsed = round(time.time() - turn_start, 2)
            emit_turn_end(miner_uid, turn=turn, metadata={
                "elapsed_s": elapsed,
                "loop_level": loop_result.get("level"),
                "loop_detector": loop_result.get("detector"),
                "circuit_break": circuit is not None,
                "tokens_this_turn": token_tracker.total_tokens,
            })

            if loop_result.get("stuck"):
                iv.loop_events.append(loop_result)
                emit_loop_detected(miner_uid, turn=turn, metadata=loop_result)
                if circuit is not None:
                    iv.rounds_completed = turn
                    break

            # Context compaction if history is long
            if should_compact(history):
                emit_auto_compaction(miner_uid, turn=turn)
                try:
                    history = compact_conversation_history(
                        history, self.judge_pool,
                    )
                    compaction_count += 1
                    iv.compaction_events += 1
                except Exception as e:
                    logger.warning(f"[orch] Compaction failed: {e}")

            # Update analysis summary for next turn from this turn's judge response
            # (not the frozen init analysis — this is what the judge just assessed)
            if followup_parsed:
                previous_summary = json.dumps(followup_parsed, indent=2)
            iv.rounds_completed = turn

        iv.total_tokens = token_tracker.total_tokens

        # ── Final verdict: judge rates genuine understanding across all turns ──
        try:
            verdict_messages = build_final_interview_verdict_messages(
                question=question,
                reference_answer=reference_answer,
                interview_summary=previous_summary,
            )
            verdict_raw, _, _ = await _call_judge_async(
                messages=verdict_messages,
                judge_model_pool=self.judge_pool,
                max_tokens=200,
                temperature=0.0,
                label=f"interview/verdict/Q{q_idx+1}",
            )
            verdict_parsed = parse_judge_json(verdict_raw)
            iv.interview_score = max(0.0, min(1.0, float(verdict_parsed.get("interview_score", 0.0))))
            iv.final_analysis = verdict_parsed.get("reasoning", "")
            logger.info(
                f"[orch] Q{q_idx+1} interview verdict: {iv.interview_score:.3f} "
                f"({verdict_parsed.get('genuine_understanding', '?')})"
            )
        except Exception as e:
            logger.warning(f"[orch] Interview verdict failed, using round-completion proxy: {e}")
            # Fallback: normalised round completion as rough quality proxy
            iv.interview_score = min(iv.rounds_completed / 10.0, 1.0)

        emit_interview_end(miner_uid, iv.rounds_completed, iv.total_tokens, metadata={
            "compaction_events": iv.compaction_events,
        })

        return iv

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    async def _get_miner_response(
        self,
        messages: list[dict],
    ) -> StreamAccumulator:
        """Route to HF, vLLM, or Ollama depending on miner backend."""
        if self.miner_backend == "hf":
            return await stream_miner_response_hf(
                messages=messages,
                model=self.miner_hf_model,
                tokenizer=self.miner_hf_tokenizer,
            )
        elif self.miner_backend == "vllm":
            return await stream_miner_response_vllm(
                messages=messages,
                model=self.miner_model_id,
                base_url=self.miner_vllm_base_url,
            )
        else:
            return await stream_miner_response(
                messages=messages,
                model=self.miner_model_id,
                ollama_url=self.ollama_url,
                num_ctx=self.ollama_num_ctx,
            )

    async def _get_miner_response_with_recovery(
        self,
        history: list[dict],
        compaction_count: int,
        q_idx: int,
        turn: int,
        miner_uid: int,
    ) -> StreamAccumulator:
        """
        Get miner response with staged overflow recovery.
        Mirrors OpenClaw run.ts context overflow recovery block.
        """
        had_compaction_this_attempt = False
        local_compaction_count = compaction_count

        for attempt in range(MAX_OVERFLOW_COMPACTION_ATTEMPTS + 1):
            try:
                return await self._get_miner_response(history)
            except ContextOverflowError:
                if had_compaction_this_attempt:
                    # Already compacted this attempt — retry plain
                    had_compaction_this_attempt = False
                    continue

                if local_compaction_count >= MAX_OVERFLOW_COMPACTION_ATTEMPTS:
                    raise  # Re-raise; caller handles unrecoverable overflow

                logger.info(
                    f"[orch] Context overflow Q{q_idx+1} turn {turn}, "
                    f"compacting (attempt {local_compaction_count + 1})"
                )
                try:
                    history[:] = compact_conversation_history(
                        history, self.judge_pool,
                    )
                except Exception as e:
                    logger.warning(f"[orch] Compaction failed during recovery: {e}")
                    raise ContextOverflowError(f"Compaction failed: {e}") from e

                local_compaction_count += 1
                had_compaction_this_attempt = True
                emit_auto_compaction(miner_uid, turn=turn)

        raise ContextOverflowError("Max overflow recovery attempts exceeded")

    async def _get_followup_question(
        self,
        history: list[dict],
        previous_summary: str,
        q_idx: int,
        turn: int,
        total_turns: int,
    ) -> tuple[Optional[str], dict]:
        """
        Ask the judge to generate a follow-up question.
        Returns (question, parsed) where question is None if interview should stop.
        parsed contains the full judge response including 'analysis' so the caller
        can update previous_summary with fresh per-turn context.
        """
        from .config import INTERVIEW_MIN_TURNS
        # Build compact history text for the prompt
        history_text = json.dumps(
            [{"role": m["role"], "content": m["content"][:500]} for m in history],
            indent=2,
        )
        messages = build_followup_interview_messages(
            history_text, previous_summary,
            turn_num=turn, total_turns=total_turns,
        )

        try:
            raw, judge_used, _ = await _call_judge_async(
                messages=messages,
                judge_model_pool=self.judge_pool,
                max_tokens=300,
                temperature=0.3,  # Slight creativity for diverse follow-ups
                label=f"interview/followup/Q{q_idx+1}/T{turn}",
            )
            parsed = parse_judge_json(raw)

            # Honour judge early-stop only after the minimum number of turns
            if not parsed.get("continue_interview", True) and turn >= INTERVIEW_MIN_TURNS:
                return None, parsed

            question = parsed.get("next_question", "")
            if not question or question.lower() == "null":
                # Fall back: ask the judge to restate rather than silently ending early
                if turn < INTERVIEW_MIN_TURNS:
                    logger.warning(
                        f"[orch] Judge returned empty question at turn {turn}/{total_turns} "
                        f"(before min {INTERVIEW_MIN_TURNS}), retrying with generic probe"
                    )
                    return "Can you explain that in more detail, including any edge cases?", parsed
                return None, parsed

            return question, parsed

        except Exception as e:
            logger.warning(f"[orch] Follow-up generation failed: {e}")
            return None, {}

    # ──────────────────────────────────────────────────────────────────────
    # Question data extraction helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_question_text(q_data: dict) -> str:
        """Extract question text from various formats."""
        # DatasetSampler wraps: {dataset_id, dataset_name, question: {...}}
        inner = q_data.get("question", q_data)
        if isinstance(inner, str):
            return inner
        if isinstance(inner, dict):
            return (
                inner.get("question", "")
                or inner.get("text", "")
                or inner.get("instruction", "")
            )
        return str(inner) if inner else ""

    @staticmethod
    def _extract_reference_answer(q_data: dict) -> str:
        """Extract reference/ground-truth answer from various formats."""
        inner = q_data.get("question", q_data)
        if isinstance(inner, dict):
            return (
                inner.get("answer", "")
                or inner.get("response", "")
                or inner.get("solution", "")
            )
        return ""


# ──────────────────────────────────────────────────────────────────────────────
# Parallel evaluation across multiple miner GPU slots
# ──────────────────────────────────────────────────────────────────────────────


async def evaluate_miners_parallel(
    miners: list[dict],
    questions: list[dict],
    judge_model_pool: list[str],
    parallel_manager: "ParallelMinerServerManager",  # type: ignore[name-defined]
    *,
    instruction_for_validity: str = "",
) -> list[EvaluationOutput]:
    """
    Evaluate all miners in parallel using N miner GPU slots.

    Architecture
    ────────────
    • One judge vLLM server (GPU 0) handles ALL concurrent judge calls.
      vLLM internally queues requests, so N simultaneous evaluations are safe.
    • Each miner slot (GPU 1-3) runs exactly one miner vLLM server at a time.
    • Miners are distributed across slots in a round-robin "wave" pattern:
        Wave 0: slots 0,1,2 evaluate miners 0,1,2 concurrently
        Wave 1: slots 0,1,2 evaluate miners 3,4,5 concurrently
        …and so on, until all miners are evaluated.
    • Within each wave, asyncio.gather() is used so the event loop can
      multiplex judge calls from all slots without blocking.

    Args:
        miners:              Ordered list of miner dicts (uid, model_name, …)
        questions:           Pre-sampled question list (same for ALL miners, fair)
        judge_model_pool:    Judge model names (same pool for all slots)
        parallel_manager:    ParallelMinerServerManager with judge + N miner slots
        instruction_for_validity: Optional Phase-1 instruction override

    Returns:
        List of EvaluationOutput in the same order as `miners`.
    """
    # Lazy import to avoid circular dependency at module level
    from .vllm_client import ParallelMinerServerManager  # noqa: PLC0415

    n_slots = parallel_manager.num_slots
    outputs: list[EvaluationOutput | None] = [None] * len(miners)

    # Split miners into waves of `n_slots`
    waves = [miners[i : i + n_slots] for i in range(0, len(miners), n_slots)]

    for wave_idx, wave in enumerate(waves):
        logger.info(
            f"[parallel-eval] Wave {wave_idx + 1}/{len(waves)}: "
            f"evaluating {len(wave)} miner(s) across {n_slots} GPU slot(s)"
        )

        # — Step 1: start each miner slot's server for the miners in this wave —
        for slot, miner in enumerate(wave):
            model_name = miner.get("model_name", "")
            revision = miner.get("revision", "main") or "main"
            parallel_manager.start_miner_slot(slot, model_name, revision)

        # — Step 2: build one orchestrator per slot and fire all evals concurrently —
        async def _eval_one(slot: int, miner: dict) -> tuple[int, EvaluationOutput]:
            orch = parallel_manager.make_orchestrator(slot, judge_model_pool)
            out = await orch.evaluate_miner(
                miner_uid=miner.get("uid", -1),
                model_name=miner.get("model_name", ""),
                questions=questions,
                instruction_for_validity=instruction_for_validity,
            )
            return slot, out

        tasks = [
            asyncio.create_task(_eval_one(slot, miner))
            for slot, miner in enumerate(wave)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # — Step 3: collect results and stop miner servers for this wave —
        base_idx = wave_idx * n_slots
        for slot, (miner, result) in enumerate(zip(wave, results)):
            if isinstance(result, Exception):
                logger.error(
                    f"[parallel-eval] Slot {slot} failed for miner "
                    f"uid={miner.get('uid')}: {result}"
                )
                err_out = EvaluationOutput(
                    miner_uid=miner.get("uid", -1),
                    model_name=miner.get("model_name", ""),
                    error=str(result),
                )
                outputs[base_idx + slot] = err_out
            else:
                _, out = result
                outputs[base_idx + slot] = out
            parallel_manager.stop_miner_slot(slot)

    # Filter out any None entries (should not happen, but be defensive)
    return [o for o in outputs if o is not None]
