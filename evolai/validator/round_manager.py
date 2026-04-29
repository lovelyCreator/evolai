"""
Evaluation Round Manager — Round-Based Judge Sampling & Model Lifecycle

Orchestrates the full lifecycle of an evaluation round:
  1. Round start → sample judge model from pool → start judge vLLM server
  2. Per miner   → start miner vLLM server → evaluate → stop miner → free GPU
  3. Round end   → stop judge server → release ALL GPU memory → full cleanup
  4. Next round  → sample a DIFFERENT judge → repeat

Key constraint: after each evaluation round the judge server MUST be stopped
to free GPU memory. Miner servers MUST be stopped after each miner evaluation
to make space for the next miner model.

Judge sampling uses round-robin (no repetition until pool exhausted), then
reshuffles. This ensures every judge model gets equal usage across rounds.

Design basis: VALIDATOR_INTERVIEW_EVALUATION.md §15.9 Round-Based Lifecycle
"""
from __future__ import annotations

import asyncio
import gc
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import JUDGE_MODELS
from .lifecycle import emit_event

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Judge Sampler (round-robin with shuffle)
# ──────────────────────────────────────────────────────────────────────────────


class JudgeSampler:
    """
    Samples judge models from the pool without immediate repetition.

    Uses round-robin: shuffles the pool, iterates through it one by one.
    Once the pool is exhausted, reshuffles for the next cycle.
    Guarantees no consecutive duplicates across cycles.

    Usage:
        sampler = JudgeSampler(["model-A", "model-B", "model-C"])
        judge_1 = sampler.next()  # e.g. "model-B"
        judge_2 = sampler.next()  # e.g. "model-A"  (never == judge_1 unless pool size 1)
    """

    def __init__(self, pool: list[str] | None = None) -> None:
        self._pool = list(pool or JUDGE_MODELS)
        if not self._pool:
            raise ValueError("Judge model pool cannot be empty")
        self._queue: list[str] = []
        self._last_used: str | None = None
        self._round_count: int = 0

    def next(self) -> str:
        """
        Return the next judge model to use.
        Guarantees no immediate repetition (when pool > 1).
        """
        if not self._queue:
            self._reshuffle()

        model = self._queue.pop(0)
        self._last_used = model
        return model

    def _reshuffle(self) -> None:
        """Reshuffle the pool for a new cycle, avoiding immediate repeat."""
        self._queue = list(self._pool)
        random.shuffle(self._queue)

        # If pool has more than one model and the first item is the same
        # as the last used, rotate it to the end to avoid repetition
        if (
            len(self._queue) > 1
            and self._last_used is not None
            and self._queue[0] == self._last_used
        ):
            self._queue.append(self._queue.pop(0))

        self._round_count += 1

    @property
    def pool_size(self) -> int:
        return len(self._pool)

    @property
    def remaining_in_cycle(self) -> int:
        return len(self._queue)

    @property
    def round_count(self) -> int:
        return self._round_count

    @property
    def last_used(self) -> str | None:
        return self._last_used


# ──────────────────────────────────────────────────────────────────────────────
# Miner entry
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MinerEntry:
    """A miner to evaluate in a round."""

    uid: int
    model_name: str
    revision: str = "main"
    hotkey: str = ""
    track: str = "transformer"


# ──────────────────────────────────────────────────────────────────────────────
# Round result
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RoundResult:
    """Result of a complete evaluation round."""

    round_number: int = 0
    judge_model: str = ""
    miners_evaluated: int = 0
    miners_failed: int = 0
    total_elapsed_s: float = 0.0
    miner_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Round Manager
# ──────────────────────────────────────────────────────────────────────────────


class EvaluationRoundManager:
    """
    Manages the full lifecycle of evaluation rounds.

    Each round:
      1. Sample a judge model (round-robin from pool)
      2. Start the judge vLLM server
      3. For each miner:
         a. Start miner vLLM server
         b. Run evaluation via EvaluationOrchestrator
         c. Stop miner server → free GPU
      4. Stop judge server → free ALL GPU memory
      5. Full GPU cleanup (torch.cuda.empty_cache + gc)

    Usage:
        mgr = EvaluationRoundManager()
        result = await mgr.run_round(miners=[...], questions=[...])
        # GPU is fully free after run_round() returns
    """

    def __init__(
        self,
        judge_pool: list[str] | None = None,
        miner_backend: str = "vllm",
        gpu_index: int = 0,
    ) -> None:
        self._judge_sampler = JudgeSampler(judge_pool)
        self._miner_backend = miner_backend
        self._gpu_index = gpu_index
        self._server_mgr = None  # Created per-round
        self._orchestrator = None  # Created per-round
        self._round_count = 0
        self._current_judge: str | None = None

    def _ensure_server_manager(self):
        """Lazily create VLLMServerManager."""
        if self._server_mgr is None:
            from .vllm_client import VLLMServerManager

            self._server_mgr = VLLMServerManager(gpu_index=self._gpu_index)

    def _create_orchestrator(self, judge_model: str):
        """Create a fresh orchestrator for this round's judge."""
        from .orchestrator import EvaluationOrchestrator

        self._orchestrator = EvaluationOrchestrator(
            judge_model_pool=[judge_model],
            miner_backend=self._miner_backend,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    async def run_round(
        self,
        miners: list[MinerEntry],
        questions: list[dict],
        *,
        round_number: int | None = None,
    ) -> RoundResult:
        """
        Execute a complete evaluation round.

        1. Sample judge → start judge server
        2. Evaluate each miner (start → eval → stop miner server)
        3. Stop judge server → full GPU cleanup

        Args:
            miners: List of miners to evaluate this round.
            questions: Sampled questions for evaluation.
            round_number: Optional round index (auto-incremented if None).

        Returns:
            RoundResult with per-miner scores and metadata.
        """
        self._round_count += 1
        rnum = round_number if round_number is not None else self._round_count
        t0 = time.time()
        result = RoundResult(round_number=rnum)

        # ── 1. Sample judge model ────────────────────────────────────
        judge_model = self._judge_sampler.next()
        self._current_judge = judge_model
        result.judge_model = judge_model
        logger.info(
            f"[round-mgr] === Round {rnum} === Judge: {judge_model} | "
            f"Miners: {len(miners)}"
        )
        emit_event("round_start", metadata={
            "round": rnum,
            "judge_model": judge_model,
            "num_miners": len(miners),
        })

        # ── 2. Start judge server ────────────────────────────────────
        self._ensure_server_manager()
        try:
            self._server_mgr.start_judge(judge_model)
        except Exception as exc:
            error_msg = f"Failed to start judge server: {exc}"
            logger.error(f"[round-mgr] {error_msg}")
            result.errors.append(error_msg)
            result.total_elapsed_s = round(time.time() - t0, 2)
            self._cleanup_round()
            return result

        # ── 3. Create orchestrator for this round's judge ────────────
        self._create_orchestrator(judge_model)

        # ── 4. Evaluate each miner ──────────────────────────────────
        for miner in miners:
            miner_result = await self._evaluate_single_miner(
                miner, questions, rnum,
            )
            result.miner_results.append(miner_result)
            if miner_result.get("error"):
                result.miners_failed += 1
            else:
                result.miners_evaluated += 1

        # ── 5. Stop judge server + full GPU cleanup ──────────────────
        logger.info(f"[round-mgr] Round {rnum} complete. Cleaning up...")
        self._cleanup_round()

        result.total_elapsed_s = round(time.time() - t0, 2)
        emit_event("round_end", metadata={
            "round": rnum,
            "judge_model": judge_model,
            "miners_evaluated": result.miners_evaluated,
            "miners_failed": result.miners_failed,
            "total_elapsed_s": result.total_elapsed_s,
        })
        logger.info(
            f"[round-mgr] Round {rnum} done: {result.miners_evaluated} evaluated, "
            f"{result.miners_failed} failed, {result.total_elapsed_s:.1f}s total"
        )

        return result

    async def run_multiple_rounds(
        self,
        miners: list[MinerEntry],
        questions_per_round: list[list[dict]],
        *,
        num_rounds: int | None = None,
    ) -> list[RoundResult]:
        """
        Run multiple evaluation rounds sequentially.
        Each round samples a different judge model.

        Args:
            miners: Miners to evaluate each round.
            questions_per_round: Pre-sampled questions for each round.
            num_rounds: Number of rounds (defaults to len(questions_per_round)).

        Returns:
            List of RoundResult for each round.
        """
        n = num_rounds or len(questions_per_round)
        results: list[RoundResult] = []

        for i in range(n):
            questions = questions_per_round[i] if i < len(questions_per_round) else []
            result = await self.run_round(miners, questions, round_number=i + 1)
            results.append(result)

        return results

    # ──────────────────────────────────────────────────────────────────
    # Per-miner evaluation
    # ──────────────────────────────────────────────────────────────────

    async def _evaluate_single_miner(
        self,
        miner: MinerEntry,
        questions: list[dict],
        round_number: int,
    ) -> dict:
        """
        Evaluate a single miner within a round.

        Lifecycle:
          1. Start miner vLLM server
          2. Run orchestrator evaluation
          3. Stop miner server → free GPU for next miner
        """
        miner_t0 = time.time()
        miner_result: dict[str, Any] = {
            "uid": miner.uid,
            "model_name": miner.model_name,
            "revision": miner.revision,
            "round": round_number,
            "judge_model": self._current_judge,
        }

        logger.info(
            f"[round-mgr] Evaluating miner uid={miner.uid} "
            f"model={miner.model_name}@{miner.revision}"
        )
        emit_event("miner_round_start", miner_uid=miner.uid, metadata={
            "round": round_number,
            "model_name": miner.model_name,
            "judge_model": self._current_judge,
        })

        try:
            # 1. Start miner server
            self._server_mgr.start_miner(miner.model_name, miner.revision)

            # 2. Configure orchestrator for this miner
            self._orchestrator.miner_model_id = miner.model_name

            # 3. Run evaluation
            output = await self._orchestrator.evaluate_miner(
                miner_uid=miner.uid,
                model_name=miner.model_name,
                questions=questions,
            )

            miner_result["final_score"] = output.final_score
            miner_result["normalized_score"] = output.normalized_score
            miner_result["validity_passed"] = output.validity_passed
            miner_result["error"] = output.error
            miner_result["elapsed_s"] = round(time.time() - miner_t0, 2)
            miner_result["usage"] = output.usage

        except Exception as exc:
            logger.error(
                f"[round-mgr] Miner uid={miner.uid} evaluation failed: {exc}",
                exc_info=True,
            )
            miner_result["error"] = str(exc)
            miner_result["final_score"] = 0.0
            miner_result["normalized_score"] = 0.0
            miner_result["elapsed_s"] = round(time.time() - miner_t0, 2)

        finally:
            # 3. ALWAYS stop miner server to free GPU for next miner
            try:
                self._server_mgr.stop_miner()
                logger.info(
                    f"[round-mgr] Miner server stopped for uid={miner.uid}, "
                    "GPU freed for next miner"
                )
            except Exception as stop_exc:
                logger.warning(
                    f"[round-mgr] Failed to stop miner server: {stop_exc}"
                )

            # Force GPU memory cleanup between miners
            self._inter_miner_cleanup()

        emit_event("miner_round_end", miner_uid=miner.uid, metadata={
            "round": round_number,
            "final_score": miner_result.get("final_score", 0.0),
            "elapsed_s": miner_result.get("elapsed_s", 0.0),
            "error": miner_result.get("error"),
        })

        return miner_result

    # ──────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────

    def _cleanup_round(self) -> None:
        """
        Full cleanup at end of round.
        Stops ALL vLLM servers and releases ALL GPU memory.
        """
        # Stop both servers
        if self._server_mgr is not None:
            try:
                self._server_mgr.stop_all()
            except Exception as exc:
                logger.warning(f"[round-mgr] Error stopping servers: {exc}")

        # Clear orchestrator reference
        self._orchestrator = None
        self._current_judge = None

        # Full GPU memory cleanup
        self._full_gpu_cleanup()

        logger.info("[round-mgr] Round cleanup complete — GPU fully released")

    def _inter_miner_cleanup(self) -> None:
        """
        Light cleanup between miners within a round.
        Frees miner GPU memory but keeps judge server warm.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

    def _full_gpu_cleanup(self) -> None:
        """
        Aggressive GPU cleanup — frees ALL GPU memory.
        Called at end of round when both servers are stopped.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Reset peak memory stats for next round
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
        except ImportError:
            pass

        # Aggressive garbage collection
        gc.collect()

        # Release VRAM budget tracking
        try:
            from .gpu_manager import get_gpu_manager

            mgr = get_gpu_manager()
            mgr.release_role("judge")
            mgr.release_role("miner")
        except Exception:
            pass

        logger.info("[round-mgr] Full GPU cleanup done")

    # ──────────────────────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────────────────────

    @property
    def round_count(self) -> int:
        return self._round_count

    @property
    def current_judge(self) -> str | None:
        return self._current_judge

    def summary(self) -> dict:
        return {
            "rounds_completed": self._round_count,
            "current_judge": self._current_judge,
            "judge_sampler": {
                "pool_size": self._judge_sampler.pool_size,
                "remaining_in_cycle": self._judge_sampler.remaining_in_cycle,
                "last_used": self._judge_sampler.last_used,
            },
        }
