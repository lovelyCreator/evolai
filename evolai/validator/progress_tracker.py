
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    thinking_loss: float
    model_revision: str
    validator_uid: int
    dataset_names: List[str] = field(default_factory=list)
    base_loss: float = 0.0
    sq_accuracy: float = 0.0


class MinerProgressState:

    def __init__(self, uid: int, history_epochs: int):
        self.uid = uid
        self.hotkey: str = ""
        self.coldkey: str = ""
        self.history: deque = deque(maxlen=history_epochs)
        self.eval_count: int = 0

    def record(self, rec: EpochRecord) -> None:
        self.history.append(rec)
        self.eval_count += 1

    def get_losses(self) -> List[float]:
        return [r.loss for r in self.history]

    def get_thinking_losses(self) -> List[float]:
        return [r.thinking_loss for r in self.history if r.thinking_loss > 0.0]

    def get_base_losses(self) -> List[float]:
        return [r.base_loss for r in self.history if r.base_loss > 0.0]

    def get_sq_accuracies(self) -> List[float]:
        return [r.sq_accuracy for r in self.history]

    def get_latest_revision(self) -> Optional[str]:
        return self.history[-1].model_revision if self.history else None

    def count_distinct_revisions(self) -> int:
        return len({r.model_revision for r in self.history if r.model_revision})

    def is_stagnant(self, min_epochs: int = 3) -> bool:
        if len(self.history) < min_epochs:
            return False
        recent = list(self.history)[-min_epochs:]
        revisions = {r.model_revision for r in recent if r.model_revision}
        return len(revisions) <= 1

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "hotkey": self.hotkey,
            "coldkey": self.coldkey,
            "eval_count": self.eval_count,
            "history": [
                {
                    "epoch": r.epoch,
                    "loss": r.loss,
                    "thinking_loss": r.thinking_loss,
                    "base_loss": r.base_loss,
                    "sq_accuracy": r.sq_accuracy,
                    "model_revision": r.model_revision,
                    "validator_uid": r.validator_uid,
                    "dataset_names": r.dataset_names,
                }
                for r in self.history
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, history_epochs: int) -> "MinerProgressState":
        state = cls(uid=data["uid"], history_epochs=history_epochs)
        state.hotkey = data.get("hotkey", "")
        state.coldkey = data.get("coldkey", "")
        state.eval_count = data.get("eval_count", 0)
        for r in data.get("history", []):
            state.history.append(EpochRecord(
                epoch=r["epoch"],
                loss=r["loss"],
                thinking_loss=r.get("thinking_loss", 0.0),
                model_revision=r.get("model_revision", ""),
                validator_uid=r.get("validator_uid", -1),
                dataset_names=r.get("dataset_names", []),
                base_loss=r.get("base_loss", 0.0),
                sq_accuracy=r.get("sq_accuracy", 0.0),
            ))
        return state


class ProgressTracker:

    def __init__(
        self,
        w_abs: float = 0.50,
        w_flow: float = 0.25,
        w_quality: float = 0.25,
        gamma: float = 1.0,
        ema_alpha: float = 0.10,
        history_epochs: int = 20,
        min_evaluations: int = 1,
        min_flow_epochs: int = 10,
        flow_eps: float = 1e-4,
        emission_lambda: float = 0.10,
        ema_short_rounds: int = 28,
        ema_long_rounds: int = 112,
        w_improvement: float = 0.3,
        w_proximity: float = 0.7,
        emission_floor: float = 0.2,
        emission_proximity_threshold: float = 0.95,
        archive_ttl_days: int = 7,
        storage_path: Optional[Path] = None,
    ):
        self.w_abs = w_abs
        self.w_flow = w_flow
        self.w_quality = w_quality
        self.gamma = gamma
        self.ema_alpha = ema_alpha
        self.history_epochs = history_epochs
        self.min_evaluations = min_evaluations
        self.min_flow_epochs = min_flow_epochs
        self.flow_eps = flow_eps
        self.emission_lambda = emission_lambda
        # Per-miner improvement+proximity scale
        self.ema_short_alpha = 2.0 / (ema_short_rounds + 1)
        self.ema_long_alpha  = 2.0 / (ema_long_rounds + 1)
        self.w_improvement = w_improvement
        self.w_proximity   = w_proximity
        self.emission_floor = emission_floor
        self.emission_proximity_threshold = emission_proximity_threshold
        self.archive_ttl_s = archive_ttl_days * 86400
        self.storage_path = storage_path or (
            Path.home() / ".evolai" / "validator" / "progress_tracker.json"
        )
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._miners: Dict[int, MinerProgressState] = {}
        self._coldkey_archive: Dict[str, dict] = {}
        # Legacy global-best fields kept for JSON back-compat; no longer used
        # for emission logic.
        self._best_ema_loss: float = float("inf")
        self._best_ema_loss_ts: float = 0.0
        # Staleness clock for the new fraction-based emission decay.
        self._last_good_miner_ts: float = 0.0
        self._load()


    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            if "version" in data and data.get("version", 0) >= 2:
                miners_data = data.get("miners", {})
                self._coldkey_archive = data.get("coldkey_archive", {})
                self._best_ema_loss = data.get("best_ema_loss", float("inf"))
                self._best_ema_loss_ts = data.get("best_ema_loss_ts", 0.0)
                self._last_good_miner_ts = data.get("last_good_miner_ts", 0.0)
            else:
                miners_data = data
                self._coldkey_archive = {}
                self._best_ema_loss = float("inf")
                self._best_ema_loss_ts = 0.0
                self._last_good_miner_ts = 0.0

            for uid_str, mdata in miners_data.items():
                uid = int(uid_str)
                self._miners[uid] = MinerProgressState.from_dict(mdata, self.history_epochs)

            self._prune_expired_archives()
            logger.info(
                f"Progress tracker loaded: {len(self._miners)} miners, "
                f"{len(self._coldkey_archive)} archived coldkeys"
            )
        except Exception as exc:
            logger.warning(f"Failed to load progress tracker: {exc}")

    def _save(self) -> None:
        tmp = self.storage_path.with_suffix(".tmp")
        try:
            data = {
                "version": 2,
                "miners": {str(uid): s.to_dict() for uid, s in self._miners.items()},
                "coldkey_archive": self._coldkey_archive,
                "best_ema_loss": self._best_ema_loss,
                "best_ema_loss_ts": self._best_ema_loss_ts,
                "last_good_miner_ts": self._last_good_miner_ts,
            }
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.storage_path)
        except Exception as exc:
            logger.warning(f"Failed to save progress tracker: {exc}")


    def sync_uid(self, uid: int, hotkey: str, coldkey: str = "") -> bool:
        state = self._miners.get(uid)
        if state is None:
            restored = self._restore_from_archive(coldkey, hotkey, uid) if (coldkey and hotkey) else None
            if restored:
                s = restored
                logger.info(
                    f"UID {uid}: restored history from coldkey archive "
                    f"({s.eval_count} epochs)"
                )
            else:
                s = MinerProgressState(uid, self.history_epochs)
            s.hotkey = hotkey
            s.coldkey = coldkey
            self._miners[uid] = s
            return False

        hotkey_changed = state.hotkey and state.hotkey != hotkey
        coldkey_changed = coldkey and state.coldkey and state.coldkey != coldkey

        if hotkey_changed or coldkey_changed:
            _reason = "hotkey" if hotkey_changed else "coldkey"
            if hotkey_changed and coldkey_changed:
                _reason = "hotkey+coldkey"
            logger.warning(
                f"UID {uid} replaced ({_reason}): "
                f"hk {state.hotkey[:16]}… → {hotkey[:16]}…"
                + (f", ck {state.coldkey[:16]}… → {coldkey[:16]}…" if coldkey_changed else "")
            )

            if state.coldkey and state.history:
                self._archive_miner(state)

            restored = self._restore_from_archive(coldkey, hotkey, uid) if (coldkey and hotkey) else None
            if restored:
                s = restored
                logger.info(
                    f"UID {uid}: restored history from coldkey archive "
                    f"({s.eval_count} epochs)"
                )
            else:
                s = MinerProgressState(uid, self.history_epochs)
            s.hotkey = hotkey
            s.coldkey = coldkey
            self._miners[uid] = s
            self._save()
            return True

        state.hotkey = hotkey
        if coldkey:
            state.coldkey = coldkey
        return False


    def _archive_miner(self, state: MinerProgressState) -> None:
        if not state.coldkey or not state.hotkey or not state.history:
            return
        archive_key = f"{state.coldkey}:{state.hotkey}"
        self._coldkey_archive[archive_key] = {
            "archived_at": time.time(),
            "eval_count": state.eval_count,
            "coldkey": state.coldkey,
            "hotkey": state.hotkey,
            "history": [
                {
                    "epoch": r.epoch,
                    "loss": r.loss,
                    "thinking_loss": r.thinking_loss,
                    "base_loss": r.base_loss,
                    "sq_accuracy": r.sq_accuracy,
                    "model_revision": r.model_revision,
                    "validator_uid": r.validator_uid,
                    "dataset_names": r.dataset_names,
                }
                for r in state.history
            ],
        }
        logger.info(
            f"Archived history for {state.coldkey[:16]}…:{state.hotkey[:12]}… "
            f"({state.eval_count} epochs)"
        )

    def _restore_from_archive(
        self, coldkey: str, hotkey: str, uid: int,
    ) -> Optional[MinerProgressState]:
        if not coldkey or not hotkey:
            return None

        archive_key = f"{coldkey}:{hotkey}"
        if archive_key not in self._coldkey_archive:
            return None

        archive = self._coldkey_archive[archive_key]
        age_s = time.time() - archive["archived_at"]
        if age_s > self.archive_ttl_s:
            del self._coldkey_archive[archive_key]
            logger.info(
                f"Archive for {coldkey[:16]}…:{hotkey[:12]}… expired "
                f"({age_s / 86400:.1f}d > {self.archive_ttl_s / 86400:.0f}d)"
            )
            return None

        state = MinerProgressState(uid, self.history_epochs)
        state.eval_count = archive["eval_count"]
        for r in archive["history"]:
            state.history.append(EpochRecord(
                epoch=r["epoch"],
                loss=r["loss"],
                thinking_loss=r.get("thinking_loss", 0.0),
                model_revision=r.get("model_revision", ""),
                validator_uid=r.get("validator_uid", -1),
                dataset_names=r.get("dataset_names", []),
                base_loss=r.get("base_loss", 0.0),
                sq_accuracy=r.get("sq_accuracy", 0.0),
            ))

        del self._coldkey_archive[archive_key]
        logger.info(
            f"Restored {state.eval_count} epochs from archive for "
            f"{coldkey[:16]}…:{hotkey[:12]}… → UID {uid} (age {age_s / 86400:.1f}d)"
        )
        return state

    def _prune_expired_archives(self) -> None:
        now = time.time()
        expired = [
            ck for ck, a in self._coldkey_archive.items()
            if now - a.get("archived_at", 0) > self.archive_ttl_s
        ]
        for ck in expired:
            del self._coldkey_archive[ck]
        if expired:
            logger.info(f"Pruned {len(expired)} expired coldkey archive(s)")


    def record(
        self,
        uid: int,
        epoch: int,
        loss: float,
        thinking_loss: float,
        model_revision: str,
        validator_uid: int,
        dataset_names: Optional[List[str]] = None,
        base_loss: float = 0.0,
        sq_accuracy: float = 0.0,
    ) -> None:
        if uid not in self._miners:
            self._miners[uid] = MinerProgressState(uid, self.history_epochs)
        self._miners[uid].record(EpochRecord(
            epoch=epoch,
            loss=loss,
            thinking_loss=thinking_loss,
            model_revision=model_revision,
            validator_uid=validator_uid,
            dataset_names=dataset_names or [],
            base_loss=base_loss,
            sq_accuracy=sq_accuracy,
        ))
        self._save()


    def compute_score(self, uid: int) -> float:
        """Legacy single-UID score without proximity context.

        Use get_all_scores(global_best_long_ema=...) for production scoring
        which includes the full per-miner scale.  This method retains
        neutral proximity (=1.0) for backward-compatible callers.
        """
        return self._score_with_context(uid, global_best_long_ema=0.0)

    def _compute_miner_scale(
        self,
        losses: List[float],
        global_best_long_ema: float = 0.0,
    ) -> float:
        """Return the per-miner scale in [0, 1] based on improvement and proximity."""
        if len(losses) < 2:
            # Not enough history — neutral scale (proximity only).
            return self.w_proximity * 1.0 + self.w_improvement * 0.0

        short_ema = _ema_alpha(losses, self.ema_short_alpha)
        long_ema  = _ema_alpha(losses, self.ema_long_alpha)

        # improvement > 0 only when loss is genuinely falling
        improvement = max(0.0, math.tanh(long_ema - short_ema))

        # proximity: ratio of global best long EMA to own long EMA;
        # clamped to [0, 1] so miners better than current global best
        # don't get inflated scale.
        if global_best_long_ema > 0.0 and long_ema > 0.0:
            proximity = min(1.0, global_best_long_ema / long_ema)
        else:
            proximity = 1.0  # no reference yet — neutral

        return self.w_improvement * improvement + self.w_proximity * proximity

    def get_all_scores(
        self,
        global_best_long_ema: float = 0.0,
    ) -> Dict[int, float]:
        """Return scores for all miners that meet min_evaluations.

        global_best_long_ema: the best (lowest) long-EMA loss across all
        miners in this track, computed fresh each round so it cannot drift
        from a stale stored value.
        """
        return {
            uid: self._score_with_context(uid, global_best_long_ema)
            for uid, state in self._miners.items()
            if state.eval_count >= self.min_evaluations
        }

    def _score_with_context(self, uid: int, global_best_long_ema: float) -> float:
        """compute_score with per-miner scale using the supplied global best."""
        state = self._miners.get(uid)
        if state is None or state.eval_count < self.min_evaluations:
            return 0.0

        losses = state.get_losses()
        if not losses:
            return 0.0

        ema_loss = _ema(losses, self.ema_alpha)
        if not math.isfinite(ema_loss):
            return 0.0
        absolute = math.exp(-self.gamma * ema_loss)

        flow = 0.0
        if len(losses) >= self.min_flow_epochs:
            ema_series = _ema_series(losses, self.ema_alpha)
            deltas = [prev - curr for prev, curr in zip(ema_series[:-1], ema_series[1:])]
            if deltas:
                decay = 1.0 - self.ema_alpha
                n = len(deltas)
                weights = [decay ** (n - 1 - i) for i in range(n)]
                mu_delta, sigma_delta = _weighted_mean_std(deltas, weights)
                sharpe = mu_delta / (sigma_delta + self.flow_eps)
                flow = max(0.0, math.tanh(sharpe))

        think_losses = state.get_thinking_losses()
        base_losses = state.get_base_losses()
        sq_accs = state.get_sq_accuracies()
        ema_base = _ema(base_losses, self.ema_alpha) if base_losses else ema_loss
        if think_losses and math.isfinite(ema_base) and ema_base > 1e-8:
            ema_think = _ema(think_losses, self.ema_alpha)
            if math.isfinite(ema_think):
                think_gain = (ema_base - ema_think) / ema_base
                sq_acc_ema = _ema(sq_accs, self.ema_alpha) if sq_accs else 0.0
                quality = max(0.0, min(sq_acc_ema + think_gain, 2.0))
            else:
                quality = 0.0
        else:
            quality = 0.0

        raw_score = self.w_abs * absolute + self.w_flow * flow + self.w_quality * quality
        miner_scale = self._compute_miner_scale(losses, global_best_long_ema)
        return raw_score * miner_scale

    def get_miner_state(self, uid: int) -> Optional[MinerProgressState]:
        return self._miners.get(uid)

    def get_latest_loss(self, uid: int) -> Optional[float]:
        state = self._miners.get(uid)
        if state and state.history:
            return state.history[-1].loss
        return None


    def compute_global_best_long_ema(self, uids: Optional[List[int]] = None) -> float:
        """Return the best (lowest finite) long-EMA loss across the given UIDs.

        Computed fresh each call from live history — never stored — so it
        cannot drift or be gamed by stale values.
        """
        best = float("inf")
        candidates = uids if uids is not None else list(self._miners.keys())
        for uid in candidates:
            state = self._miners.get(uid)
            if state is None or state.eval_count < self.min_evaluations:
                continue
            losses = state.get_losses()
            if len(losses) < 2:
                continue
            long_ema = _ema_alpha(losses, self.ema_long_alpha)
            if math.isfinite(long_ema) and long_ema < best:
                best = long_ema
        return best if math.isfinite(best) else 0.0

    def update_global_best(self) -> bool:
        """Legacy method — kept for call-site compatibility.

        Updates the stored _best_ema_loss for JSON persistence and logging,
        but emission logic no longer depends on it.
        """
        improved = False
        for uid, state in self._miners.items():
            if state.eval_count < self.min_evaluations:
                continue
            losses = state.get_losses()
            if not losses:
                continue
            ema_loss = _ema(losses, self.ema_alpha)
            if ema_loss < self._best_ema_loss:
                self._best_ema_loss = ema_loss
                self._best_ema_loss_ts = time.time()
                improved = True
                logger.info(
                    f"New global best EMA loss: {ema_loss:.6f} (UID {uid})"
                )
        if improved:
            self._save()
        return improved

    def compute_emission_scale(self, uids: Optional[List[int]] = None) -> float:
        """Fraction-based global emission scale.

        Returns 1.0 if any miner in `uids` is currently improving
        (short EMA < long EMA) OR has proximity >= emission_proximity_threshold
        (i.e. is genuinely near the frontier, even if loss has plateaued).
        If neither condition holds, decays exponentially from the last time a
        good miner was seen, with a floor of `self.emission_floor` to prevent
        subnet death.

        This replaces the old staleness-based get_emission_scale().
        """
        candidates = uids if uids is not None else list(self._miners.keys())

        # Compute the global best long-EMA once for proximity checks.
        global_best = self.compute_global_best_long_ema(uids=candidates)

        any_good = False
        for uid in candidates:
            state = self._miners.get(uid)
            if state is None or state.eval_count < self.min_evaluations:
                continue
            losses = state.get_losses()
            if len(losses) < 2:
                continue
            short_ema = _ema_alpha(losses, self.ema_short_alpha)
            long_ema  = _ema_alpha(losses, self.ema_long_alpha)
            # Condition 1: still actively improving
            if short_ema < long_ema:
                any_good = True
                break
            # Condition 2: plateaued but genuinely near the frontier
            if global_best > 0.0 and math.isfinite(long_ema):
                proximity = min(1.0, global_best / long_ema)
                if proximity >= self.emission_proximity_threshold:
                    any_good = True
                    break

        if any_good:
            self._last_good_miner_ts = time.time()
            self._save()
            return 1.0

        # No improving miner — decay from last good timestamp
        if self._last_good_miner_ts == 0.0:
            # Never seen a good miner yet (cold start) — give full scale
            return 1.0
        staleness_days = (time.time() - self._last_good_miner_ts) / 86400.0
        scale = math.exp(-self.emission_lambda * staleness_days)
        return max(self.emission_floor, min(1.0, scale))

    # ---- Legacy shims for backward compatibility ----

    def is_emission_active(self) -> bool:
        return True

    def get_emission_scale(self) -> float:
        """Deprecated — use compute_emission_scale() instead."""
        return self.compute_emission_scale()

    def get_best_ema_loss(self) -> float:
        return self._best_ema_loss

    def get_staleness_days(self) -> float:
        if self._last_good_miner_ts == 0.0:
            return 0.0
        return (time.time() - self._last_good_miner_ts) / 86400.0


def _ema(values: List[float], alpha: float) -> float:
    ema = values[0]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
    return ema


def _ema_alpha(values: List[float], alpha: float) -> float:
    """Same as _ema but accepts an explicit alpha (used for short/long windows)."""
    return _ema(values, alpha)


def _ema_series(values: List[float], alpha: float) -> List[float]:
    if not values:
        return []
    ema = values[0]
    out = [ema]
    for v in values[1:]:
        ema = alpha * v + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _weighted_mean_std(values: List[float], weights: List[float]) -> tuple[float, float]:
    if not values or not weights or len(values) != len(weights):
        return 0.0, 0.0
    total = sum(weights)
    if total <= 0.0:
        return 0.0, 0.0
    mean = sum(v * w for v, w in zip(values, weights)) / total
    variance = sum(w * (v - mean) * (v - mean) for v, w in zip(values, weights)) / total
    return mean, math.sqrt(max(0.0, variance))
