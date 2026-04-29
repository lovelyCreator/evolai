from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    GPU_TOTAL_VRAM_GB,
    GPU_RESERVE_VRAM_GB,
    JUDGE_VRAM_FRACTION,
    MINER_VRAM_FRACTION,
    GPU_OOM_COOLDOWN_BASE_S,
    GPU_OOM_COOLDOWN_MULTIPLIER,
    GPU_OOM_COOLDOWN_CAP_S,
    GPU_FAILURE_WINDOW_DECAY_S,
    GPU_VRAM_WARNING_THRESHOLD,
    GPU_VRAM_CRITICAL_THRESHOLD,
    VLLM_GPU_MEMORY_UTILIZATION,
)

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:

    index: int
    name: str
    total_vram_gb: float
    free_vram_gb: float = 0.0
    used_vram_gb: float = 0.0


def detect_gpus() -> list[GPUInfo]:

    try:
        import torch

        if torch.cuda.is_available():
            gpus: list[GPUInfo] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024**3)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_gb = free_mem / (1024**3)
                used_gb = total_gb - free_gb
                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        total_vram_gb=round(total_gb, 2),
                        free_vram_gb=round(free_gb, 2),
                        used_vram_gb=round(used_gb, 2),
                    )
                )
            if gpus:
                logger.info(f"[gpu-detect] Found {len(gpus)} GPU(s) via torch.cuda")
                return gpus
    except ImportError:
        pass
    except Exception as exc:
        logger.warning(f"[gpu-detect] torch.cuda detection failed: {exc}")


    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(
                        GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            total_vram_gb=round(float(parts[2]) / 1024, 2),
                            free_vram_gb=round(float(parts[3]) / 1024, 2),
                            used_vram_gb=round(float(parts[4]) / 1024, 2),
                        )
                    )
            if gpus:
                logger.info(f"[gpu-detect] Found {len(gpus)} GPU(s) via nvidia-smi")
                return gpus
    except FileNotFoundError:
        logger.warning("[gpu-detect] nvidia-smi not found")
    except Exception as exc:
        logger.warning(f"[gpu-detect] nvidia-smi detection failed: {exc}")

    logger.warning("[gpu-detect] No GPUs detected")
    return []


def get_live_vram_usage(gpu_index: int = 0) -> dict[str, float]:
    try:
        import torch

        if torch.cuda.is_available() and gpu_index < torch.cuda.device_count():
            free_mem, total_mem = torch.cuda.mem_get_info(gpu_index)
            total_gb = total_mem / (1024**3)
            free_gb = free_mem / (1024**3)
            used_gb = total_gb - free_gb
            return {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "utilization": round(used_gb / max(total_gb, 0.001), 4),
            }
    except Exception:
        pass


    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [float(p.strip()) for p in result.stdout.strip().split(",")]
            total_gb = parts[0] / 1024
            used_gb = parts[1] / 1024
            free_gb = parts[2] / 1024
            return {
                "total_gb": round(total_gb, 2),
                "used_gb": round(used_gb, 2),
                "free_gb": round(free_gb, 2),
                "utilization": round(used_gb / max(total_gb, 0.001), 4),
            }
    except Exception:
        pass

    return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "utilization": 0.0}


@dataclass
class VRAMBudget:

    gpu_index: int
    total_vram_gb: float
    reserved_gb: float = GPU_RESERVE_VRAM_GB
    allocations: dict[str, float] = field(default_factory=dict)

    @property
    def allocated_gb(self) -> float:
        return sum(self.allocations.values())

    @property
    def available_gb(self) -> float:
        return self.total_vram_gb - self.reserved_gb - self.allocated_gb

    @property
    def utilization(self) -> float:
        used = self.allocated_gb + self.reserved_gb
        return used / max(self.total_vram_gb, 0.001)

    def can_allocate(self, gb: float) -> bool:
        return self.available_gb >= gb

    def allocate(self, role: str, gb: float) -> bool:
        if not self.can_allocate(gb):
            logger.warning(
                f"[vram-budget] Cannot allocate {gb:.2f} GB for {role} on GPU {self.gpu_index}. "
                f"Available: {self.available_gb:.2f} GB"
            )
            return False
        self.allocations[role] = gb
        logger.info(
            f"[vram-budget] Allocated {gb:.2f} GB for {role} on GPU {self.gpu_index}. "
            f"Remaining: {self.available_gb:.2f} GB"
        )
        return True

    def release(self, role: str) -> float:
        released = self.allocations.pop(role, 0.0)
        if released > 0:
            logger.info(
                f"[vram-budget] Released {released:.2f} GB from {role} on GPU {self.gpu_index}. "
                f"Available: {self.available_gb:.2f} GB"
            )
        return released

    def summary(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "total_gb": self.total_vram_gb,
            "reserved_gb": self.reserved_gb,
            "allocated_gb": round(self.allocated_gb, 2),
            "available_gb": round(self.available_gb, 2),
            "utilization": round(self.utilization, 4),
            "allocations": dict(self.allocations),
        }


@dataclass
class GPUSlot:

    gpu_index: int
    last_used_at: float = 0.0
    consecutive_errors: int = 0
    last_error_at: float = 0.0
    cooldown_until: float = 0.0
    total_oom_events: int = 0

    @property
    def is_in_cooldown(self) -> bool:
        now = time.time()
        if now >= self.cooldown_until:


            if self.cooldown_until > 0:
                self._half_open_reset()
            return False
        return True

    def _half_open_reset(self) -> None:
        self.consecutive_errors = 0
        self.cooldown_until = 0.0
        logger.info(f"[gpu-slot] GPU {self.gpu_index} cooldown expired, resetting error counters")

    def record_success(self) -> None:
        self.consecutive_errors = 0
        self.last_error_at = 0.0
        self.cooldown_until = 0.0
        self.last_used_at = time.time()

    def record_failure(self, is_oom: bool = False) -> None:
        now = time.time()


        if (
            self.last_error_at > 0
            and (now - self.last_error_at) > GPU_FAILURE_WINDOW_DECAY_S
        ):
            self.consecutive_errors = 0
            logger.info(
                f"[gpu-slot] GPU {self.gpu_index} error window decayed, resetting counter"
            )

        self.consecutive_errors += 1
        self.last_error_at = now

        if is_oom:
            self.total_oom_events += 1


        delay = GPU_OOM_COOLDOWN_BASE_S * (
            GPU_OOM_COOLDOWN_MULTIPLIER ** (self.consecutive_errors - 1)
        )
        delay = min(delay, GPU_OOM_COOLDOWN_CAP_S)


        new_cooldown = now + delay
        if new_cooldown > self.cooldown_until:
            self.cooldown_until = new_cooldown

        logger.warning(
            f"[gpu-slot] GPU {self.gpu_index} failure #{self.consecutive_errors} "
            f"(oom={is_oom}). Cooldown: {delay:.0f}s until "
            f"{time.strftime('%H:%M:%S', time.localtime(self.cooldown_until))}"
        )

    def summary(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "in_cooldown": self.is_in_cooldown,
            "consecutive_errors": self.consecutive_errors,
            "total_oom_events": self.total_oom_events,
            "cooldown_remaining_s": max(0, self.cooldown_until - time.time()),
            "last_used_at": self.last_used_at,
        }


def select_gpu_for_role(
    role: str,
    required_vram_gb: float,
    slots: list[GPUSlot],
    budgets: list[VRAMBudget],
) -> int | None:
    candidates: list[tuple[int, float, float]] = []

    for slot, budget in zip(slots, budgets):
        if slot.is_in_cooldown:
            continue
        if not budget.can_allocate(required_vram_gb):
            continue
        candidates.append((slot.gpu_index, budget.utilization, slot.last_used_at))

    if not candidates:
        return None


    candidates.sort(key=lambda c: (c[1], c[2]))
    return candidates[0][0]


class GPUManager:

    def __init__(self) -> None:
        self._gpus: list[GPUInfo] = []
        self._slots: list[GPUSlot] = []
        self._budgets: list[VRAMBudget] = []
        self._initialized: bool = False
        self._created_at: float = time.time()

    def initialize(self) -> None:
        if self._initialized:
            return

        self._gpus = detect_gpus()

        if not self._gpus:
            logger.warning("[gpu-mgr] No GPUs detected — running in CPU-only mode")
            self._initialized = True
            return

        for gpu in self._gpus:
            total = gpu.total_vram_gb
            if GPU_TOTAL_VRAM_GB > 0:
                total = GPU_TOTAL_VRAM_GB

            self._slots.append(GPUSlot(gpu_index=gpu.index))
            self._budgets.append(
                VRAMBudget(
                    gpu_index=gpu.index,
                    total_vram_gb=total,
                    reserved_gb=GPU_RESERVE_VRAM_GB,
                )
            )

        self._initialized = True
        logger.info(
            f"[gpu-mgr] Initialized with {len(self._gpus)} GPU(s): "
            + ", ".join(
                f"GPU {g.index}: {g.name} ({g.total_vram_gb} GB)" for g in self._gpus
            )
        )

    @property
    def gpu_count(self) -> int:
        return len(self._gpus)

    @property
    def has_gpus(self) -> bool:
        return len(self._gpus) > 0

    def get_slot(self, gpu_index: int) -> GPUSlot | None:
        for s in self._slots:
            if s.gpu_index == gpu_index:
                return s
        return None

    def get_budget(self, gpu_index: int) -> VRAMBudget | None:
        for b in self._budgets:
            if b.gpu_index == gpu_index:
                return b
        return None

    def allocate_role(
        self, role: str, gpu_index: int | None = None
    ) -> tuple[int, float] | None:
        if not self.has_gpus:
            logger.warning("[gpu-mgr] No GPUs available for allocation")
            return None

        fraction = JUDGE_VRAM_FRACTION if role == "judge" else MINER_VRAM_FRACTION

        if gpu_index is None:

            for budget in self._budgets:
                required_gb = budget.total_vram_gb * fraction
                gpu_index = select_gpu_for_role(
                    role, required_gb, self._slots, self._budgets
                )
                if gpu_index is not None:
                    break

        if gpu_index is None:
            logger.error(f"[gpu-mgr] No GPU with sufficient VRAM for {role}")
            return None

        budget = self.get_budget(gpu_index)
        if budget is None:
            return None

        required_gb = budget.total_vram_gb * fraction
        if budget.allocate(role, required_gb):
            return (gpu_index, required_gb)

        return None

    def release_role(self, role: str) -> None:
        for budget in self._budgets:
            budget.release(role)

    def record_success(self, gpu_index: int) -> None:
        slot = self.get_slot(gpu_index)
        if slot:
            slot.record_success()

    def record_failure(self, gpu_index: int, is_oom: bool = False) -> None:
        slot = self.get_slot(gpu_index)
        if slot:
            slot.record_failure(is_oom=is_oom)

    def check_vram_warnings(self) -> list[dict]:
        warnings: list[dict] = []
        for gpu in self._gpus:
            live = get_live_vram_usage(gpu.index)
            util = live.get("utilization", 0.0)
            if util >= GPU_VRAM_CRITICAL_THRESHOLD:
                warnings.append(
                    {
                        "gpu_index": gpu.index,
                        "level": "critical",
                        "utilization": util,
                        "message": f"GPU {gpu.index} VRAM critical: {util:.1%}",
                    }
                )
            elif util >= GPU_VRAM_WARNING_THRESHOLD:
                warnings.append(
                    {
                        "gpu_index": gpu.index,
                        "level": "warning",
                        "utilization": util,
                        "message": f"GPU {gpu.index} VRAM high: {util:.1%}",
                    }
                )
        return warnings

    def calculate_model_vram_estimate(
        self, model_name: str, gpu_memory_utilization: float | None = None
    ) -> float:
        gpu_util = gpu_memory_utilization or VLLM_GPU_MEMORY_UTILIZATION
        model_lower = model_name.lower()


        import re

        match = re.search(r"(\d+\.?\d*)b", model_lower)
        if match:
            params_b = float(match.group(1))

            vram_gb = params_b * 2 / 1024 * 1.2

            moe_match = re.search(r"a(\d+\.?\d*)b", model_lower)
            if moe_match:
                active_params = float(moe_match.group(1))
                vram_gb = active_params * 2 / 1024 * 1.5
            logger.info(
                f"[gpu-mgr] VRAM estimate for {model_name}: {vram_gb:.1f} GB "
                f"(params={params_b}B)"
            )
            return round(vram_gb, 2)


        if self._budgets:
            return round(self._budgets[0].total_vram_gb * gpu_util, 2)
        return 10.0

    def summary(self) -> dict:
        return {
            "initialized": self._initialized,
            "gpu_count": self.gpu_count,
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "total_vram_gb": g.total_vram_gb,
                }
                for g in self._gpus
            ],
            "slots": [s.summary() for s in self._slots],
            "budgets": [b.summary() for b in self._budgets],
            "vram_warnings": self.check_vram_warnings(),
        }


_gpu_manager: GPUManager | None = None


def get_gpu_manager() -> GPUManager:
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager
