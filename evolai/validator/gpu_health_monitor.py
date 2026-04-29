from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .config import (
    GPU_HEALTH_POLL_INTERVAL_S,
    GPU_STARTUP_GRACE_S,
    GPU_MAX_RESTARTS_PER_HOUR,
    GPU_RESTART_COOLDOWN_S,
    GPU_VRAM_WARNING_THRESHOLD,
    GPU_VRAM_CRITICAL_THRESHOLD,
    VLLM_JUDGE_PORT,
    VLLM_MINER_PORT,
)
from .gpu_manager import GPUManager, get_gpu_manager, get_live_vram_usage

logger = logging.getLogger(__name__)


@dataclass
class ServerHealthStatus:

    role: str
    port: int
    is_running: bool
    response_time_ms: float = 0.0
    error: str | None = None
    model_name: str | None = None


@dataclass
class GPUHealthStatus:

    gpu_index: int
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    vram_free_gb: float = 0.0
    vram_utilization: float = 0.0
    level: str = "healthy"
    servers: list[ServerHealthStatus] = field(default_factory=list)


class RestartTracker:

    def __init__(self, max_per_hour: int = GPU_MAX_RESTARTS_PER_HOUR):
        self._max_per_hour = max_per_hour
        self._timestamps: dict[str, list[float]] = {}
        self._last_restart: dict[str, float] = {}

    def can_restart(self, role: str) -> bool:
        now = time.time()


        last = self._last_restart.get(role, 0.0)
        if now - last < GPU_RESTART_COOLDOWN_S:
            logger.info(
                f"[restart-tracker] {role} restart blocked: "
                f"cooldown ({GPU_RESTART_COOLDOWN_S - (now - last):.0f}s remaining)"
            )
            return False


        timestamps = self._timestamps.get(role, [])

        cutoff = now - 3600
        timestamps = [t for t in timestamps if t > cutoff]
        self._timestamps[role] = timestamps

        if len(timestamps) >= self._max_per_hour:
            logger.warning(
                f"[restart-tracker] {role} restart blocked: "
                f"rate limit ({self._max_per_hour}/hour exceeded)"
            )
            return False

        return True

    def record_restart(self, role: str) -> None:
        now = time.time()
        self._last_restart[role] = now
        if role not in self._timestamps:
            self._timestamps[role] = []
        self._timestamps[role].append(now)

    def summary(self) -> dict:
        now = time.time()
        result: dict = {}
        for role in set(list(self._timestamps.keys()) + list(self._last_restart.keys())):
            timestamps = self._timestamps.get(role, [])
            recent = [t for t in timestamps if t > now - 3600]
            result[role] = {
                "restarts_last_hour": len(recent),
                "max_per_hour": self._max_per_hour,
                "can_restart": self.can_restart(role),
                "cooldown_remaining_s": max(
                    0, GPU_RESTART_COOLDOWN_S - (now - self._last_restart.get(role, 0))
                ),
            }
        return result


class GPUHealthMonitor:

    def __init__(
        self,
        gpu_manager: GPUManager | None = None,
        vllm_server_manager: object | None = None,
        poll_interval_s: float = GPU_HEALTH_POLL_INTERVAL_S,
    ):
        self._gpu_manager = gpu_manager or get_gpu_manager()
        self._vllm_mgr = vllm_server_manager
        self._poll_interval = poll_interval_s
        self._restart_tracker = RestartTracker()
        self._created_at = time.time()
        self._task: asyncio.Task | None = None
        self._running = False
        self._checking = False
        self._last_status: list[GPUHealthStatus] = []
        self._event_handlers: list = []

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"[gpu-health] Monitor started (interval={self._poll_interval}s, "
            f"grace={GPU_STARTUP_GRACE_S}s)"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("[gpu-health] Monitor stopped")

    def on_event(self, handler) -> None:
        self._event_handlers.append(handler)

    async def _emit_event(self, event_type: str, data: dict) -> None:
        for handler in self._event_handlers:
            try:
                result = handler(event_type, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[gpu-health] Monitor loop error: {exc}")
                await asyncio.sleep(self._poll_interval)

    async def _check_health(self) -> None:

        if self._checking:
            return
        self._checking = True

        try:

            elapsed = time.time() - self._created_at
            if elapsed < GPU_STARTUP_GRACE_S:
                logger.debug(
                    f"[gpu-health] Startup grace: {GPU_STARTUP_GRACE_S - elapsed:.0f}s remaining"
                )
                return

            statuses: list[GPUHealthStatus] = []


            if not self._gpu_manager.has_gpus:
                return

            for i in range(self._gpu_manager.gpu_count):
                status = await self._check_single_gpu(i)
                statuses.append(status)

            self._last_status = statuses


            await self._emit_event(
                "gpu:health_check",
                {
                    "statuses": [self._status_to_dict(s) for s in statuses],
                    "timestamp": time.time(),
                },
            )

        finally:
            self._checking = False

    async def _check_single_gpu(self, gpu_index: int) -> GPUHealthStatus:

        vram = get_live_vram_usage(gpu_index)
        util = vram.get("utilization", 0.0)

        level = "healthy"
        if util >= GPU_VRAM_CRITICAL_THRESHOLD:
            level = "critical"
        elif util >= GPU_VRAM_WARNING_THRESHOLD:
            level = "warning"

        status = GPUHealthStatus(
            gpu_index=gpu_index,
            vram_total_gb=vram.get("total_gb", 0.0),
            vram_used_gb=vram.get("used_gb", 0.0),
            vram_free_gb=vram.get("free_gb", 0.0),
            vram_utilization=util,
            level=level,
        )


        for role, port in [("judge", VLLM_JUDGE_PORT), ("miner", VLLM_MINER_PORT)]:
            server_status = await self._check_server_health(role, port)
            status.servers.append(server_status)


            if not server_status.is_running and server_status.error:
                await self._handle_unhealthy_server(role, gpu_index, server_status)


        if level == "critical":
            logger.error(
                f"[gpu-health] GPU {gpu_index} VRAM CRITICAL: {util:.1%} "
                f"({vram.get('used_gb', 0):.1f}/{vram.get('total_gb', 0):.1f} GB)"
            )
        elif level == "warning":
            logger.warning(
                f"[gpu-health] GPU {gpu_index} VRAM warning: {util:.1%} "
                f"({vram.get('used_gb', 0):.1f}/{vram.get('total_gb', 0):.1f} GB)"
            )

        return status

    async def _check_server_health(self, role: str, port: int) -> ServerHealthStatus:
        import httpx

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/health")
                elapsed_ms = (time.time() - start) * 1000

                if resp.status_code == 200:
                    return ServerHealthStatus(
                        role=role,
                        port=port,
                        is_running=True,
                        response_time_ms=round(elapsed_ms, 1),
                    )
                else:
                    return ServerHealthStatus(
                        role=role,
                        port=port,
                        is_running=False,
                        response_time_ms=round(elapsed_ms, 1),
                        error=f"HTTP {resp.status_code}",
                    )
        except httpx.ConnectError:
            return ServerHealthStatus(
                role=role,
                port=port,
                is_running=False,
                error="connection_refused",
            )
        except httpx.TimeoutException:
            return ServerHealthStatus(
                role=role,
                port=port,
                is_running=False,
                error="timeout",
            )
        except Exception as exc:
            return ServerHealthStatus(
                role=role,
                port=port,
                is_running=False,
                error=str(exc),
            )

    async def _handle_unhealthy_server(
        self,
        role: str,
        gpu_index: int,
        server_status: ServerHealthStatus,
    ) -> None:
        is_oom = self._detect_oom(server_status)


        self._gpu_manager.record_failure(gpu_index, is_oom=is_oom)

        reason = "oom" if is_oom else "crash"
        logger.warning(
            f"[gpu-health] {role} server unhealthy on GPU {gpu_index}: "
            f"{server_status.error} (reason={reason})"
        )


        if not self._restart_tracker.can_restart(role):
            await self._emit_event(
                "gpu:restart_blocked",
                {
                    "role": role,
                    "gpu_index": gpu_index,
                    "reason": reason,
                    "error": server_status.error,
                },
            )
            return


        if self._vllm_mgr is not None:
            try:
                logger.info(f"[gpu-health] Attempting auto-restart of {role} server...")
                self._restart_tracker.record_restart(role)

                if role == "judge" and hasattr(self._vllm_mgr, "judge"):
                    judge_client = self._vllm_mgr.judge
                    if judge_client.current_model:
                        model_name = judge_client.current_model.split("@")[0]
                        revision = (
                            judge_client.current_model.split("@")[1]
                            if "@" in judge_client.current_model
                            else "main"
                        )
                        judge_client.stop_server()

                        gpu_util = judge_client.gpu_memory_utilization
                        if is_oom:
                            gpu_util = max(0.2, gpu_util - 0.05)
                            logger.info(
                                f"[gpu-health] Reducing {role} GPU memory utilization "
                                f"to {gpu_util:.2f} after OOM"
                            )
                        judge_client.start_server(
                            model_name, revision, gpu_memory_utilization=gpu_util
                        )
                        self._gpu_manager.record_success(gpu_index)
                        await self._emit_event(
                            "gpu:restart_success",
                            {"role": role, "gpu_index": gpu_index, "reason": reason},
                        )
                elif role == "miner" and hasattr(self._vllm_mgr, "miner"):
                    miner_client = self._vllm_mgr.miner
                    if miner_client.current_model:
                        model_name = miner_client.current_model.split("@")[0]
                        revision = (
                            miner_client.current_model.split("@")[1]
                            if "@" in miner_client.current_model
                            else "main"
                        )
                        miner_client.stop_server()
                        gpu_util = miner_client.gpu_memory_utilization
                        if is_oom:
                            gpu_util = max(0.2, gpu_util - 0.05)
                        miner_client.start_server(
                            model_name, revision, gpu_memory_utilization=gpu_util
                        )
                        self._gpu_manager.record_success(gpu_index)
                        await self._emit_event(
                            "gpu:restart_success",
                            {"role": role, "gpu_index": gpu_index, "reason": reason},
                        )
            except Exception as exc:
                logger.error(f"[gpu-health] Auto-restart of {role} failed: {exc}")
                await self._emit_event(
                    "gpu:restart_failed",
                    {
                        "role": role,
                        "gpu_index": gpu_index,
                        "reason": reason,
                        "error": str(exc),
                    },
                )

    def _detect_oom(self, server_status: ServerHealthStatus) -> bool:
        if server_status.error is None:
            return False
        error_lower = server_status.error.lower()
        oom_indicators = [
            "out of memory",
            "oom",
            "cuda out of memory",
            "cudamalloc",
            "torch.cuda.outofmemoryerror",
            "memory allocation",
        ]
        return any(ind in error_lower for ind in oom_indicators)

    def _status_to_dict(self, status: GPUHealthStatus) -> dict:
        return {
            "gpu_index": status.gpu_index,
            "vram_total_gb": status.vram_total_gb,
            "vram_used_gb": status.vram_used_gb,
            "vram_free_gb": status.vram_free_gb,
            "vram_utilization": status.vram_utilization,
            "level": status.level,
            "servers": [
                {
                    "role": s.role,
                    "port": s.port,
                    "is_running": s.is_running,
                    "response_time_ms": s.response_time_ms,
                    "error": s.error,
                }
                for s in status.servers
            ],
        }

    @property
    def last_status(self) -> list[GPUHealthStatus]:
        return self._last_status

    def summary(self) -> dict:
        return {
            "running": self._running,
            "poll_interval_s": self._poll_interval,
            "uptime_s": round(time.time() - self._created_at, 1),
            "restart_tracker": self._restart_tracker.summary(),
            "last_status": [self._status_to_dict(s) for s in self._last_status],
        }


_health_monitor: GPUHealthMonitor | None = None


def get_health_monitor(
    gpu_manager: GPUManager | None = None,
    vllm_server_manager: object | None = None,
) -> GPUHealthMonitor:
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = GPUHealthMonitor(
            gpu_manager=gpu_manager,
            vllm_server_manager=vllm_server_manager,
        )
    return _health_monitor
