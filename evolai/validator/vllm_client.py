"""
vLLM Server Manager — Dual Server Architecture

Manages two separate vLLM server processes:
  - Judge server  (port 8001): stays warm across ALL miners in a round
  - Miner server  (port 8000): swapped per miner UID under evaluation

Both expose OpenAI-compatible /v1/chat/completions.
Judge calls go through judge_client.py, NOT through this module directly.

GPU awareness via gpu_manager.GPUManager:
  - VRAM budget allocation before starting servers
  - OOM detection → GPU slot cooldown + reduced memory utilization on restart
  - Health probes integrated with gpu_health_monitor.py

Design basis: VALIDATOR_INTERVIEW_EVALUATION.md §Local Model Deployment + §15 GPU Management
"""
import logging
import subprocess
import time
import httpx
from typing import Optional, Dict, List
import psutil

from .config import (
    VLLM_JUDGE_PORT,
    VLLM_MINER_PORT,
    VLLM_GPU_MEMORY_UTILIZATION,
    VLLM_JUDGE_GPU_MEMORY_UTILIZATION,
    VLLM_MINER_GPU_MEMORY_UTILIZATION,
    VLLM_JUDGE_GPU_INDEX,
    VLLM_MINER_GPU_INDEX,
    VLLM_JUDGE_MAX_MODEL_LEN,
    VLLM_MINER_MAX_MODEL_LEN,
    VLLM_PARALLEL_MINER_BASE_PORT,
    VLLM_PARALLEL_MINER_GPU_INDICES,
    VLLM_JUDGE_TENSOR_PARALLEL_SIZE,
    LOCAL_API_KEY,
    SERVER_START_TIMEOUT_S,
    SERVER_HEALTH_RETRIES,
    SERVER_HEALTH_INTERVAL_S,
    MINER_SERVER_SHUTDOWN_TIMEOUT_S,
    VLLM_HTTP_CLIENT_TIMEOUT_S,
    VLLM_HEALTH_CHECK_TIMEOUT_S,
    VLLM_POLL_INTERVAL_S,
    VLLM_STOP_WAIT_S,
)

logger = logging.getLogger(__name__)


class VLLMClient:
    """
    Client for a single vLLM OpenAI-compatible API server.
    
    Manages the subprocess lifecycle and provides chat-completions access.
    Use VLLMServerManager (below) for the dual judge+miner architecture.
    """
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000,
                 max_model_len: int = VLLM_MINER_MAX_MODEL_LEN,
                 gpu_memory_utilization: float = VLLM_GPU_MEMORY_UTILIZATION):
        self.host = host
        self.port = port
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.base_url = f"http://{host}:{port}/v1"
        self.client = httpx.Client(timeout=VLLM_HTTP_CLIENT_TIMEOUT_S)
        self.server_process = None
        self.current_model = None
        
    def is_server_running(self) -> bool:
        """Check if vLLM server is running"""
        try:
            response = self.client.get(f"http://{self.host}:{self.port}/health", timeout=VLLM_HEALTH_CHECK_TIMEOUT_S)
            return response.status_code == 200
        except Exception:
            return False
    
    def start_server(self, model_name: str, revision: str = "main",
                     gpu_memory_utilization: float | None = None,
                     gpu_index: "int | list[int] | None" = None,
                     tensor_parallel_size: int = 1):
        """
        Start vLLM server for a specific model.
        
        Args:
            model_name: HuggingFace model name
            revision: Model revision
            gpu_memory_utilization: GPU memory fraction (uses instance default if None)
            gpu_index: Restrict server to specific CUDA device(s) via CUDA_VISIBLE_DEVICES.
                       int  → single GPU (e.g. 0)
                       list → multiple GPUs for tensor-parallel (e.g. [0, 1])
                       None → no restriction (all visible GPUs)
            tensor_parallel_size: Tensor parallel degree (default 1, use 2 to split across 2 GPUs)
        """
        gpu_util = gpu_memory_utilization or self.gpu_memory_utilization
        
        # Check if already running with this model
        if self.is_server_running() and self.current_model == f"{model_name}@{revision}":
            logger.info(f"vLLM server already running with {model_name}@{revision}")
            return
        
        # Stop existing server if running different model
        if self.is_server_running():
            logger.info("Stopping existing vLLM server...")
            self.stop_server()

        # Kill any orphaned vLLM processes on this port BEFORE spawning our new one.
        # This must happen here (not in stop_server) so we never accidentally kill
        # a freshly started server belonging to another evaluation slot.
        for _proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                _cmdline = _proc.info.get('cmdline', [])
                if _cmdline and any('vllm' in a.lower() for a in _cmdline):
                    if str(self.port) in ' '.join(_cmdline):
                        logger.warning(f"Killing orphaned vLLM process {_proc.pid} on port {self.port}")
                        _proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        device_str = str(gpu_index) if gpu_index is not None else "auto"
        logger.info(
            f"Starting vLLM server for {model_name}@{revision} on port {self.port} "
            f"(CUDA device={device_str}, tp={tensor_parallel_size})"
        )
        
        # Build environment with CUDA device isolation
        import os
        env = os.environ.copy()
        if gpu_index is not None:
            if isinstance(gpu_index, list):
                # TP mode: expose multiple GPUs; vLLM will use all of them
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_index)
            else:
                # Single-GPU mode
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        
        # Start vLLM server (binary may live in a separate virtualenv).
        # Re-read env at call time so --vllm-bin CLI flag works even after
        # config.py was already imported.
        vllm_bin = os.environ.get("VLLM_EXECUTABLE", "vllm")

        # Validate the executable path to prevent running arbitrary binaries.
        import shutil
        resolved = shutil.which(vllm_bin)
        if resolved is None:
            raise FileNotFoundError(
                f"VLLM_EXECUTABLE={vllm_bin!r} not found on PATH. "
                "Set VLLM_EXECUTABLE to the full path of the vllm binary."
            )

        # Cap --max-model-len to the model's own max_position_embeddings.
        # Some models (e.g. Gemma 1.6B) have a smaller context window than our
        # configured default; vLLM will reject the request with a pydantic
        # ValidationError if we exceed it.
        effective_max_model_len = self.max_model_len
        try:
            from transformers import AutoConfig as _AutoConfig
            _cfg = _AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=False)
            model_max_len = getattr(_cfg, "max_position_embeddings", None)
            if model_max_len and model_max_len < effective_max_model_len:
                logger.info(
                    f"Capping --max-model-len from {effective_max_model_len} to "
                    f"{model_max_len} (model's max_position_embeddings)"
                )
                effective_max_model_len = model_max_len
        except Exception as _cfg_err:
            logger.warning(f"Could not read model config for max_position_embeddings: {_cfg_err}")

        cmd = [
            vllm_bin, "serve",
            model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--revision", revision,
            "--dtype", "auto",
            "--gpu-memory-utilization", str(gpu_util),
            "--max-model-len", str(effective_max_model_len),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--api-key", LOCAL_API_KEY,
            "--served-model-name", model_name,
        ]

        # Redirect stdout to a log file so HuggingFace download progress is
        # visible and the pipe buffer never blocks the process. stderr stays
        # as PIPE so we can surface it on early exit.
        import os as _os, tempfile as _tempfile
        _log_dir = _os.getenv("VLLM_LOG_DIR", _tempfile.gettempdir())
        log_path = _os.path.join(_log_dir, f"vllm_server_{self.port}.log")
        try:
            self._server_log_fh = open(log_path, "w")
        except OSError:
            import tempfile
            self._server_log_fh = tempfile.NamedTemporaryFile(
                mode="w", prefix=f"vllm_{self.port}_", suffix=".log", delete=False
            )
            log_path = self._server_log_fh.name
        logger.info(f"vLLM stdout → {log_path}  (tail -f {log_path})")
        logger.info("vLLM cmd: %s", " ".join(cmd))

        self.server_process = subprocess.Popen(
            cmd,
            stdout=self._server_log_fh,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        # Wait for server to be ready (poll every 5 s, max SERVER_START_TIMEOUT_S)
        logger.info(f"Waiting for vLLM server to be ready (max {SERVER_START_TIMEOUT_S}s)...")
        start_time = time.time()
        last_log_s = 0

        while time.time() - start_time < SERVER_START_TIMEOUT_S:
            # Fail fast: if the subprocess already died, surface its stderr
            if self.server_process.poll() is not None:
                stderr_tail = ""
                stdout_content = ""
                try:
                    stderr_tail = self.server_process.stderr.read()  # read all available
                except Exception:
                    pass
                # vLLM writes its Python traceback to stdout (captured in log file).
                # Show full log if small; else show first 4KB + last 8KB to capture
                # both the early crash message AND the final stack trace.
                try:
                    with open(log_path, "r") as _lf:
                        raw = _lf.read()
                    if len(raw) <= 12000:
                        stdout_content = raw
                    else:
                        stdout_content = (
                            raw[:4000]
                            + f"\n\n... [{len(raw) - 12000} bytes omitted] ...\n\n"
                            + raw[-8000:]
                        )
                except Exception:
                    pass

                # Always log the full crash output here so it's visible in the
                # console regardless of how the RuntimeError propagates upward
                # (the error message often gets truncated in table columns).
                logger.error(
                    f"vLLM process exited early (rc={self.server_process.returncode})\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"=== stdout log ({log_path}) ===\n{stdout_content}\n"
                    f"=== stderr ===\n{stderr_tail}\n"
                    f"=== end ==="
                )
                raise RuntimeError(
                    f"vLLM server process exited early (rc={self.server_process.returncode}).\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"--- stdout log ({log_path}) ---\n{stdout_content}\n"
                    f"--- stderr ---\n{stderr_tail}"
                )

            if self.is_server_running():
                self.current_model = f"{model_name}@{revision}"
                elapsed = round(time.time() - start_time)
                logger.info(f"vLLM server ready at {self.base_url} ({elapsed}s)")
                return

            elapsed = int(time.time() - start_time)
            if elapsed - last_log_s >= 15:
                logger.info(f"  Still waiting for vLLM server... ({elapsed}s elapsed)")
                last_log_s = elapsed

            time.sleep(VLLM_POLL_INTERVAL_S)

        raise RuntimeError(f"vLLM server failed to start within {SERVER_START_TIMEOUT_S}s")
    
    def stop_server(self):
        """Stop vLLM server and free GPU memory."""
        if self.server_process:
            try:
                # Terminate gracefully
                self.server_process.terminate()
                self.server_process.wait(timeout=MINER_SERVER_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                # Force kill if doesn't stop
                self.server_process.kill()
                try:
                    self.server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

            self.server_process = None
            self.current_model = None
            logger.info(f"vLLM server on port {self.port} stopped")

        # Give the GPU driver time to reclaim VRAM from the terminated process.
        # Without this pause, the next vLLM server can start before the driver
        # has released memory, causing it to see insufficient free VRAM and exit rc=1.
        logger.info(f"Waiting {VLLM_STOP_WAIT_S}s for GPU driver to reclaim VRAM after vLLM shutdown...")
        time.sleep(VLLM_STOP_WAIT_S)

        # Free PyTorch allocator cache in this process as well
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> str:
        """
        Generate text using vLLM /v1/completions (legacy text completion).
        
        For chat-based interactions prefer chat_generate() instead.
        """
        response = self.client.post(
            f"{self.base_url}/completions",
            json={
                "model": self.current_model.split('@')[0] if self.current_model else "unknown",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['text']

    def chat_generate(
        self,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate via /v1/chat/completions (OpenAI-compatible chat endpoint).
        
        Args:
            messages: List of {"role": ..., "content": ...} chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            Assistant content string.
        """
        model_name = (
            self.current_model.split("@")[0] if self.current_model else "unknown"
        )
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            },
            headers={"Authorization": f"Bearer {LOCAL_API_KEY}"},
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def __del__(self):
        """Cleanup on deletion — guarded against interpreter shutdown."""
        try:
            self.stop_server()
        except Exception:
            pass
        try:
            self.client.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Dual-server manager
# ──────────────────────────────────────────────────────────────────────────────

class VLLMServerManager:
    """
    Manages the dual judge + miner vLLM server architecture.
    
    - Judge server stays warm across the entire evaluation round.
    - Miner server is swapped per miner UID.
    - GPU-aware: allocates VRAM budget before starting, releases on stop.
    - On 2-GPU systems: judge runs on VLLM_JUDGE_GPU_INDEX (default 0),
      miner runs on VLLM_MINER_GPU_INDEX (default 1), fully isolated via
      CUDA_VISIBLE_DEVICES so each server sees only its own GPU.
    
    Usage:
        mgr = VLLMServerManager()
        mgr.start_judge("Qwen/Qwen3-30B-A3B-Instruct-2507")
        for miner in miners:
            mgr.start_miner(miner.model_name, miner.revision)
            # ... evaluate ...
        mgr.stop_all()
    """

    def __init__(
        self,
        judge_gpu_index: int = VLLM_JUDGE_GPU_INDEX,
        miner_gpu_index: int = VLLM_MINER_GPU_INDEX,
        judge_tensor_parallel_size: int = VLLM_JUDGE_TENSOR_PARALLEL_SIZE,
        judge_gpu_indices: "list[int] | None" = None,
    ) -> None:
        self.judge_tensor_parallel_size = judge_tensor_parallel_size
        # If TP > 1 and explicit indices given, use them; otherwise expand from judge_gpu_index
        if judge_gpu_indices is not None:
            self.judge_gpu_indices: list[int] = judge_gpu_indices
        elif judge_tensor_parallel_size > 1:
            self.judge_gpu_indices = list(range(judge_gpu_index, judge_gpu_index + judge_tensor_parallel_size))
        else:
            self.judge_gpu_indices = [judge_gpu_index]
        self.judge_gpu_index = self.judge_gpu_indices[0]  # primary GPU for health monitor compat
        self.miner_gpu_index = miner_gpu_index
        # Keep gpu_index as the judge GPU for backward-compat with health monitor
        self.gpu_index = self.judge_gpu_index
        self.judge = VLLMClient(
            port=VLLM_JUDGE_PORT,
            max_model_len=VLLM_JUDGE_MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_JUDGE_GPU_MEMORY_UTILIZATION,
        )
        self.miner = VLLMClient(
            port=VLLM_MINER_PORT,
            max_model_len=VLLM_MINER_MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_MINER_GPU_MEMORY_UTILIZATION,
        )
        self._gpu_manager = None  # Lazy import to avoid circular deps

    def _ensure_gpu_manager(self):
        """Lazily import and initialize GPU manager."""
        if self._gpu_manager is None:
            try:
                from .gpu_manager import get_gpu_manager
                self._gpu_manager = get_gpu_manager()
                self._gpu_manager.initialize()
            except Exception as exc:
                logger.warning(f"[server-mgr] GPU manager unavailable: {exc}")

    def start_judge(self, model_name: str, revision: str = "main") -> None:
        """Start judge server (stays warm across all miners) on the judge GPU(s)."""
        self._ensure_gpu_manager()
        gpu_display = self.judge_gpu_indices if self.judge_tensor_parallel_size > 1 else self.judge_gpu_index
        logger.info(
            f"[server-mgr] Starting judge server: {model_name} on GPU {gpu_display} "
            f"(tp={self.judge_tensor_parallel_size})"
        )

        # Allocate VRAM budget if GPU manager is available
        if self._gpu_manager and self._gpu_manager.has_gpus:
            result = self._gpu_manager.allocate_role("judge", self.judge_gpu_index)
            if result is None:
                logger.warning(
                    "[server-mgr] VRAM allocation failed for judge — "
                    "proceeding without budget tracking"
                )

        # Pass list of GPU indices when TP > 1, single int when TP = 1
        gpu_arg = self.judge_gpu_indices if self.judge_tensor_parallel_size > 1 else self.judge_gpu_index
        self.judge.start_server(
            model_name, revision,
            gpu_index=gpu_arg,
            tensor_parallel_size=self.judge_tensor_parallel_size,
        )

        # Record success on GPU slot
        if self._gpu_manager and self._gpu_manager.has_gpus:
            self._gpu_manager.record_success(self.judge_gpu_index)

    def start_miner(self, model_name: str, revision: str = "main") -> None:
        """Start (or swap) miner server for a specific miner model on the miner GPU."""
        self._ensure_gpu_manager()
        logger.info(
            f"[server-mgr] Starting miner server: {model_name}@{revision} on GPU {self.miner_gpu_index}"
        )

        # Release old miner VRAM if swapping
        if self._gpu_manager and self._gpu_manager.has_gpus:
            self._gpu_manager.release_role("miner")
            result = self._gpu_manager.allocate_role("miner", self.miner_gpu_index)
            if result is None:
                logger.warning(
                    "[server-mgr] VRAM allocation failed for miner — "
                    "proceeding without budget tracking"
                )

        self.miner.start_server(model_name, revision, gpu_index=self.miner_gpu_index)

        if self._gpu_manager and self._gpu_manager.has_gpus:
            self._gpu_manager.record_success(self.miner_gpu_index)

    def stop_miner(self) -> None:
        """Stop the miner server (between miners)."""
        self.miner.stop_server()
        # Release VRAM
        if self._gpu_manager and self._gpu_manager.has_gpus:
            self._gpu_manager.release_role("miner")

    def stop_all(self) -> None:
        """Stop both judge and miner servers."""
        logger.info("[server-mgr] Stopping all vLLM servers")
        self.miner.stop_server()
        self.judge.stop_server()
        # Release all VRAM
        if self._gpu_manager and self._gpu_manager.has_gpus:
            self._gpu_manager.release_role("judge")
            self._gpu_manager.release_role("miner")

    def probe_judge(self) -> bool:
        """Health check for judge server."""
        return self.judge.is_server_running()

    def probe_miner(self) -> bool:
        """Health check for miner server."""
        return self.miner.is_server_running()


# ──────────────────────────────────────────────────────────────────────────────
# Parallel miner manager (4+ GPU mode)
# ──────────────────────────────────────────────────────────────────────────────


class ParallelMinerServerManager:
    """
    Manages ONE shared judge server + N independent miner server slots.

    Intended for multi-GPU machines (e.g. 4× A100 80 GB):
      GPU 0        → judge  (port VLLM_JUDGE_PORT,         stays warm)
      GPU 1,2,3    → miners (ports VLLM_PARALLEL_MINER_BASE_PORT + slot_index)

    Each miner slot runs a fully independent vLLM server process with its own:
      • CUDA_VISIBLE_DEVICES (so it can never touch the judge GPU)
      • Port (base_port + slot_index)
      • VLLMClient instance

    All slots SHARE the judge server — vLLM handles concurrent requests from
    multiple EvaluationOrchestrator instances natively.

    Usage::

        mgr = ParallelMinerServerManager(
            judge_gpu_index=0,
            miner_gpu_indices=[1, 2, 3],
        )
        mgr.start_judge("Qwen/Qwen3-30B-A3B-Instruct-2507")

        # Assign miners to slots round-robin, swap per-miner
        for i, miner in enumerate(miners):
            slot = i % mgr.num_slots
            mgr.start_miner_slot(slot, miner.model_name, miner.revision)
            orch = mgr.make_orchestrator(slot, judge_pool=[...])
            result = await orch.evaluate_miner(...)
            mgr.stop_miner_slot(slot)

        mgr.stop_all()
    """

    def __init__(
        self,
        judge_gpu_index: int = VLLM_JUDGE_GPU_INDEX,
        miner_gpu_indices: "list[int] | None" = None,
        miner_base_port: int = VLLM_PARALLEL_MINER_BASE_PORT,
        judge_tensor_parallel_size: int = VLLM_JUDGE_TENSOR_PARALLEL_SIZE,
        judge_gpu_indices: "list[int] | None" = None,
    ) -> None:
        self.judge_tensor_parallel_size = judge_tensor_parallel_size
        # Resolve judge GPU list (supports TP > 1)
        if judge_gpu_indices is not None:
            self.judge_gpu_indices: list[int] = judge_gpu_indices
        elif judge_tensor_parallel_size > 1:
            self.judge_gpu_indices = list(range(judge_gpu_index, judge_gpu_index + judge_tensor_parallel_size))
        else:
            self.judge_gpu_indices = [judge_gpu_index]
        self.judge_gpu_index = self.judge_gpu_indices[0]  # primary GPU
        self.miner_gpu_indices: list[int] = (
            miner_gpu_indices
            if miner_gpu_indices is not None
            else list(VLLM_PARALLEL_MINER_GPU_INDICES)
        )
        self.miner_base_port = miner_base_port

        # Judge client (shared)
        self.judge = VLLMClient(
            port=VLLM_JUDGE_PORT,
            max_model_len=VLLM_JUDGE_MAX_MODEL_LEN,
            gpu_memory_utilization=VLLM_JUDGE_GPU_MEMORY_UTILIZATION,
        )

        # One VLLMClient per miner slot
        self._miner_slots: list[VLLMClient] = [
            VLLMClient(
                port=miner_base_port + slot,
                max_model_len=VLLM_MINER_MAX_MODEL_LEN,
                gpu_memory_utilization=VLLM_MINER_GPU_MEMORY_UTILIZATION,
            )
            for slot in range(len(self.miner_gpu_indices))
        ]

    @property
    def num_slots(self) -> int:
        """Number of parallel miner slots available."""
        return len(self._miner_slots)

    def miner_base_url(self, slot: int) -> str:
        """Return the OpenAI-compatible base URL for miner slot `slot`."""
        port = self.miner_base_port + slot
        return f"http://127.0.0.1:{port}/v1"

    def start_judge(self, model_name: str, revision: str = "main") -> None:
        """Start the shared judge server on the judge GPU(s) (call once before any evaluations)."""
        gpu_display = self.judge_gpu_indices if self.judge_tensor_parallel_size > 1 else self.judge_gpu_index
        logger.info(
            f"[parallel-mgr] Starting judge: {model_name} on GPU {gpu_display} "
            f"(tp={self.judge_tensor_parallel_size})"
        )
        gpu_arg = self.judge_gpu_indices if self.judge_tensor_parallel_size > 1 else self.judge_gpu_index
        self.judge.start_server(
            model_name, revision,
            gpu_index=gpu_arg,
            tensor_parallel_size=self.judge_tensor_parallel_size,
        )

    def start_miner_slot(
        self, slot: int, model_name: str, revision: str = "main"
    ) -> None:
        """
        Start (or swap) the miner server for `slot`.

        The server process is pinned to `miner_gpu_indices[slot]` via
        CUDA_VISIBLE_DEVICES so it is completely isolated from the judge.
        """
        if slot >= self.num_slots:
            raise ValueError(f"Invalid slot {slot} — only {self.num_slots} slots configured")
        gpu_idx = self.miner_gpu_indices[slot]
        logger.info(
            f"[parallel-mgr] Starting miner slot {slot}: {model_name}@{revision} "
            f"on GPU {gpu_idx} port {self.miner_base_port + slot}"
        )
        self._miner_slots[slot].start_server(model_name, revision, gpu_index=gpu_idx)

    def stop_miner_slot(self, slot: int) -> None:
        """Stop miner server for `slot` and release its GPU memory."""
        if slot >= self.num_slots:
            return
        logger.info(f"[parallel-mgr] Stopping miner slot {slot}")
        self._miner_slots[slot].stop_server()

    def probe_slot(self, slot: int) -> bool:
        """Health check for a specific miner slot."""
        if slot >= self.num_slots:
            return False
        return self._miner_slots[slot].is_server_running()

    def stop_all(self) -> None:
        """Gracefully stop all servers (judge + all miner slots)."""
        logger.info("[parallel-mgr] Stopping all servers")
        for slot in range(self.num_slots):
            self._miner_slots[slot].stop_server()
        self.judge.stop_server()

    def make_orchestrator(
        self,
        slot: int,
        judge_pool: list[str],
    ) -> "EvaluationOrchestrator":  # type: ignore[name-defined]   # imported at call site
        """
        Create an EvaluationOrchestrator bound to miner slot `slot`.

        The orchestrator points its miner calls at the slot's vLLM port,
        and its judge calls at the shared judge port.
        """
        # Lazy import to avoid circular dependency
        from .orchestrator import EvaluationOrchestrator  # noqa: PLC0415

        # The model_id stored in the slot's VLLMClient
        client = self._miner_slots[slot]
        miner_model_id = (
            client.current_model.split("@")[0]
            if client.current_model
            else ""
        )
        return EvaluationOrchestrator(
            judge_model_pool=judge_pool,
            miner_backend="vllm",
            miner_model_id=miner_model_id,
            miner_vllm_base_url=self.miner_base_url(slot),
        )
