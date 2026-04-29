
from __future__ import annotations

import os as _os
import secrets as _secrets


def _env(name: str, default, *, cast=None):
    raw = _os.environ.get(name)
    if raw is None:
        return default
    if cast is not None:
        return cast(raw)
    return raw


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, default, cast=lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"})


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = _os.environ.get(name)
    if raw is None:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _detect_gpu_vram_gb() -> float:
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            return _torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    except Exception:
        pass
    return 0.0

_GPU_VRAM_GB: float = _detect_gpu_vram_gb()
_LARGE_GPU: bool = _GPU_VRAM_GB >= 60.0


VLLM_GPU_MEMORY_UTILIZATION: float = _env("VLLM_GPU_MEMORY_UTILIZATION", 0.90, cast=float)

VLLM_JUDGE_PORT: int = _env("VLLM_JUDGE_PORT", 8001, cast=int)
VLLM_MINER_PORT: int = _env("VLLM_MINER_PORT", 8000, cast=int)

GPU_TOTAL_VRAM_GB: float = 0.0
GPU_RESERVE_VRAM_GB: float = 2.0
JUDGE_VRAM_FRACTION: float = 0.90
MINER_VRAM_FRACTION: float = 0.85

GPU_HEALTH_POLL_INTERVAL_S: float = 30.0
GPU_STARTUP_GRACE_S: float = 60.0
GPU_MAX_RESTARTS_PER_HOUR: int = 3
GPU_RESTART_COOLDOWN_S: float = 120.0

GPU_OOM_COOLDOWN_BASE_S: float = 60.0
GPU_OOM_COOLDOWN_MULTIPLIER: float = 5.0
GPU_OOM_COOLDOWN_CAP_S: float = 3600.0
GPU_FAILURE_WINDOW_DECAY_S: float = 86400.0

GPU_VRAM_WARNING_THRESHOLD: float = 0.90
GPU_VRAM_CRITICAL_THRESHOLD: float = 0.95


VALID_PARAM_RANGES_B: list[tuple[float, float]] = [
    (0.45, 0.48),
    (1.5,  1.8),
    (3.5,  3.8),
]
VALIDATION_GPU_REQUIRED_GB: float = 8.0
EVALUATION_GPU_REQUIRED_GB: float = _env("EVALUATION_GPU_REQUIRED_GB", 10.0, cast=float)


HF_LOSS_MAX_SEQ_LEN: int = _env("HF_LOSS_MAX_SEQ_LEN", 8192, cast=int)
HF_LOSS_BATCH_SIZE: int = _env("HF_LOSS_BATCH_SIZE", 32, cast=int)


_HF_EVAL_SIZE_DEFAULTS: dict = {
    (0.45, 0.48): (
        _env("HF_EVAL_BATCH_450M",  256 if _LARGE_GPU else 128, cast=int),
        _env("HF_EVAL_SEQLEN_450M", 8192 if _LARGE_GPU else 4096, cast=int),
    ),
    (1.5, 1.8): (
        _env("HF_EVAL_BATCH_1B6",  128 if _LARGE_GPU else 48, cast=int),
        _env("HF_EVAL_SEQLEN_1B6", 8192 if _LARGE_GPU else 4096, cast=int),
    ),
    (3.5, 3.8): (
        _env("HF_EVAL_BATCH_3B7",  64 if _LARGE_GPU else 16, cast=int),
        _env("HF_EVAL_SEQLEN_3B7", 8192 if _LARGE_GPU else 4096, cast=int),
    ),
}


def get_eval_config_for_model_size(num_params_b: float) -> tuple[int, int]:
    _env_batch = _os.environ.get("HF_LOSS_BATCH_SIZE")
    _env_seq   = _os.environ.get("HF_LOSS_MAX_SEQ_LEN")

    for (lo, hi), (batch, seq_len) in _HF_EVAL_SIZE_DEFAULTS.items():
        if lo <= num_params_b <= hi:
            return (
                int(_env_batch) if _env_batch else batch,
                int(_env_seq)   if _env_seq   else seq_len,
            )

    return HF_LOSS_BATCH_SIZE, HF_LOSS_MAX_SEQ_LEN


HF_EVAL_ENABLE_4BIT: bool = _env(
    "HF_EVAL_ENABLE_4BIT",
    False,
    cast=lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"},
)

HF_EVAL_PREFER_FLASH_ATTN: bool = _env(
    "HF_EVAL_PREFER_FLASH_ATTN",
    True,
    cast=lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"},
)

HF_EVAL_TORCH_COMPILE: bool = _env(
    "HF_EVAL_TORCH_COMPILE",
    _LARGE_GPU,
    cast=lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"},
)


OWNER_API_URL: str = _env("OWNER_API_URL", "https://evolai-gate.hf.space")


DAILY_ALPHA_EMISSION: int = 2952  # 41% of 7200 total subnet emission goes to miners
DAILY_TAO_EMISSION: float = _env("DAILY_TAO_EMISSION", 0.5, cast=float)


STAGNATION_BURN_UID: int = 0


EVAL_REFERENCE_TOKENIZER: str = _env(
    "EVAL_REFERENCE_TOKENIZER", "Qwen/Qwen3-4B-Thinking-2507"
)

EVAL_THINK_MAX_NEW_TOKENS: int = _env("EVAL_THINK_MAX_NEW_TOKENS", 512, cast=int)

# Mamba2's O(1) recurrent state means per-token cost is constant regardless of
# sequence length. Budget matches transformer: models that output </think> stop
# early via token-ID; models that don't exhaust the budget but get think_gain≈0.
EVAL_THINK_MAX_NEW_TOKENS_MAMBA2: int = _env(
    "EVAL_THINK_MAX_NEW_TOKENS_MAMBA2", 1024, cast=int
)


SIDE_QUEST_N: int = _env("SIDE_QUEST_N", 2, cast=int)


SIDE_QUEST_MAX_NEW_TOKENS: int = _env("SIDE_QUEST_MAX_NEW_TOKENS", 50, cast=int)


SIDE_QUEST_MAX_CTX: int = _env("SIDE_QUEST_MAX_CTX", 4096, cast=int)


SIDE_QUEST_WEIGHT: float = _env("SIDE_QUEST_WEIGHT", 0.30, cast=float)


EPOCH_BLOCKS: int = _env("EPOCH_BLOCKS", 360, cast=int)


N_EVAL: int = _env("N_EVAL", 10, cast=int)


HISTORY_EPOCHS: int = _env("HISTORY_EPOCHS", 1800, cast=int)


W_ABS: float = _env("W_ABS", 0.50, cast=float)


W_FLOW: float = _env("W_FLOW", 0.25, cast=float)


W_QUALITY: float = _env("W_QUALITY", 0.25, cast=float)


PROGRESS_GAMMA: float = _env("PROGRESS_GAMMA", 1.0, cast=float)


PROGRESS_EMA_ALPHA: float = _env("PROGRESS_EMA_ALPHA", 0.10, cast=float)


PROGRESS_MIN_EVALUATIONS: int = _env("PROGRESS_MIN_EVALUATIONS", 1, cast=int)



# Sharpe-Flow mechanism parameters
MIN_FLOW_EPOCHS: int = _env("MIN_FLOW_EPOCHS", 10, cast=int)
FLOW_EPS: float = _env("FLOW_EPS", 1e-4, cast=float)
EMISSION_LAMBDA: float = _env("EMISSION_LAMBDA", 0.10, cast=float)

# Per-miner improvement+proximity scale parameters.
# Short/long EMA windows (in rounds) for the golden-cross improvement signal.
# α = 2/(N+1) is the standard EWM formula.
# At 20 rounds/day (360 blocks × 12s = 72 min/round):
#   short = 1 week  = 7d × 20 = 140 rounds
#   long  = 90 days          = 1800 rounds
EMA_SHORT_ROUNDS: int = _env("EMA_SHORT_ROUNDS", 140, cast=int)
EMA_LONG_ROUNDS: int = _env("EMA_LONG_ROUNDS", 1800, cast=int)
# Additive weights: miner_scale = W_IMPROVEMENT×improvement + W_PROXIMITY×proximity
W_IMPROVEMENT: float = _env("W_IMPROVEMENT", 0.3, cast=float)
W_PROXIMITY: float = _env("W_PROXIMITY", 0.7, cast=float)
# Global emission floor when no miner is improving (prevents subnet death).
EMISSION_FLOOR: float = _env("EMISSION_FLOOR", 0.2, cast=float)
# Proximity threshold: a miner at or above this fraction of the frontier
# is counted as "good" even if their loss isn't currently falling.
# Prevents emission decay when the subnet has genuinely reached a hard plateau.
EMISSION_PROXIMITY_THRESHOLD: float = _env("EMISSION_PROXIMITY_THRESHOLD", 0.95, cast=float)


EVAL_PENALTY_LOSS: float = _env("EVAL_PENALTY_LOSS", 10.0, cast=float)


WEIGHT_EXPONENT: float = _env("WEIGHT_EXPONENT", 2.0, cast=float)


COLDKEY_ARCHIVE_TTL_DAYS: int = _env("COLDKEY_ARCHIVE_TTL_DAYS", 7, cast=int)


_active_datasets_raw: str = _env(
    "ACTIVE_DATASETS",
    "evolai/universal_qa",
)
ACTIVE_DATASETS: list[str] = [
    d.strip() for d in _active_datasets_raw.split(",") if d.strip()
]

DATASET_SIZES: dict[str, int] = {
    "evolai/universal_qa": _env("DATASET_SIZE_UNIVERSAL_QA", 10000, cast=int),
}


# ── vLLM Server Configuration ────────────────────────────────────────────────

LOCAL_API_KEY: str = _env("VLLM_API_KEY", _secrets.token_urlsafe(32))

SERVER_START_TIMEOUT_S: float = _env("VLLM_START_TIMEOUT_S", 120.0, cast=float)
SERVER_HEALTH_RETRIES: int = _env("SERVER_HEALTH_RETRIES", 3, cast=int)
SERVER_HEALTH_INTERVAL_S: float = _env("SERVER_HEALTH_INTERVAL_S", 5.0, cast=float)
MINER_SERVER_SHUTDOWN_TIMEOUT_S: float = _env("MINER_SERVER_SHUTDOWN_TIMEOUT_S", 30.0, cast=float)

VLLM_HTTP_CLIENT_TIMEOUT_S: float = _env("VLLM_HTTP_CLIENT_TIMEOUT_S", 60.0, cast=float)
VLLM_HEALTH_CHECK_TIMEOUT_S: float = _env("VLLM_HEALTH_CHECK_TIMEOUT_S", 10.0, cast=float)
VLLM_POLL_INTERVAL_S: float = _env("VLLM_POLL_INTERVAL_S", 3.0, cast=float)
VLLM_STOP_WAIT_S: float = _env("VLLM_STOP_WAIT_S", 5.0, cast=float)

VLLM_JUDGE_GPU_INDEX: int = _env("VLLM_JUDGE_GPU_INDEX", 0, cast=int)
VLLM_MINER_GPU_INDEX: int = _env("VLLM_MINER_GPU_INDEX", 1, cast=int)
VLLM_JUDGE_MAX_MODEL_LEN: int = _env("VLLM_JUDGE_MAX_MODEL_LEN", 32768, cast=int)
VLLM_MINER_MAX_MODEL_LEN: int = _env("VLLM_MINER_MAX_MODEL_LEN", 8192, cast=int)
VLLM_JUDGE_GPU_MEMORY_UTILIZATION: float = _env("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", 0.95, cast=float)
VLLM_MINER_GPU_MEMORY_UTILIZATION: float = _env("VLLM_MINER_GPU_MEMORY_UTILIZATION", 0.90, cast=float)
VLLM_PARALLEL_MINER_BASE_PORT: int = _env("VLLM_PARALLEL_MINER_BASE_PORT", 9000, cast=int)
VLLM_PARALLEL_MINER_GPU_INDICES: list[int] = _env_int_list("VLLM_PARALLEL_MINER_GPU_INDICES", [])
VLLM_JUDGE_TENSOR_PARALLEL_SIZE: int = _env("VLLM_JUDGE_TENSOR_PARALLEL_SIZE", 1, cast=int)

VLLM_BASE_URL: str = _env("VLLM_BASE_URL", f"http://127.0.0.1:{VLLM_JUDGE_PORT}/v1")
VLLM_MINER_BASE_URL: str = _env("VLLM_MINER_BASE_URL", f"http://127.0.0.1:{VLLM_MINER_PORT}/v1")


# ── Judge Model Configuration ────────────────────────────────────────────────

_judge_models_raw: str = _env("JUDGE_MODELS", "Qwen/Qwen3-30B-A3B-Instruct-2507")
JUDGE_MODELS: list[str] = [m.strip() for m in _judge_models_raw.split(",") if m.strip()]

JUDGE_TEMPERATURE: float = _env("JUDGE_TEMPERATURE", 0.0, cast=float)
JUDGE_TIMEOUT_S: float = _env("JUDGE_TIMEOUT_S", 120.0, cast=float)
JUDGE_RETRY_ATTEMPTS: int = _env("JUDGE_RETRY_ATTEMPTS", 4, cast=int)
JUDGE_PROMPT_OVERHEAD_TOKENS: int = _env("JUDGE_PROMPT_OVERHEAD_TOKENS", 512, cast=int)
LOCAL_JUDGE_ENDPOINTS: dict[str, str] = {}


# ── Miner Response Streaming ─────────────────────────────────────────────────

OLLAMA_CHAT_URL: str = _env("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_NUM_CTX: int = _env("OLLAMA_NUM_CTX", 16384, cast=int)
OLLAMA_NUM_PREDICT: int = _env("OLLAMA_NUM_PREDICT", 2048, cast=int)


# ── Judge Cooldown & Fallback ────────────────────────────────────────────────

JUDGE_COOLDOWN_OOM_MS: int = _env("JUDGE_COOLDOWN_OOM_MS", 300_000, cast=int)
JUDGE_COOLDOWN_ERROR_MS: int = _env("JUDGE_COOLDOWN_ERROR_MS", 60_000, cast=int)
PROBE_INTERVAL_MS: int = _env("PROBE_INTERVAL_MS", 5_000, cast=int)
PROBE_MARGIN_MS: int = _env("PROBE_MARGIN_MS", 10_000, cast=int)


# ── Context & Token Management ───────────────────────────────────────────────

CONTEXT_WINDOW_HARD_MIN_TOKENS: int = _env("CONTEXT_WINDOW_HARD_MIN_TOKENS", 256, cast=int)
CONTEXT_WINDOW_WARN_BELOW_TOKENS: int = _env("CONTEXT_WINDOW_WARN_BELOW_TOKENS", 2048, cast=int)
SAFETY_MARGIN: float = _env("TOKEN_ESTIMATE_SAFETY_MARGIN", 1.2, cast=float)


# ── Conversation Compaction ──────────────────────────────────────────────────

COMPACTION_TRIGGER_K: int = _env("COMPACTION_TRIGGER_K", 10, cast=int)
COMPACTION_MAX_TOKENS: int = _env("COMPACTION_MAX_TOKENS", 1024, cast=int)
SUMMARY_KEEP_RECENT: int = _env("SUMMARY_KEEP_RECENT", 3, cast=int)
MAX_OVERFLOW_COMPACTION_ATTEMPTS: int = _env("MAX_OVERFLOW_COMPACTION_ATTEMPTS", 2, cast=int)


# ── Interview Configuration ──────────────────────────────────────────────────

INTERVIEW_BASE_TURNS: int = _env("INTERVIEW_BASE_TURNS", 3, cast=int)
INTERVIEW_MAX_TURNS: int = _env("INTERVIEW_MAX_TURNS", 8, cast=int)
INTERVIEW_MIN_TURNS: int = _env("INTERVIEW_MIN_TURNS", 1, cast=int)
NUM_QUESTIONS: int = _env("NUM_QUESTIONS", 20, cast=int)
TOTAL_INTERVIEW_TOKEN_LIMIT: int = _env("TOTAL_INTERVIEW_TOKEN_LIMIT", 20480, cast=int)
PER_RESPONSE_TOKEN_LIMIT: int = _env("PER_RESPONSE_TOKEN_LIMIT", 4096, cast=int)


# ── Rate Limiting ────────────────────────────────────────────────────────────

JUDGE_RATE_LIMIT_MAX_REQUESTS: int = _env("JUDGE_RATE_LIMIT_MAX_REQUESTS", 30, cast=int)
JUDGE_RATE_LIMIT_WINDOW_MS: int = _env("JUDGE_RATE_LIMIT_WINDOW_MS", 60_000, cast=int)
JUDGE_MIN_DELAY_MS: int = _env("JUDGE_MIN_DELAY_MS", 100, cast=int)
JUDGE_MAX_DELAY_MS: int = _env("JUDGE_MAX_DELAY_MS", 30_000, cast=int)
JUDGE_JITTER: float = _env("JUDGE_JITTER", 0.2, cast=float)
MINER_RATE_LIMIT_MAX_REQUESTS: int = _env("MINER_RATE_LIMIT_MAX_REQUESTS", 20, cast=int)
MINER_RATE_LIMIT_WINDOW_MS: int = _env("MINER_RATE_LIMIT_WINDOW_MS", 60_000, cast=int)


# ── Retry Configuration ─────────────────────────────────────────────────────

BASE_RUN_RETRY_ITERATIONS: int = _env("BASE_RUN_RETRY_ITERATIONS", 4, cast=int)
RUN_RETRY_PER_PROFILE: int = _env("RUN_RETRY_PER_PROFILE", 8, cast=int)


# ── Loop Detection ───────────────────────────────────────────────────────────

ANSWER_LOOP_HISTORY_SIZE: int = _env("ANSWER_LOOP_HISTORY_SIZE", 20, cast=int)
LOOP_WARNING_THRESHOLD: int = _env("LOOP_WARNING_THRESHOLD", 3, cast=int)
LOOP_WARNING_PENALTY: float = _env("LOOP_WARNING_PENALTY", 0.1, cast=float)
LOOP_CIRCUIT_BREAKER: int = _env("LOOP_CIRCUIT_BREAKER", 5, cast=int)


# ── Miner Answer Sanitization ───────────────────────────────────────────────

MINER_ANSWER_MAX_CHARS: int = _env("MINER_ANSWER_MAX_CHARS", 8192, cast=int)
MAX_RESPONSE_CONTEXT_SHARE: float = _env("MAX_RESPONSE_CONTEXT_SHARE", 0.3, cast=float)
HARD_MAX_RESPONSE_CHARS: int = _env("HARD_MAX_RESPONSE_CHARS", 16384, cast=int)
MIN_KEEP_CHARS: int = _env("MIN_KEEP_CHARS", 512, cast=int)


# ── Scoring ──────────────────────────────────────────────────────────────────

VALIDITY_SCORE: float = _env("VALIDITY_SCORE", 0.2, cast=float)
VALIDITY_CORRECTNESS_THRESH: float = _env("VALIDITY_CORRECTNESS_THRESH", 0.5, cast=float)
CONTESTED_SCORE_LOW: float = _env("CONTESTED_SCORE_LOW", 0.35, cast=float)
CONTESTED_SCORE_HIGH: float = _env("CONTESTED_SCORE_HIGH", 0.65, cast=float)
MAX_RAW_SCORE: float = VALIDITY_SCORE + (NUM_QUESTIONS * 1.0) + 2.0
LOW_CONFIDENCE_DISCOUNT: float = _env("LOW_CONFIDENCE_DISCOUNT", 0.5, cast=float)
PHASE2_WEIGHT: float = _env("PHASE2_WEIGHT", 0.4, cast=float)
PHASE2_SKIP_INTERVIEW_THRESHOLD: float = _env("PHASE2_SKIP_INTERVIEW_THRESHOLD", 0.3, cast=float)
INTERVIEW_WEIGHT: float = _env("INTERVIEW_WEIGHT", 0.6, cast=float)
