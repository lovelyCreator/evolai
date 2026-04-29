
import logging
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .error_handling import (
    classify_error,
    GPUOutOfMemoryError,
    ModelLoadError,
)
from .health_checks import WatchdogTimer
from .metrics import get_metrics, Timer
from .resource_manager import ResourceManager

logger = logging.getLogger(__name__)


def prefetch_model_to_disk(model_name: str, revision: Optional[str], dest_dir: str) -> None:
    """Download all model files into dest_dir without loading into GPU memory.

    Pass dest_dir to load_model(prefetch_dir=...) to skip re-downloading.
    Ignores JAX/TF/Rust checkpoints to save bandwidth.
    """
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_name,
        revision=revision or "main",
        local_dir=dest_dir,
        ignore_patterns=[
            "*.msgpack", "flax_model*", "tf_model*", "rust_model*",
        ],
    )
    logger.info(f"Prefetched {model_name}@{revision} → {dest_dir}")


def purge_hf_model_cache(model_name: str) -> None:
    try:
        hf_home = os.environ.get(
            "HF_HOME", os.path.join(Path.home(), ".cache", "huggingface")
        )
        hub_cache = os.path.join(hf_home, "hub")
        safe_name = model_name.replace("/", "--")
        model_cache_dir = os.path.join(hub_cache, f"models--{safe_name}")

        if os.path.isdir(model_cache_dir):
            shutil.rmtree(model_cache_dir)
            logger.info(f"Purged HF model cache: {model_cache_dir}")
        else:
            logger.debug(f"No HF cache found for {model_name} at {model_cache_dir}")
    except Exception as exc:
        logger.warning(f"Failed to purge HF cache for {model_name}: {exc}")


class TimeoutError(Exception):
    pass


def timeout(seconds: int):
    def decorator(func):
        if os.name != "nt" and hasattr(signal, "SIGALRM"):
            def _handle_timeout(signum, frame):
                raise TimeoutError(f"Operation timed out after {seconds}s")

            def wrapper(*args, **kwargs):
                old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
        else:
            import threading

            def wrapper(*args, **kwargs):
                result = [None]
                exception = [None]

                def target():
                    try:
                        result[0] = func(*args, **kwargs)
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=target, daemon=True)
                thread.start()
                thread.join(timeout=seconds)

                if thread.is_alive():
                    raise TimeoutError(f"Operation timed out after {seconds}s")
                if exception[0] is not None:
                    raise exception[0]
                return result[0]

        return wrapper
    return decorator


class ModelValidator:

    def __init__(self, device: str = "cuda", health_checker=None):
        self.device = device
        self.hf_api = HfApi()
        self.resource_mgr = ResourceManager()
        self.health_checker = health_checker
        self.metrics = get_metrics()

        self.num_gpus = torch.cuda.device_count()
        self.miner_gpus = list(range(self.num_gpus)) if self.num_gpus else [0]
        self.current_miner_gpu_index = 0
        logger.info(f"GPU allocation: {self.num_gpus} GPUs detected")


    def validate_model(
        self,
        model_name: str,
        revision: Optional[str] = None,
        track: str = "transformer",
    ) -> Tuple[bool, List[str], Dict]:
        issues: List[str] = []
        info: Dict = {}

        if "/" not in model_name:
            issues.append("Model name must be in format: username/evolai-model-name")
        else:
            model_part = model_name.split("/")[-1]
            if "evolai" not in model_part.lower():
                issues.append(
                    "Model name must contain 'evolai' in the model part (after /)"
                )

        try:
            logger.info(f"[1/4] Checking GPU memory for {model_name}")
            from .config import VALIDATION_GPU_REQUIRED_GB

            if not self.resource_mgr.gpu_manager.check_available_memory(
                required_gb=VALIDATION_GPU_REQUIRED_GB
            ):
                issues.append("Insufficient GPU memory available (need ~8GB free)")
                return False, issues, info
            logger.info("[1/4] GPU memory check passed")

            logger.info(f"[2/4] Loading config for {model_name}@{revision}")
            config = AutoConfig.from_pretrained(
                model_name, revision=revision, trust_remote_code=False
            )
            logger.info("[2/4] Config loaded successfully")

            logger.info("[3/4] Loading model on CPU for parameter counting")
            _cpu_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision=revision,
                config=config,
                trust_remote_code=False,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            logger.info("[3/4] Model loaded on CPU")

            total_params = sum(p.numel() for p in _cpu_model.parameters())
            total_params_b = total_params / 1e9
            logger.info(f"[3/4] {total_params_b:.2f}B params")

            del _cpu_model
            import gc as _gc
            _gc.collect()

            info["total_params"] = total_params
            info["total_params_b"] = round(total_params_b, 2)
            info["architecture"] = (
                config.model_type if hasattr(config, "model_type") else "unknown"
            )

            logger.info("[4/4] Validating parameter count")
            from .config import VALID_PARAM_RANGES_B

            if not any(lo <= total_params_b <= hi for lo, hi in VALID_PARAM_RANGES_B):
                ranges_str = ", ".join(
                    f"{lo:.2f}-{hi:.2f}B" for lo, hi in VALID_PARAM_RANGES_B
                )
                issues.append(
                    f"Model has {total_params_b:.2f}B parameters. "
                    f"Must be one of: {ranges_str}"
                )

            logger.info(f"[4/4] Validation complete, is_valid={len(issues) == 0}")

        except torch.cuda.OutOfMemoryError as e:
            self.resource_mgr.emergency_cleanup()
            issues.append(f"GPU out of memory during validation: {e}")
            return False, issues, info
        except Exception as e:
            issues.append(f"Failed to load model: {e}")
            return False, issues, info

        return len(issues) == 0, issues, info


    def load_model(
        self,
        model_name: str,
        revision: Optional[str] = None,
        use_vllm: bool = False,
        timeout_seconds: int = 600,
        prefetch_dir: Optional[str] = None,
        force_float32: bool = False,
    ):
        self.metrics.get_counter("model_loads_total").inc()

        temp_dir = tempfile.mkdtemp(
            prefix=f"evolai_model_{model_name.replace('/', '_')}_"
        )
        loaded_model = None

        def cleanup():
            nonlocal loaded_model
            try:
                if loaded_model is not None:
                    logger.info(f"Cleaning up model {model_name} from GPU memory")
                    try:
                        del loaded_model
                        loaded_model = None
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup model from GPU: {e}")
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Temp directory {temp_dir} deleted")
                purge_hf_model_cache(model_name)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir {temp_dir}: {e}")

        def on_timeout():
            logger.error(
                f"Model loading timed out after {timeout_seconds}s: {model_name}"
            )
            cleanup()
            self.metrics.get_counter("model_loads_failed").inc()

        try:
            with WatchdogTimer(
                timeout_seconds, on_timeout, name=f"load_model_{model_name}"
            ):
                with Timer(self.metrics.get_histogram("model_load_duration_seconds")):
                    from .config import (
                        EVALUATION_GPU_REQUIRED_GB,
                        HF_EVAL_ENABLE_4BIT,
                        HF_EVAL_PREFER_FLASH_ATTN,
                        HF_EVAL_TORCH_COMPILE,
                    )

                    if not self.resource_mgr.gpu_manager.check_available_memory(
                        required_gb=EVALUATION_GPU_REQUIRED_GB
                    ):
                        raise RuntimeError(
                            f"Insufficient GPU memory to load model "
                            f"(need ~{EVALUATION_GPU_REQUIRED_GB:.0f} GB free)"
                        )

                    if torch.cuda.is_available():
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True

                    # On non-bf16 hardware (e.g. V100), SSM/Mamba2 models
                    # trained in bfloat16 overflow when cast to float16
                    # (exp/softplus in selective scan → inf → NaN loss).
                    # force_float32=True uses fp32 instead, which is safe on
                    # all hardware and fits easily for sub-4B models.
                    if force_float32:
                        compute_dtype = torch.float32
                    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        compute_dtype = torch.bfloat16
                    else:
                        compute_dtype = torch.float16

                    # If a pre-downloaded local dir is provided, load from
                    # there directly — no network call needed.
                    _src = (
                        prefetch_dir
                        if prefetch_dir and os.path.isdir(prefetch_dir)
                        else model_name
                    )
                    _is_local = _src != model_name

                    base_kwargs = {
                        "trust_remote_code": False,
                        "device_map": "auto",
                        "low_cpu_mem_usage": True,
                        "torch_dtype": compute_dtype,
                    }
                    if not _is_local:
                        base_kwargs["revision"] = revision
                        base_kwargs["cache_dir"] = temp_dir

                    load_attempts = []

                    quantization_config = None
                    if HF_EVAL_ENABLE_4BIT and torch.cuda.is_available():
                        try:
                            from transformers import BitsAndBytesConfig

                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=compute_dtype,
                            )
                        except Exception as exc:
                            logger.warning(f"4-bit quantization unavailable: {exc}")

                    if quantization_config is not None:
                        quant_kwargs = dict(base_kwargs)
                        quant_kwargs["quantization_config"] = quantization_config
                        load_attempts.append((
                            "4bit+flash_attention_2",
                            {**quant_kwargs, "attn_implementation": "flash_attention_2"},
                        ))
                        load_attempts.append((
                            "4bit+sdpa",
                            {**quant_kwargs, "attn_implementation": "sdpa"},
                        ))
                        load_attempts.append(("4bit", quant_kwargs))

                    if HF_EVAL_PREFER_FLASH_ATTN and torch.cuda.is_available():
                        load_attempts.append((
                            "fp16+flash_attention_2",
                            {**base_kwargs, "attn_implementation": "flash_attention_2"},
                        ))
                        load_attempts.append((
                            "fp16+sdpa",
                            {**base_kwargs, "attn_implementation": "sdpa"},
                        ))

                    load_attempts.append(("fp16", base_kwargs))


                    _prev_hf_bars = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
                    try:
                        from transformers.utils import logging as _tf_logging

                        _tf_logging.disable_progress_bar()
                        _tf_progress_disabled = True
                    except Exception:
                        _tf_progress_disabled = False

                    last_error = None
                    model = None
                    try:
                        for attempt_name, attempt_kwargs in load_attempts:
                            try:
                                logger.info(
                                    f"Loading {model_name} ({attempt_name})"
                                )
                                model = AutoModelForCausalLM.from_pretrained(
                                    _src, **attempt_kwargs
                                )
                                logger.info(
                                    f"Loaded {model_name} ({attempt_name})"
                                )
                                break
                            except Exception as exc:
                                last_error = exc
                                logger.warning(
                                    f"HF load failed for {model_name} "
                                    f"({attempt_name}): {exc}"
                                )
                    finally:
                        if _prev_hf_bars is None:
                            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                        else:
                            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = _prev_hf_bars
                        if _tf_progress_disabled:
                            try:
                                _tf_logging.enable_progress_bar()
                            except Exception:
                                pass

                    if model is None:
                        if (
                            last_error is not None
                            and "does not recognize this architecture"
                            in str(last_error)
                        ):
                            raise RuntimeError(
                                f"Failed to load model '{model_name}': "
                                f"unrecognised architecture.\n"
                                f"Upgrade Transformers to >=5.3.0:\n"
                                f"  pip install 'transformers>=5.3.0'"
                            )
                        raise RuntimeError(
                            f"Failed to load model via HF path: {last_error}"
                        )

                    loaded_model = model

                    _tok_extra = (
                        {}
                        if _is_local
                        else {"revision": revision, "cache_dir": temp_dir}
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        _src,
                        trust_remote_code=False,
                        **_tok_extra,
                    )
                    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                        tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.padding_side = "right"

                    if hasattr(model, "config"):
                        model.config.use_cache = False
                    model.eval()

                    if HF_EVAL_TORCH_COMPILE and hasattr(torch, "compile"):
                        try:
                            model = torch.compile(
                                model, mode="reduce-overhead", dynamic=True
                            )
                            logger.info(f"torch.compile applied to {model_name}")
                        except Exception as exc:
                            logger.warning(
                                f"torch.compile skipped for {model_name}: {exc}"
                            )

                    return model, tokenizer, False, cleanup

        except torch.cuda.OutOfMemoryError as e:
            cleanup()
            self.resource_mgr.emergency_cleanup()
            self.metrics.get_counter("oom_errors_total").inc()
            self.metrics.get_counter("model_loads_failed").inc()
            raise GPUOutOfMemoryError(
                f"OOM loading model {model_name}", original_error=e
            )
        except (RuntimeError, ModelLoadError, GPUOutOfMemoryError):
            cleanup()
            self.metrics.get_counter("model_loads_failed").inc()
            raise
        except Exception as e:
            cleanup()
            error = classify_error(e)
            self.metrics.get_counter("model_loads_failed").inc()
            self.metrics.get_counter("errors_total").inc()
            raise ModelLoadError(
                f"Failed to load model {model_name}: {error}", original_error=e
            )
