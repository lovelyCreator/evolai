
import time
import logging
from typing import Optional, Callable, TypeVar, Any
from dataclasses import dataclass
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ErrorSeverity(Enum):
    TRANSIENT = "transient"
    RECOVERABLE = "recoverable"
    PERMANENT = "permanent"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    NETWORK = "network"
    GPU_OOM = "gpu_oom"
    DISK_FULL = "disk_full"
    MODEL_LOAD = "model_load"
    INFERENCE = "inference"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class ValidatorError(Exception):
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.RECOVERABLE,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.original_error = original_error
        self.timestamp = time.time()


class GPUOutOfMemoryError(ValidatorError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.GPU_OOM,
            severity=ErrorSeverity.RECOVERABLE,
            original_error=original_error
        )


class ModelLoadError(ValidatorError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_LOAD,
            severity=ErrorSeverity.RECOVERABLE,
            original_error=original_error
        )


class DiskSpaceError(ValidatorError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.DISK_FULL,
            severity=ErrorSeverity.CRITICAL,
            original_error=original_error
        )


class NetworkError(ValidatorError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.TRANSIENT,
            original_error=original_error
        )


class ContextOverflowError(ValidatorError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.RECOVERABLE,
            original_error=original_error
        )


class JudgeCallError(ValidatorError):
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.TRANSIENT,
            original_error=original_error
        )


class RateLimitError(ValidatorError):
    def __init__(self, message: str, retry_after_ms: Optional[int] = None,
                 original_error: Optional[Exception] = None):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.TRANSIENT,
            original_error=original_error
        )
        self.retry_after_ms = retry_after_ms


class FailoverReason(Enum):
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    TIMEOUT = "timeout"
    FORMAT = "format"
    MODEL_NOT_FOUND = "model_not_found"
    CONTEXT_OVERFLOW = "context_overflow"
    OOM = "oom"
    UNKNOWN = "unknown"


FAILOVER_HTTP_MAP: dict = {
    FailoverReason.BILLING: 402,
    FailoverReason.RATE_LIMIT: 429,
    FailoverReason.AUTH: 401,
    FailoverReason.TIMEOUT: 408,
    FailoverReason.FORMAT: 400,
    FailoverReason.MODEL_NOT_FOUND: 404,
}


FAILOVER_SHOULD_RETRY: dict = {
    FailoverReason.BILLING: False,
    FailoverReason.RATE_LIMIT: True,
    FailoverReason.AUTH: False,
    FailoverReason.TIMEOUT: True,
    FailoverReason.FORMAT: False,
    FailoverReason.MODEL_NOT_FOUND: False,
    FailoverReason.CONTEXT_OVERFLOW: False,
    FailoverReason.OOM: False,
    FailoverReason.UNKNOWN: True,
}


class FailoverError(ValidatorError):
    def __init__(
        self,
        message: str,
        reason: FailoverReason,
        original_error: Optional[Exception] = None,
    ):
        severity = (
            ErrorSeverity.TRANSIENT
            if FAILOVER_SHOULD_RETRY.get(reason, True)
            else ErrorSeverity.RECOVERABLE
        )
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=severity,
            original_error=original_error,
        )
        self.reason = reason

    @property
    def http_status(self) -> int:
        return FAILOVER_HTTP_MAP.get(self.reason, 500)

    @property
    def should_retry(self) -> bool:
        return FAILOVER_SHOULD_RETRY.get(self.reason, True)

    def __repr__(self) -> str:
        return f"FailoverError(reason={self.reason.value}, status={self.http_status})"


def classify_judge_error(error: Exception) -> FailoverReason:

    response = getattr(error, "response", None)
    status = getattr(response, "status_code", 0) or getattr(response, "status", 0)

    if status:
        for reason, code in FAILOVER_HTTP_MAP.items():
            if status == code:
                return reason


    msg = str(error).lower()
    if "context" in msg and ("length" in msg or "overflow" in msg):
        return FailoverReason.CONTEXT_OVERFLOW
    if "out of memory" in msg or "oom" in msg or "cuda" in msg:
        return FailoverReason.OOM
    if "rate" in msg and "limit" in msg:
        return FailoverReason.RATE_LIMIT
    if "unauthorized" in msg or "forbidden" in msg or "auth" in msg:
        return FailoverReason.AUTH
    if "timeout" in msg or "timed out" in msg:
        return FailoverReason.TIMEOUT
    if "not found" in msg and "model" in msg:
        return FailoverReason.MODEL_NOT_FOUND
    if "billing" in msg or "quota" in msg or "insufficient" in msg:
        return FailoverReason.BILLING

    return FailoverReason.UNKNOWN


def classify_error(error: Exception) -> ValidatorError:
    import torch
    

    if isinstance(error, ValidatorError):
        return error
    

    if isinstance(error, torch.cuda.OutOfMemoryError):
        return GPUOutOfMemoryError(
            f"GPU out of memory: {str(error)}",
            original_error=error
        )
    

    if is_likely_context_overflow(error):
        return ContextOverflowError(
            f"Context window overflow: {str(error)}",
            original_error=error
        )
    

    if isinstance(error, (OSError, IOError)):
        error_msg = str(error).lower()
        if "no space left" in error_msg or "disk full" in error_msg:
            return DiskSpaceError(
                f"Disk space exhausted: {str(error)}",
                original_error=error
            )
    

    if "connection" in str(error).lower() or "timeout" in str(error).lower():
        return NetworkError(
            f"Network error: {str(error)}",
            original_error=error
        )
    

    if "rate" in str(error).lower() and "limit" in str(error).lower():
        return RateLimitError(
            f"Rate limited: {str(error)}",
            original_error=error
        )
    

    if "pretrained" in str(error).lower() or "checkpoint" in str(error).lower():
        return ModelLoadError(
            f"Model load failed: {str(error)}",
            original_error=error
        )
    

    return ValidatorError(
        f"Unclassified error: {str(error)}",
        category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.RECOVERABLE,
        original_error=error
    )


def is_likely_context_overflow(error: Exception) -> bool:
    msg = str(error).lower()
    overflow_patterns = [
        "context length",
        "context_length_exceeded",
        "maximum context",
        "token limit",
        "too many tokens",
        "prompt is too long",
        "input too long",
        "max_tokens",
        "model_max_len",
        "exceeds the model",
    ]
    return any(p in msg for p in overflow_patterns)


def with_retry(
    config: Optional[RetryConfig] = None,
    retryable_categories: Optional[set] = None
):
    config = config or RetryConfig()
    retryable = retryable_categories or {
        ErrorCategory.NETWORK,
        ErrorCategory.GPU_OOM,
        ErrorCategory.MODEL_LOAD
    }
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:

                    classified_error = classify_error(e)
                    last_error = classified_error
                    

                    if classified_error.category not in retryable:
                        logger.error(
                            f"{func.__name__} failed with non-retryable error: "
                            f"{classified_error.category.value}"
                        )
                        raise classified_error
                    

                    if attempt >= config.max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {attempt} attempts"
                        )
                        raise classified_error
                    

                    delay = min(
                        config.initial_delay_seconds * (config.exponential_base ** (attempt - 1)),
                        config.max_delay_seconds
                    )
                    

                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{config.max_attempts}), "
                        f"retrying in {delay:.2f}s. Error: {str(classified_error)}"
                    )
                    
                    time.sleep(delay)
            

            raise last_error or RuntimeError(f"{func.__name__} failed")
        
        return wrapper
    return decorator


class CircuitBreaker:
    
    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 60.0,
        name: str = "circuit_breaker"
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.name = name
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:

        if self.state == "OPEN":

            if time.time() - self.last_failure_time < self.timeout_seconds:
                raise RuntimeError(
                    f"Circuit breaker {self.name} is OPEN. "
                    f"Failures: {self.failure_count}/{self.failure_threshold}"
                )
            else:

                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
                self.state = "HALF_OPEN"
                self.success_count = 0
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info(f"Circuit breaker {self.name} CLOSING after {self.success_count} successes")
                self.state = "CLOSED"
                self.success_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            logger.error(
                f"Circuit breaker {self.name} OPENING after {self.failure_count} failures"
            )
            self.state = "OPEN"
        

        if self.state == "HALF_OPEN":
            logger.warning(f"Circuit breaker {self.name} returning to OPEN from HALF_OPEN")
            self.state = "OPEN"
            self.success_count = 0
    
    def get_state(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time
        }
    
    def reset(self):
        logger.info(f"Manually resetting circuit breaker {self.name}")
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
