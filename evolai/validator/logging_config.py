"""
Structured JSON Logging Configuration for EvolAI Validator

Provides structured logging in JSON format for better searchability,
filtering, and integration with log aggregation systems (ELK, Datadog, etc.).
"""

import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    
    Outputs logs in JSON format with consistent fields for easy parsing.
    """
    
    def __init__(self, extra_fields: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON formatter
        
        Args:
            extra_fields: Additional fields to include in every log entry
        """
        super().__init__()
        self.extra_fields = extra_fields or {}
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }
        
        # Add extra fields from record
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data
        
        # Add configured extra fields
        log_data.update(self.extra_fields)
        
        return json.dumps(log_data)


class StructuredLogger:
    """
    Enhanced logger with structured logging support
    
    Wraps standard Python logger with methods for structured logging.
    """
    
    def __init__(self, name: str):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
    
    def _log_structured(self, level: int, message: str, extra: Optional[Dict] = None, exc_info=None):
        """
        Log with structured extra data
        
        Args:
            level: Log level
            message: Log message
            extra: Extra structured data
            exc_info: Exception info
        """
        record = logging.LogRecord(
            name=self.logger.name,
            level=level,
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=exc_info,
        )
        
        if extra:
            record.extra_data = extra
        
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data"""
        self._log_structured(logging.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log_structured(logging.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data"""
        self._log_structured(logging.WARNING, message, kwargs)
    
    def error(self, message: str, exc_info=None, **kwargs):
        """Log error message with structured data"""
        self._log_structured(logging.ERROR, message, kwargs, exc_info=exc_info)
    
    def critical(self, message: str, exc_info=None, **kwargs):
        """Log critical message with structured data"""
        self._log_structured(logging.CRITICAL, message, kwargs, exc_info=exc_info)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    use_json: bool = True,
    extra_fields: Optional[Dict[str, Any]] = None
):
    """
    Setup structured logging for the validator
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        use_json: Whether to use JSON formatting
        extra_fields: Additional fields to include in every log (e.g., validator_id)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if use_json:
        console_formatter = JSONFormatter(extra_fields)
    else:
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if use_json:
            file_formatter = JSONFormatter(extra_fields)
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Silence noisy libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("vllm").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging initialized: level={log_level}, json={use_json}, file={log_file}")


def get_structured_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)
