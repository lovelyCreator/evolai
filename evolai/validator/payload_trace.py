"""
Payload Trace Diagnostics

Derived from OpenClaw src/agents/cache-trace.ts + anthropic-payload-log.ts.
Creates SHA-256 fingerprints of judge call payloads at each pipeline stage
and writes JSONL trace files for debugging judge inconsistencies.

Enabled via EVOLAI_TRACE_PAYLOADS=1 environment variable.
Trace files are written to logs/judge_traces/trace.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Trace stages (mirroring OpenClaw cache-trace.ts stages)
STAGE_PROMPT_BEFORE = "prompt:before"
STAGE_PROMPT_SANITIZED = "prompt:sanitized"
STAGE_PROMPT_TRUNCATED = "prompt:truncated"
STAGE_STREAM_CONTEXT = "stream:context"
STAGE_RESPONSE_RECEIVED = "response:received"
STAGE_RESPONSE_PARSED = "response:parsed"
STAGE_COMPACTION = "compaction:applied"


def _fingerprint(messages: list[dict]) -> str:
    """
    Compute SHA-256 fingerprint of messages list.
    Derived from OpenClaw cache-trace.ts per-stage fingerprinting.
    """
    serialized = json.dumps(messages, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _total_chars(messages: list[dict]) -> int:
    """Sum of all content lengths across messages."""
    return sum(len(m.get("content", "")) for m in messages)


class PayloadTracer:
    """
    Diagnostic tracer for judge call payloads.

    Derived from OpenClaw cache-trace.ts + anthropic-payload-log.ts.
    Creates JSONL trace records with SHA-256 fingerprints per pipeline stage.

    Usage:
        tracer = PayloadTracer()
        tracer.trace("prompt:before", messages, {"model": "Qwen3-30B"})
        tracer.trace("prompt:sanitized", sanitized_messages)
        tracer.trace("response:received", messages, {"raw_length": 1234})
    """

    def __init__(self, trace_dir: str = "logs/judge_traces") -> None:
        self.trace_dir = trace_dir
        self._enabled = os.environ.get(
            "EVOLAI_TRACE_PAYLOADS", ""
        ).lower() in ("1", "true", "yes")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def trace(
        self,
        stage: str,
        messages: list[dict],
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Record a single trace event.

        Args:
            stage: Pipeline stage name (e.g., "prompt:before", "response:received")
            messages: The chat messages at this pipeline stage
            metadata: Optional extra fields (model, miner_uid, turn, etc.)
        """
        if not self._enabled:
            return

        fingerprint = _fingerprint(messages)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "fingerprint": fingerprint,
            "message_count": len(messages),
            "total_chars": _total_chars(messages),
        }
        if metadata:
            record.update(metadata)

        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            trace_path = os.path.join(self.trace_dir, "trace.jsonl")
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            # Fire-and-forget — never block evaluation for tracing failures
            logger.debug(f"[payload-trace] Failed to write trace: {e}")

    def trace_response(
        self,
        stage: str,
        response_text: str,
        model: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Trace a judge response (not message list).
        Creates fingerprint from response text instead of messages.
        """
        if not self._enabled:
            return

        fingerprint = hashlib.sha256(response_text.encode()).hexdigest()[:16]
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "fingerprint": fingerprint,
            "response_chars": len(response_text),
            "model": model,
        }
        if metadata:
            record.update(metadata)

        try:
            os.makedirs(self.trace_dir, exist_ok=True)
            trace_path = os.path.join(self.trace_dir, "trace.jsonl")
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────────────────

_tracer: PayloadTracer | None = None


def get_tracer() -> PayloadTracer:
    """Return the module-level PayloadTracer singleton."""
    global _tracer
    if _tracer is None:
        _tracer = PayloadTracer()
    return _tracer
