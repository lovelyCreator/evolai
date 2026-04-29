"""
Streaming Generation — Ollama NDJSON Driver

Derived from OpenClaw src/agents/ollama-stream.ts.
Provides async streaming of miner model responses via Ollama's
native /api/chat endpoint (NDJSON line-delimited JSON).

Also supports vLLM /v1/chat/completions SSE streaming as fallback.
"""

from __future__ import annotations

import re
import json
import asyncio
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import aiohttp

from .config import (
    OLLAMA_CHAT_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    PER_RESPONSE_TOKEN_LIMIT,
    VLLM_MINER_BASE_URL,
    LOCAL_API_KEY,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Stream Accumulator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamAccumulator:
    """
    Accumulates content and tool_calls across NDJSON chunks.
    Derived from OpenClaw ollama-stream.ts createOllamaStreamFn():
      - content: accumulated across ALL chunks (intermediate + final)
      - tool_calls: collected from intermediate done=False chunks ONLY
        (the final done=True chunk does NOT repeat tool_calls)
      - reasoning: fallback for Qwen3 models that emit thinking tokens
    """
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    prompt_eval_count: int = 0    # input tokens reported by Ollama
    eval_count: int = 0           # output tokens reported by Ollama
    done_reason: str = ""


def _guard_integer_literals(raw: str) -> str:
    """
    Python equivalent of OpenClaw's quoteUnsafeIntegerLiterals() (ollama-stream.ts).
    Ollama sometimes emits large integer tool_call IDs that exceed JS/Python
    float64 precision. Wrap bare integers > 15 digits in quotes before json.loads().
    """
    return re.sub(r'(?<!["\w])(\d{16,})(?!["\w])', r'"\1"', raw)


# ──────────────────────────────────────────────────────────────────────────────
# NDJSON parser (Ollama native /api/chat)
# ──────────────────────────────────────────────────────────────────────────────

async def parse_ndjson_stream(
    response: aiohttp.ClientResponse,
) -> AsyncIterator[dict]:
    """
    Async generator that yields parsed JSON objects from Ollama's NDJSON stream.
    Each line in the response body is one JSON object terminated by newline.
    """
    buffer = b""
    async for chunk in response.content.iter_any():
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            text = _guard_integer_literals(text)
            try:
                yield json.loads(text)
            except json.JSONDecodeError as e:
                logger.warning(f"[ndjson] skipping malformed line: {e}")


def convert_to_ollama_messages(messages: list[dict]) -> list[dict]:
    """
    Convert OpenAI-style chat messages to Ollama's message format.
    Ollama uses the same structure but doesn't support 'name' field.
    """
    ollama_msgs = []
    for msg in messages:
        ollama_msgs.append({
            "role": msg["role"],
            "content": msg.get("content", ""),
        })
    return ollama_msgs


# ──────────────────────────────────────────────────────────────────────────────
# Main streaming driver — Ollama
# ──────────────────────────────────────────────────────────────────────────────

async def stream_miner_response(
    messages: list[dict],
    model: str,
    *,
    ollama_url: str = OLLAMA_CHAT_URL,
    num_ctx: int = OLLAMA_NUM_CTX,
    num_predict: int = OLLAMA_NUM_PREDICT,
    token_budget: int = PER_RESPONSE_TOKEN_LIMIT,
    timeout_s: float = 180.0,
    on_chunk: callable | None = None,
) -> StreamAccumulator:
    """
    Stream a miner model response from Ollama's native /api/chat endpoint.
    
    Key design decisions (from OpenClaw ollama-stream.ts):
      1. num_ctx MUST be set explicitly — Ollama default is 4096, far too small.
      2. tool_calls arrive in intermediate done:false chunks ONLY.
      3. Qwen3 models may emit empty content — fall back to message.reasoning.
      4. Enforce token_budget: stop accumulation mid-stream if exceeded.
    
    Args:
        messages: Chat messages in OpenAI format.
        model: Ollama model tag (e.g. "mistral", "llama3.1").
        ollama_url: Ollama /api/chat endpoint URL.
        num_ctx: Context window for Ollama (set explicitly!).
        num_predict: Max output tokens (Ollama's name for max_tokens).
        token_budget: Stop accumulating after this many output tokens.
        timeout_s: HTTP timeout in seconds.
        on_chunk: Optional callback invoked per streaming chunk for live
                  dashboard visibility (design doc §Streaming Judge Responses).
                  Signature: on_chunk(content_delta: str, acc: StreamAccumulator)
    
    Returns:
        StreamAccumulator with accumulated content and token counts.
    """
    acc = StreamAccumulator()
    ollama_msgs = convert_to_ollama_messages(messages)

    payload = {
        "model": model,
        "messages": ollama_msgs,
        "stream": True,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(ollama_url, json=payload) as resp:
            resp.raise_for_status()

            async for chunk in parse_ndjson_stream(resp):
                msg = chunk.get("message", {})
                content_delta = msg.get("content", "")
                reasoning_delta = msg.get("reasoning", "")
                is_done = chunk.get("done", False)

                # Accumulate content
                if content_delta:
                    acc.content += content_delta
                    # Live dashboard callback (design doc §Streaming Judge Responses)
                    if on_chunk is not None:
                        try:
                            on_chunk(content_delta, acc)
                        except Exception:
                            pass  # Fire-and-forget — never block streaming
                if reasoning_delta:
                    acc.reasoning += reasoning_delta

                # tool_calls: intermediate chunks only (done=False)
                if not is_done:
                    tool_calls = msg.get("tool_calls")
                    if tool_calls:
                        acc.tool_calls.extend(tool_calls)

                # Final chunk: grab token counts
                if is_done:
                    acc.prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    acc.eval_count = chunk.get("eval_count", 0)
                    acc.done_reason = chunk.get("done_reason", "stop")

                # Budget enforcement: stop early if output tokens exceeded
                if acc.eval_count > token_budget > 0:
                    logger.warning(
                        f"[stream] token budget exceeded "
                        f"({acc.eval_count}/{token_budget}), stopping"
                    )
                    break

    # Qwen3 fallback: if content is empty but reasoning exists
    if not acc.content.strip() and acc.reasoning.strip():
        logger.debug("[stream] Qwen3 fallback: using reasoning as content")
        acc.content = acc.reasoning

    return acc


# ──────────────────────────────────────────────────────────────────────────────
# vLLM SSE streaming (alternative to Ollama)
# ──────────────────────────────────────────────────────────────────────────────

async def stream_miner_response_vllm(
    messages: list[dict],
    model: str,
    *,
    base_url: str = VLLM_MINER_BASE_URL,
    max_tokens: int = PER_RESPONSE_TOKEN_LIMIT,
    temperature: float = 0.0,
    timeout_s: float = 180.0,
) -> StreamAccumulator:
    """
    Stream a miner model response via vLLM's /v1/chat/completions SSE.
    
    This is an alternative to the Ollama NDJSON driver for validators
    that use vLLM for miner model serving.
    """
    acc = StreamAccumulator()

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {LOCAL_API_KEY}"}
    url = f"{base_url}/chat/completions"

    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()

            async for line in resp.content:
                text = line.decode("utf-8", errors="replace").strip()
                if not text or text == "data: [DONE]":
                    continue
                if text.startswith("data: "):
                    text = text[6:]
                try:
                    chunk = json.loads(text)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content_delta = delta.get("content", "")
                if content_delta:
                    acc.content += content_delta

                # Token usage (vLLM reports in final chunk)
                usage = chunk.get("usage")
                if usage:
                    acc.prompt_eval_count = usage.get("prompt_tokens", 0)
                    acc.eval_count = usage.get("completion_tokens", 0)

                finish = choices[0].get("finish_reason")
                if finish:
                    acc.done_reason = finish

    return acc


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace inline inference (no server required)
# ──────────────────────────────────────────────────────────────────────────────

async def stream_miner_response_hf(
    messages: list[dict],
    model,
    tokenizer,
    *,
    max_tokens: int = PER_RESPONSE_TOKEN_LIMIT,
    temperature: float = 0.7,
) -> StreamAccumulator:
    """
    Run HuggingFace transformers inference inline (in a thread-pool executor).

    Used when the miner model is already loaded in-process via HF transformers
    rather than served by a vLLM or Ollama subprocess.
    The call is offloaded to a thread so it does not block the async event loop.
    """
    import concurrent.futures

    acc = StreamAccumulator()

    def _generate() -> str:
        import torch

        chat = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat, return_tensors="pt", truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (skip the prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        text = await loop.run_in_executor(pool, _generate)

    acc.content = text
    acc.eval_count = len(text.split())   # approximate token count
    acc.done_reason = "stop"
    return acc
