"""
Model Fingerprinting — Anti-Gaming / Copy-Detection Layer

Miners who simply re-upload (or minimally modify) another miner's model
to harvest emissions are a known Bittensor attack vector.  This module
defends against it by computing a *structural fingerprint* for every
model submitted to the subnet and rejecting models whose fingerprint
matches one already registered to a different UID.

──────────────────────────────────────────────────────────────────────────────
Attack surface covered
──────────────────────────────────────────────────────────────────────────────

1.  Identical upload — miner re-uploads an existing model under a different
    HuggingFace repo name.  Detected by exact SHA-256 weight hash match.

2.  Trivial rename — miner copies weights but renames layers or changes only
    config metadata (model_type, name strings).  Detected by architecture
    fingerprint which hashes numeric config fields, ignoring string labels.

3.  Partial fine-tune — miner applies minimal gradient steps (<<1% weight
    delta) to evade exact hash checks.  Detected by the fuzzy fingerprint:
    sample a fixed set of layer indices, bucket their L2 norms, and compare.
    Two models must differ by > FINGERPRINT_FUZZY_TOLERANCE to be considered
    distinct.

──────────────────────────────────────────────────────────────────────────────
Fingerprint composition
──────────────────────────────────────────────────────────────────────────────

  exact_hash        — SHA-256 of sampled raw weight bytes (fixed seed layer
                      selection).  Matches perfectly identical models even
                      with different repo names.

  arch_hash         — SHA-256 of numeric architecture config values (hidden
                      size, layers, heads, intermediate size, vocab size …).
                      Catches config-only copies that swap architecture names
                      but keep the same structure.

  fuzzy_vector      — List[float]: L2 norm of sampled parameter tensors,
                      quantised to FINGERPRINT_BUCKET_COUNT buckets.
                      Used for near-copy detection via cosine similarity.

  param_count       — Total trainable parameter count.  Cheap invariant.

  layer_names_hash  — SHA-256 of the sorted list of parameter names (not
                      weights).  Detects obvious layer renaming.

──────────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────────

    from .model_fingerprint import compute_model_fingerprint, fingerprints_collide

    fp = compute_model_fingerprint(model, config)
    collision, reason = fingerprints_collide(fp, existing_fp)
    if collision:
        logger.warning(f"Copy detected: {reason}")
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public data-class
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ModelFingerprint:
    """Portable fingerprint that can be JSON-serialised and stored in the registry."""

    exact_hash: str                     # SHA-256 of deterministically sampled weight bytes
    arch_hash: str                      # SHA-256 of numeric architecture config values
    layer_names_hash: str               # SHA-256 of sorted parameter names
    param_count: int                    # Total trainable parameters
    fuzzy_vector: list[float] = field(default_factory=list)  # Quantised norm vector

    # ── convenience ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "exact_hash": self.exact_hash,
            "arch_hash": self.arch_hash,
            "layer_names_hash": self.layer_names_hash,
            "param_count": self.param_count,
            "fuzzy_vector": self.fuzzy_vector,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelFingerprint":
        return cls(
            exact_hash=d["exact_hash"],
            arch_hash=d["arch_hash"],
            layer_names_hash=d["layer_names_hash"],
            param_count=d["param_count"],
            fuzzy_vector=d.get("fuzzy_vector", []),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (mirrors config.py pattern — defaults can be overridden by env)
# ──────────────────────────────────────────────────────────────────────────────

import os as _os

_SEED: int = 42  # Hardcoded — must be identical on every validator.
# Number of parameter tensors to sample for the exact hash
_SAMPLE_N: int = int(_os.environ.get("FINGERPRINT_SAMPLE_N", 30))
# Maximum bytes to read from each sampled tensor
_MAX_BYTES_PER_TENSOR: int = int(_os.environ.get("FINGERPRINT_MAX_BYTES_PER_TENSOR", 4096))
# Number of buckets in the fuzzy norm vector
_BUCKET_COUNT: int = int(_os.environ.get("FINGERPRINT_BUCKET_COUNT", 64))
# Cosine similarity above this → near-copy
_FUZZY_THRESHOLD: float = float(_os.environ.get("FINGERPRINT_FUZZY_THRESHOLD", 0.995))
# Max elements read per tensor when computing the fuzzy L2 norm.
# Caps work at 32 768 × 4 bytes = 128 KB per tensor regardless of model size
# so a 30B model with 8192×8192 projection layers doesn't cause a multi-second stall.
_NORM_SLICE_ELEMENTS: int = int(_os.environ.get("FINGERPRINT_NORM_SLICE_ELEMENTS", 32_768))


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _numeric_config_bytes(config: Any) -> bytes:
    """
    Extract numeric fields from a HuggingFace config object and serialise
    them deterministically.  String fields (model_type, architectures, etc.)
    are intentionally excluded so that superficial name changes cannot evade
    the hash.
    """
    if config is None:
        return b""

    numeric_items: list[tuple[str, Any]] = []
    cfg_dict: dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)

    for key, val in sorted(cfg_dict.items()):
        if isinstance(val, (int, float, bool)):
            numeric_items.append((key, val))
        elif isinstance(val, (list, tuple)) and all(isinstance(v, (int, float)) for v in val):
            numeric_items.append((key, list(val)))

    # Stable bytes representation
    import json
    return json.dumps(numeric_items, sort_keys=True).encode()


def _sample_layer_indices(all_names: list[str], n: int, seed: int) -> list[int]:
    """Deterministically sample *n* indices from *all_names*."""
    rng = random.Random(seed)
    indices = list(range(len(all_names)))
    return sorted(rng.sample(indices, min(n, len(indices))))


def _tensor_bytes(tensor: torch.Tensor, max_bytes: int) -> bytes:
    """Return the raw bytes of *tensor* up to *max_bytes*, on CPU.

    Always calls ``.cpu()`` so the function is safe even if the caller
    passes a CUDA tensor.  No float32 cast — hashes the native dtype bytes
    directly, which is both faster and memory-efficient for fp16 models.
    """
    t = tensor.detach().cpu().contiguous()
    raw = t.numpy().tobytes()
    return raw[:max_bytes]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Uses numpy when available (10-50× faster than pure Python for large
    vectors), with a pure-Python fallback for environments without numpy.
    Vectors are pre-normalised by compute_model_fingerprint(), so the dot
    product equals cosine similarity directly — no magnitude division needed.
    """
    if len(a) != len(b) or not a:
        return 0.0
    try:
        import numpy as _np
        # Since both vectors are already unit-normalised, dot == cosine sim.
        return float(_np.dot(a, b))
    except ImportError:
        # Pure-Python fallback — vectors are unit-normalised, so just dot.
        return sum(x * y for x, y in zip(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def compute_model_fingerprint(
    model: torch.nn.Module,
    config: Any = None,
    seed: int = _SEED,
    sample_n: int = _SAMPLE_N,
    max_bytes_per_tensor: int = _MAX_BYTES_PER_TENSOR,
    bucket_count: int = _BUCKET_COUNT,
    norm_slice_elements: int = _NORM_SLICE_ELEMENTS,
) -> ModelFingerprint:
    """
    Compute a ``ModelFingerprint`` for *model*.

    Performance notes (30B model, CPU, float16)
    ────────────────────────────────────────────
    • Models are **always validated one at a time** — this function runs once
      per miner inside ``ModelValidator.validate_model()``, while the CPU-loaded
      model is still in memory.  It does NOT run concurrently across miners.
    • ``check_copy_gaming()`` never loads other miners' models — it only scans a
      small in-memory JSON dict of stored fingerprint records (≤ 512 entries for
      256 UIDs × 2 tracks).  That scan is <1 ms via ``numpy.dot``.
    • The fingerprint itself takes ~50–200 ms total:
        - exact_hash:   30 tensors × 4 096 bytes  → trivial
        - arch_hash:    JSON of numeric config fields → trivial
        - fuzzy norms:  64 tensors, each sliced to *norm_slice_elements* elements
                        (default 32 768 = 128 KB) → capped cost regardless of
                        whether the sampled tensor is 1 KB or 1 GB.
    • Strictly CPU-only: all tensor reads use ``.detach().cpu()`` and the
      entire function runs inside ``torch.no_grad()``.  Passing a GPU model
      is safe — tensors are pulled to CPU lazily at point of use.

    Args:
        model:                Loaded PyTorch model (eval mode; CPU or GPU).
        config:               HuggingFace config object (optional but recommended).
        seed:                 RNG seed for deterministic layer sampling.
        sample_n:             Number of parameter tensors to sample for exact hash.
        max_bytes_per_tensor: Maximum bytes to hash per tensor (exact hash).
        bucket_count:         Number of tensors sampled for the fuzzy norm vector.
        norm_slice_elements:  Max elements read per tensor when computing L2 norms.
                              Caps per-tensor work; does not affect hash accuracy.

    Returns:
        ModelFingerprint
    """
    with torch.no_grad():
        return _compute_fingerprint_impl(
            model=model,
            config=config,
            seed=seed,
            sample_n=sample_n,
            max_bytes_per_tensor=max_bytes_per_tensor,
            bucket_count=bucket_count,
            norm_slice_elements=norm_slice_elements,
        )


def _compute_fingerprint_impl(
    model: torch.nn.Module,
    config: Any,
    seed: int,
    sample_n: int,
    max_bytes_per_tensor: int,
    bucket_count: int,
    norm_slice_elements: int,
) -> ModelFingerprint:
    """Inner implementation — called inside torch.no_grad() by compute_model_fingerprint."""
    # ── Guard: warn if any parameter is on GPU (tensors pulled to CPU below) ─
    _devices = {p.device.type for p in model.parameters()}
    if "cuda" in _devices:
        logger.warning(
            "compute_model_fingerprint: model has CUDA parameters — "
            "all tensors will be pulled to CPU for fingerprinting. "
            "Consider loading the model with device_map='cpu' before calling this."
        )

    # ── Single-pass parameter collection (CPU, no grad) ──────────────────────
    # Store .detach().cpu() references so every downstream access is safe
    # and consistent regardless of where the model was loaded.
    # .numel() is a metadata call — no tensor data is read here.
    all_names: list[str] = []
    param_dict: dict[str, torch.Tensor] = {}
    total_params: int = 0
    for name, p in model.named_parameters():
        all_names.append(name)
        param_dict[name] = p.detach().cpu()   # <── always CPU from here on
        total_params += p.numel()

    # ── Layer names hash ─────────────────────────────────────────────────────
    sorted_names = sorted(all_names)
    layer_names_hash = _sha256_bytes("\n".join(sorted_names).encode())

    # ── Exact weight hash (deterministic sample) ─────────────────────────────
    sampled_indices = _sample_layer_indices(all_names, sample_n, seed)
    sampled_names = [all_names[i] for i in sampled_indices]

    hasher = hashlib.sha256()
    for name in sampled_names:
        tensor = param_dict[name]   # already CPU
        raw = _tensor_bytes(tensor, max_bytes_per_tensor)
        hasher.update(name.encode())
        hasher.update(raw)
    exact_hash = hasher.hexdigest()

    # ── Architecture hash ─────────────────────────────────────────────────────
    arch_bytes = _numeric_config_bytes(config)
    arch_hash = _sha256_bytes(arch_bytes)

    # ── Fuzzy norm vector ─────────────────────────────────────────────────────
    # Sample *bucket_count* tensors (different seed to avoid correlation with exact hash)
    fuzzy_indices = _sample_layer_indices(all_names, bucket_count, seed + 1)
    fuzzy_names = [all_names[i] for i in fuzzy_indices]

    fuzzy_vector: list[float] = []
    for name in fuzzy_names:
        tensor = param_dict[name]   # already CPU
        # Slice to *norm_slice_elements* before computing norm.
        # 32 768 elements × 2 bytes (fp16) = 64 KB max read per tensor.
        # A 30B attention projection (8192×8192, 128 MB) costs the same as a
        # tiny bias.  The sliced norm is a stable proxy: copies share weights
        # exactly, while fine-tuned variants diverge smoothly.
        flat = tensor.flatten()[:norm_slice_elements]       # still CPU
        norm = float(flat.to(torch.float32).norm().item())  # no CUDA op
        fuzzy_vector.append(norm)

    # Normalise so that scale differences do not affect cosine similarity
    mag = math.sqrt(sum(v * v for v in fuzzy_vector))
    if mag > 0:
        fuzzy_vector = [v / mag for v in fuzzy_vector]

    logger.debug(
        f"Fingerprint computed: params={total_params:,} "
        f"exact={exact_hash[:12]}… arch={arch_hash[:12]}…"
    )

    return ModelFingerprint(
        exact_hash=exact_hash,
        arch_hash=arch_hash,
        layer_names_hash=layer_names_hash,
        param_count=total_params,
        fuzzy_vector=fuzzy_vector,
    )


def fingerprints_collide(
    new_fp: ModelFingerprint,
    existing_fp: ModelFingerprint,
    fuzzy_threshold: float = _FUZZY_THRESHOLD,
) -> tuple[bool, str]:
    """
    Determine whether *new_fp* is a copy of *existing_fp*.

    Checks are applied in order from cheapest to most expensive:

    1. exact_hash match  → identical model
    2. arch_hash + layer_names_hash + param_count match → structural clone
    3. fuzzy cosine similarity ≥ *fuzzy_threshold* → near-copy (minor fine-tune)

    Returns:
        (collision: bool, reason: str)
    """
    # 1 — Identical weights
    if new_fp.exact_hash == existing_fp.exact_hash:
        return True, "exact_weight_copy"

    # 2 — Same architecture + layer layout + param count (config-level copy)
    if (
        new_fp.arch_hash == existing_fp.arch_hash
        and new_fp.layer_names_hash == existing_fp.layer_names_hash
        and new_fp.param_count == existing_fp.param_count
    ):
        return True, "structural_clone"

    # 3 — Near-copy via cosine similarity of norm vectors
    if new_fp.fuzzy_vector and existing_fp.fuzzy_vector:
        sim = _cosine_similarity(new_fp.fuzzy_vector, existing_fp.fuzzy_vector)
        if sim >= fuzzy_threshold:
            return True, f"near_copy_cosine={sim:.4f}"

    return False, ""
