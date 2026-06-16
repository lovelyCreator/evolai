#!/usr/bin/env python3
"""
EvolAI base-model probe — push a (reasoning) base model AS-IS and register it.

Why: the live scoring (progress_tracker.py) gives 25% weight to `think_gain` —
how much lower the answer CE is WITH `<think>` reasoning than without. A
non-reasoning model (Qwen2.5) scores 0 there no matter how it's trained. This
script commits a native reasoning model (default Qwen3-1.7B, in the 1.5-1.8B
band) with NO fine-tuning, so we can see whether think_gain alone lifts UID 55
off zero before investing in reasoning-preserving training + iteration.

NOTE: we deliberately do NOT fine-tune here. train_full_ft.py's answer-only +
empty-<think> recipe would teach the model NOT to think, destroying the very
think_gain we're switching for.

Usage (VPS, one GPU):
  CUDA_VISIBLE_DEVICES=1 python push_model.py                       # Qwen3-1.7B, push+register
  CUDA_VISIBLE_DEVICES=1 python push_model.py --skip-register       # push only
  CUDA_VISIBLE_DEVICES=1 python push_model.py --base-model <hf_id>  # any in-band base
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

HF_TOKEN    = os.getenv("HF_TOKEN")
MODEL_NAME  = os.getenv("EVOLAI_MODEL_NAME")
TRACK       = os.getenv("EVOLAI_TRACK", "transformer")
NETUID      = int(os.getenv("EVOLAI_NETUID", "47"))
WALLET_NAME = os.getenv("EVOLAI_WALLET_NAME")
HOTKEY      = os.getenv("EVOLAI_HOTKEY")
NETWORK     = os.getenv("BT_NETWORK", "finney")
OUTPUT_DIR  = "model_probe"

# Valid parameter bands enforced by the validator (config.py VALID_PARAM_RANGES_B).
VALID_BANDS = [(0.45, 0.48), (1.5, 1.8), (3.5, 3.8)]


def fetch_and_save(base_model):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Downloading base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )

    n_b = sum(p.numel() for p in model.parameters()) / 1e9
    band = next(((lo, hi) for lo, hi in VALID_BANDS if lo <= n_b <= hi), None)
    print(f"  Parameters: {n_b:.3f}B", end="  ")
    if band is None:
        bands = ", ".join(f"{lo:.2f}-{hi:.2f}B" for lo, hi in VALID_BANDS)
        raise SystemExit(
            f"\n  ✗ {n_b:.3f}B is OUT OF BAND. Must be one of: {bands}. "
            "Pick a different --base-model."
        )
    print(f"-> in band {band[0]:.2f}-{band[1]:.2f}B ✓")

    has_think = "<think>" in (getattr(tok, "chat_template", "") or "")
    print(f"  chat template references <think>: {has_think} "
          f"({'reasoning model — think_gain possible' if has_think else 'NOT a reasoning template — think_gain likely 0'})")

    print(f"  Saving to {OUTPUT_DIR}/ ...")
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUTPUT_DIR)
    return OUTPUT_DIR


def push_to_hub(local_dir, repo_id):
    from huggingface_hub import HfApi
    tag = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"  Uploading {local_dir}/ -> {repo_id} ...")
    info = api.upload_folder(folder_path=local_dir, repo_id=repo_id,
                             repo_type="model",
                             commit_message=f"EvolAI base-model probe {tag}")
    rev = getattr(info, "oid", None) or tag
    print(f"  Pushed — revision: {rev}")
    return rev


def re_register(revision):
    print(f"  evolcli miner register {MODEL_NAME} --revision {revision[:16]}...")
    r = subprocess.run([
        "evolcli", "miner", "register", MODEL_NAME,
        "--wallet-name", WALLET_NAME, "--hotkey", HOTKEY,
        "--track", TRACK, "--revision", revision, "--netuid", str(NETUID),
    ])
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--skip-register", action="store_true")
    a = p.parse_args()

    missing = [k for k, v in {
        "HF_TOKEN": HF_TOKEN, "EVOLAI_MODEL_NAME": MODEL_NAME,
        "EVOLAI_WALLET_NAME": WALLET_NAME, "EVOLAI_HOTKEY": HOTKEY,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing .env values: {', '.join(missing)}")

    print("=" * 60)
    print("  EvolAI base-model probe (no fine-tune)")
    print("=" * 60)
    print(f"  HF repo   : {MODEL_NAME}")
    print(f"  Base model: {a.base_model}")
    print(f"  Track     : {TRACK} | Netuid: {NETUID} | Network: {NETWORK}")
    print()

    out = fetch_and_save(a.base_model)
    rev = push_to_hub(out, MODEL_NAME)

    if a.skip_register:
        print("\n--skip-register: done (not registered).")
        print(f"Manual: evolcli miner register {MODEL_NAME} "
              f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
              f"--track {TRACK} --revision {rev}")
        return

    print("  Re-registering on-chain ...")
    if re_register(rev):
        print("\n✓ Done. Validators will evaluate the new revision next epoch.")
        print("  Watch ~10 epochs: python diagnose_miner.py --uid 55")
        print("  Looking for: any nonzero incentive => think_gain is helping.")
    else:
        print("\n⚠ Registration failed — run manually:")
        print(f"  evolcli miner register {MODEL_NAME} "
              f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
              f"--track {TRACK} --revision {rev}")


if __name__ == "__main__":
    main()
