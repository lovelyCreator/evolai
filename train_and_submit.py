#!/usr/bin/env python3
"""
EvolAI auto-train pipeline.

Steps:
  1. Resolve your miner UID from the Bittensor metagraph
  2. Fetch this epoch's challenge indices via evolcli
  3. Load those rows from evolai/universal_qa
  4. Fine-tune your model on them (LoRA → merge)
  5. Push merged model to HuggingFace
  6. Re-register on-chain with the new commit hash

Usage:
  python train_and_submit.py
  python train_and_submit.py --skip-register   # train + push only
  python train_and_submit.py --dry-run         # fetch challenge only, no training
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Config from .env ──────────────────────────────────────────────────────────

HF_TOKEN    = os.getenv("HF_TOKEN")
MODEL_NAME  = os.getenv("EVOLAI_MODEL_NAME")        # e.g. Roystar/evolai-qwen2.5-1.5b
TRACK       = os.getenv("EVOLAI_TRACK", "transformer")
NETUID      = int(os.getenv("EVOLAI_NETUID", "47"))
WALLET_NAME = os.getenv("EVOLAI_WALLET_NAME")
HOTKEY      = os.getenv("EVOLAI_HOTKEY")
NETWORK     = os.getenv("BT_NETWORK", "finney")

# Base model to fine-tune FROM (first run, or if you want a fresh start).
# Defaults to the Qwen2.5-1.5B instruct base; override via env BASE_MODEL.
BASE_MODEL  = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

OUTPUT_DIR  = "model_output"

# ── Column name detection (mirrors challenge_client.py) ───────────────────────

_INSTR_COLS = ("instruction", "input", "question", "prompt", "human")
_RESP_COLS  = ("response", "output", "answer", "completion", "assistant", "gpt")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Resolve miner UID
# ─────────────────────────────────────────────────────────────────────────────

def get_miner_uid() -> int:
    import bittensor as bt

    subtensor = bt.Subtensor(NETWORK)
    metagraph = subtensor.metagraph(netuid=NETUID)
    wallet    = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY)
    my_hotkey = wallet.hotkey.ss58_address

    for uid, hk in enumerate(metagraph.hotkeys):
        if hk == my_hotkey:
            subtensor.close()
            print(f"  UID = {uid}  ({my_hotkey[:12]}...)")
            return uid

    subtensor.close()
    raise SystemExit(
        f"Hotkey not found in metagraph (netuid={NETUID}).\n"
        f"Hotkey: {my_hotkey}\n"
        "Make sure your miner is registered on the subnet."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fetch challenge
# ─────────────────────────────────────────────────────────────────────────────

def fetch_challenge(uid: int) -> dict:
    out_path = f"challenge_uid{uid}.json"
    print(f"  Running: evolcli miner get-challenge {uid} --netuid {NETUID} -o {out_path}")

    result = subprocess.run(
        [
            "evolcli", "miner", "get-challenge", str(uid),
            "--netuid", str(NETUID),
            "--network", NETWORK,
            "-o", out_path,
        ],
        capture_output=False,
    )

    if result.returncode != 0:
        raise SystemExit("get-challenge failed — see output above.")

    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    epoch = data.get("epoch", "?")
    n_validators = data.get("validator_count", 0)
    print(f"  Epoch={epoch}, validators={n_validators}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 3. Load training rows
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sample(row: dict) -> dict | None:
    keys = {k.lower(): k for k in row.keys()}
    instr_key = next((keys[c] for c in _INSTR_COLS if c in keys), None)
    resp_key  = next((keys[c] for c in _RESP_COLS  if c in keys), None)
    if instr_key and resp_key:
        return {
            "instruction": str(row[instr_key]).strip(),
            "response":    str(row[resp_key]).strip(),
        }
    # Fallback: treat the first non-empty string field as plain text
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return {"instruction": v.strip(), "response": ""}
    return None


def load_training_data(challenge: dict) -> list[dict]:
    from datasets import load_dataset

    samples: list[dict] = []
    union = challenge.get("union", {})

    for ds_name, ds_info in union.items():
        indices = ds_info["indices"]
        print(f"  Loading {len(indices)} rows from {ds_name} ...")
        ds = load_dataset(ds_name, split="train")
        for row in ds.select(indices):
            s = _extract_sample(dict(row))
            if s:
                samples.append(s)

    print(f"  Total training samples: {len(samples)}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fine-tune (LoRA → merge)
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(samples: list[dict], base_model: str) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"  Base model : {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_sample(s: dict) -> str:
        if getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "user",      "content": s["instruction"]},
                {"role": "assistant", "content": s["response"]},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)
        return (
            f"### Human: {s['instruction']}\n\n"
            f"### Assistant: {s['response']}"
        )

    dataset = Dataset.from_list([{"text": format_sample(s)} for s in samples])
    print(f"  Dataset size: {len(dataset)} formatted samples")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32),
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    print("  Training ...")
    trainer.train()

    print("  Merging LoRA weights into base model ...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  Saved to {OUTPUT_DIR}/")

    return OUTPUT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# 5. Push to HuggingFace
# ─────────────────────────────────────────────────────────────────────────────

def push_to_hub(local_dir: str, repo_id: str) -> str:
    from huggingface_hub import HfApi

    tag = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist yet
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    print(f"  Uploading {local_dir}/ → {repo_id} ...")
    commit_info = api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"EvolAI auto-train {tag}",
    )

    # commit_info.oid is the full commit SHA on the Hub
    revision = getattr(commit_info, "oid", None) or tag
    print(f"  Pushed — revision: {revision}")
    return revision


# ─────────────────────────────────────────────────────────────────────────────
# 6. Re-register on-chain
# ─────────────────────────────────────────────────────────────────────────────

def re_register(revision: str) -> bool:
    print(f"  evolcli miner register {MODEL_NAME} --revision {revision[:16]}...")
    result = subprocess.run(
        [
            "evolcli", "miner", "register",
            MODEL_NAME,
            "--wallet-name", WALLET_NAME,
            "--hotkey",      HOTKEY,
            "--track",       TRACK,
            "--revision",    revision,
            "--netuid",      str(NETUID),
        ],
        capture_output=False,
    )
    return result.returncode == 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EvolAI auto-train + submit")
    parser.add_argument("--skip-register", action="store_true",
                        help="Train and push to HF, but skip on-chain re-registration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch challenge only; skip training, push, and register")
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help=f"Base model to fine-tune from (default: {BASE_MODEL})")
    args = parser.parse_args()

    # ── Validate env ──────────────────────────────────────────────────────────
    missing = {k: v for k, v in {
        "HF_TOKEN":            HF_TOKEN,
        "EVOLAI_MODEL_NAME":   MODEL_NAME,
        "EVOLAI_WALLET_NAME":  WALLET_NAME,
        "EVOLAI_HOTKEY":       HOTKEY,
    }.items() if not v}
    if missing:
        raise SystemExit(f"Missing .env values: {', '.join(missing.keys())}")

    print("=" * 60)
    print("  EvolAI Auto-Train Pipeline")
    print("=" * 60)
    print(f"  HF model  : {MODEL_NAME}")
    print(f"  Base model: {args.base_model}")
    print(f"  Track     : {TRACK}  |  Netuid: {NETUID}  |  Network: {NETWORK}")
    print()

    # Step 1 ── Resolve UID
    print("[1/5] Resolving miner UID ...")
    uid = get_miner_uid()
    print()

    # Step 2 ── Fetch challenge
    print("[2/5] Fetching challenge ...")
    challenge = fetch_challenge(uid)
    print()

    if args.dry_run:
        print("--dry-run: stopping after challenge fetch.")
        return

    # Step 3 ── Load data
    print("[3/5] Loading training data ...")
    samples = load_training_data(challenge)
    if not samples:
        raise SystemExit(
            "No training samples loaded.\n"
            "Validators may not have committed seeds yet — try again in a few minutes."
        )
    print()

    # Step 4 ── Fine-tune
    print("[4/5] Fine-tuning ...")
    output_dir = fine_tune(samples, base_model=args.base_model)
    print()

    # Step 5 ── Push to HuggingFace
    print("[5/5] Pushing to HuggingFace ...")
    revision = push_to_hub(output_dir, repo_id=MODEL_NAME)
    print()

    # Step 6 ── Re-register
    if args.skip_register:
        print("--skip-register: skipping on-chain registration.")
        print(f"\nTo register manually:\n"
              f"  evolcli miner register {MODEL_NAME} "
              f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
              f"--track {TRACK} --revision {revision}")
    else:
        print("[6/6] Re-registering on-chain ...")
        ok = re_register(revision)
        if ok:
            print("\n✓ Done. Validators will evaluate the new revision on the next epoch.")
        else:
            print("\n⚠ Registration returned non-zero. Check output above.")
            print(f"Manual command:\n"
                  f"  evolcli miner register {MODEL_NAME} "
                  f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
                  f"--track {TRACK} --revision {revision}")


if __name__ == "__main__":
    main()
