#!/usr/bin/env python3
"""
EvolAI full-dataset trainer (Option B) — no validator-seed dependency.

Fine-tunes BASE_MODEL with LoRA on the entire evolai/universal_qa dataset,
merges the weights (makes the model fingerprint-unique to clear copy-detection),
pushes to your HuggingFace repo, and re-registers the new revision on-chain.

Unlike train_and_submit.py / kaggle_pipeline.py, this does NOT require validator
seeds to be committed on-chain — it trains on the full dataset directly.

Avoids `trl` entirely (uses peft + transformers.Trainer) for compatibility with
transformers 5.x. Requires `peft` (pip install peft).

  python train_full.py                    # full run (all rows, 1 epoch)
  python train_full.py --max-samples 200  # quick smoke test end-to-end
  python train_full.py --skip-register    # train + push, no on-chain step

Pin to a single GPU to avoid contention / multi-GPU DataParallel, e.g.:
  CUDA_VISIBLE_DEVICES=0 python train_full.py
"""
import argparse
import os
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
BASE_MODEL  = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET     = os.getenv("EVOLAI_DATASET", "evolai/universal_qa")
OUTPUT_DIR  = "model_output"

_INSTR_COLS = ("instruction", "input", "question", "prompt", "human")
_RESP_COLS  = ("response", "output", "answer", "completion", "assistant", "gpt")


def extract_sample(row):
    keys = {k.lower(): k for k in row.keys()}
    ik = next((keys[c] for c in _INSTR_COLS if c in keys), None)
    rk = next((keys[c] for c in _RESP_COLS if c in keys), None)
    if ik and rk:
        return {"instruction": str(row[ik]).strip(), "response": str(row[rk]).strip()}
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return {"instruction": v.strip(), "response": ""}
    return None


def build_samples(max_samples=None):
    from datasets import load_dataset
    print(f"  Loading dataset: {DATASET}")
    ds = load_dataset(DATASET, split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    samples = []
    for row in ds:
        s = extract_sample(dict(row))
        if s:
            samples.append(s)
    print(f"  Usable samples: {len(samples)}")
    if not samples:
        raise SystemExit("No usable samples extracted from dataset.")
    return samples


def train(samples, base_model, max_len=1024):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, Trainer,
        TrainingArguments, DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    print(f"  Base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def to_text(s):
        if getattr(tok, "chat_template", None):
            msgs = [{"role": "user", "content": s["instruction"]},
                    {"role": "assistant", "content": s["response"]}]
            return tok.apply_chat_template(msgs, tokenize=False)
        return f"### Human: {s['instruction']}\n\n### Assistant: {s['response']}"

    ds = Dataset.from_list([{"text": to_text(s)} for s in samples])
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                batched=True, remove_columns=["text"])

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model.config.use_cache = False

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
        lora_dropout=0.05, target_modules="all-linear", bias="none",
    ))
    model.enable_input_require_grads()  # required with gradient checkpointing + LoRA
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=use_bf16, fp16=not use_bf16,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
    )
    print("  Training ...")
    trainer.train()

    print("  Merging LoRA into base weights ...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"  Saved to {OUTPUT_DIR}/")
    return OUTPUT_DIR


def push_to_hub(local_dir, repo_id):
    from huggingface_hub import HfApi
    tag = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"  Uploading {local_dir}/ -> {repo_id} ...")
    info = api.upload_folder(folder_path=local_dir, repo_id=repo_id,
                             repo_type="model",
                             commit_message=f"EvolAI full-dataset train {tag}")
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
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--skip-register", action="store_true")
    p.add_argument("--base-model", default=BASE_MODEL)
    a = p.parse_args()

    missing = [k for k, v in {
        "HF_TOKEN": HF_TOKEN, "EVOLAI_MODEL_NAME": MODEL_NAME,
        "EVOLAI_WALLET_NAME": WALLET_NAME, "EVOLAI_HOTKEY": HOTKEY,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing .env values: {', '.join(missing)}")

    print("=" * 60)
    print("  EvolAI Full-Dataset Trainer (Option B)")
    print("=" * 60)
    print(f"  HF model  : {MODEL_NAME}")
    print(f"  Base model: {a.base_model}")
    print(f"  Dataset   : {DATASET}")
    print(f"  Track     : {TRACK} | Netuid: {NETUID} | Network: {NETWORK}")
    print()

    samples = build_samples(a.max_samples)
    out = train(samples, a.base_model)
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
    else:
        print("\n⚠ Registration failed — run manually:")
        print(f"  evolcli miner register {MODEL_NAME} "
              f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
              f"--track {TRACK} --revision {rev}")


if __name__ == "__main__":
    main()
