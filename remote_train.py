#!/usr/bin/env python3
"""
Self-contained training script — runs on a remote GPU (RunPod).
Uploaded and executed by auto_pipeline.py via SSH.

Usage (set by auto_pipeline.py):
  HF_TOKEN=<token> MODEL_NAME=<hf_repo> python remote_train.py challenge.json

Output: prints "COMMIT_HASH:<sha>" on the last line so auto_pipeline.py
can parse the revision for on-chain registration.
"""

import json
import os
import sys
from datetime import datetime, timezone

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")

if not HF_TOKEN or not MODEL_NAME:
    raise SystemExit("HF_TOKEN and MODEL_NAME must be set as environment variables.")

challenge_path = sys.argv[1] if len(sys.argv) > 1 else "challenge.json"
with open(challenge_path, encoding="utf-8") as f:
    challenge = json.load(f)

epoch = challenge.get("epoch", "?")
print(f"Epoch: {epoch}", flush=True)

# ── Load training rows ────────────────────────────────────────────────────────

from datasets import load_dataset, Dataset

_INSTR_COLS = ("instruction", "input", "question", "prompt", "human")
_RESP_COLS  = ("response", "output", "answer", "completion", "assistant", "gpt")


def extract_sample(row: dict) -> dict | None:
    keys = {k.lower(): k for k in row.keys()}
    ik = next((keys[c] for c in _INSTR_COLS if c in keys), None)
    rk = next((keys[c] for c in _RESP_COLS  if c in keys), None)
    if ik and rk:
        return {"instruction": str(row[ik]).strip(), "response": str(row[rk]).strip()}
    for v in row.values():
        if isinstance(v, str) and v.strip():
            return {"instruction": v.strip(), "response": ""}
    return None


samples: list[dict] = []
for ds_name, ds_info in challenge.get("union", {}).items():
    indices = ds_info["indices"]
    print(f"Loading {len(indices)} rows from {ds_name} ...", flush=True)
    ds = load_dataset(ds_name, split="train")
    for row in ds.select(indices):
        s = extract_sample(dict(row))
        if s:
            samples.append(s)

print(f"Total samples: {len(samples)}", flush=True)
if not samples:
    raise SystemExit("No training samples found.")

# ── Load model ────────────────────────────────────────────────────────────────

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    device_map="auto",
)
model.config.use_cache = False
print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params", flush=True)

# ── Attach LoRA ───────────────────────────────────────────────────────────────

from peft import LoraConfig, TaskType, get_peft_model

model = get_peft_model(
    model,
    LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    ),
)
model.print_trainable_parameters()

# ── Format dataset ────────────────────────────────────────────────────────────

def format_sample(s: dict) -> str:
    if getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "user",      "content": s["instruction"]},
            {"role": "assistant", "content": s["response"]},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False)
    return f"### Human: {s['instruction']}\n\n### Assistant: {s['response']}"


train_dataset = Dataset.from_list([{"text": format_sample(s)} for s in samples])
print(f"Training dataset: {len(train_dataset)} samples", flush=True)

# ── Train ─────────────────────────────────────────────────────────────────────

from transformers import TrainingArguments
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=TrainingArguments(
        output_dir="/tmp/model_out",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
    ),
    dataset_text_field="text",
    max_seq_length=2048,
)

print("Training ...", flush=True)
trainer.train()
print("Training complete.", flush=True)

# ── Merge LoRA + save ─────────────────────────────────────────────────────────

print("Merging LoRA weights ...", flush=True)
merged = trainer.model.merge_and_unload()
merged.save_pretrained("/tmp/model_out", safe_serialization=True)
tokenizer.save_pretrained("/tmp/model_out")

# ── Push to HuggingFace ───────────────────────────────────────────────────────

from huggingface_hub import HfApi, login

login(token=HF_TOKEN, add_to_git_credential=False)
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=MODEL_NAME, repo_type="model", exist_ok=True)

tag = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")
print(f"Pushing to {MODEL_NAME} ...", flush=True)
commit_info = api.upload_folder(
    folder_path="/tmp/model_out",
    repo_id=MODEL_NAME,
    repo_type="model",
    commit_message=f"EvolAI auto-train epoch={epoch} {tag}",
)

commit_hash = getattr(commit_info, "oid", None) or tag
print(f"Pushed successfully.", flush=True)

# This line is parsed by auto_pipeline.py — must be the exact format
print(f"COMMIT_HASH:{commit_hash}", flush=True)
