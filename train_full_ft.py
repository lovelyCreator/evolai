#!/usr/bin/env python3
"""
EvolAI FULL-parameter fine-tuner — aligned to the validator's actual loss.

Why this exists (vs train_full.py)
──────────────────────────────────
The subnet's reward is the *improvement* in your best loss, and it decays
~0.995 every evaluation (loss_evaluator.py: compute_reward + cumulative_reward).
A static model therefore bleeds back to zero incentive. To re-earn you must set
a NEW lower best-loss, and the validator measures loss as the cross-entropy of
the ground-truth ANSWER conditioned on the question prompt — answer tokens only,
with an empty <think></think> baseline block (loss_evaluator.py:_batched_ce /
_build_base_prompt_from_think).

train_full.py is misaligned with that: LoRA (small weight movement), and
DataCollatorForLanguageModeling computes loss over ALL tokens (question + answer).
This trainer fixes both:

  • FULL-parameter fine-tune (every weight moves → bigger loss drop, and the
    fingerprint is automatically distinct from base Qwen).
  • Answer-only loss with labels=-100 on the prompt, using the chat template +
    an empty "<think>\n\n</think>\n\n" block — mirroring the eval's base prompt.

Usage (on the VPS, pinned to one GPU):
  CUDA_VISIBLE_DEVICES=0 python train_full_ft.py --max-samples 200   # ~3min smoke test
  CUDA_VISIBLE_DEVICES=0 python train_full_ft.py                     # full run (tmux it)
  CUDA_VISIBLE_DEVICES=0 python train_full_ft.py --skip-register     # train+push only

Memory: full FT of Qwen2.5-1.5B + AdamW fits one 32GB 5090 with gradient
checkpointing at batch 1 / accum 16. Bump --batch if VRAM allows.
"""
import argparse
import os

# Reduce CUDA fragmentation — must be set before torch initializes CUDA.
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
BASE_MODEL  = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DATASET     = os.getenv("EVOLAI_DATASET", "evolai/universal_qa")
OUTPUT_DIR  = "model_output_ft"

# Empty-thinking scaffold the validator uses for its *base* CE measurement.
THINK_BLOCK = "<think>\n\n</think>\n\n"

_INSTR_COLS = ("instruction", "input", "question", "prompt", "human")
_RESP_COLS  = ("response", "output", "answer", "completion", "assistant", "gpt")


def extract_sample(row):
    keys = {k.lower(): k for k in row.keys()}
    ik = next((keys[c] for c in _INSTR_COLS if c in keys), None)
    rk = next((keys[c] for c in _RESP_COLS if c in keys), None)
    if ik and rk:
        q, a = str(row[ik]).strip(), str(row[rk]).strip()
        if q and a:
            return {"q": q, "a": a}
    return None


def build_samples(max_samples=None):
    from datasets import load_dataset
    print(f"  Loading dataset: {DATASET}")
    ds = load_dataset(DATASET, split="train")
    if max_samples:
        # Shuffle first: the dataset spans multiple domains across shards, so
        # the first N rows are NOT representative of the eval's mix.
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    samples = [s for row in ds if (s := extract_sample(dict(row)))]
    print(f"  Usable (q,a) pairs: {len(samples)}")
    if not samples:
        raise SystemExit(
            "No usable question/answer pairs extracted. Inspect the dataset "
            "columns and extend _INSTR_COLS / _RESP_COLS."
        )
    return samples


def make_dataset(samples, tok, max_len):
    """Tokenize to prompt(question)+answer with answer-only labels.

    Mirrors the validator's _batched_ce: prompt and answer are tokenized
    separately (exact prompt length), concatenated, and CE is taken on the
    answer span only (labels=-100 on the prompt).
    """
    from datasets import Dataset

    has_template = bool(getattr(tok, "chat_template", None))
    eos = tok.eos_token_id

    def build(s):
        if has_template:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": s["q"]}],
                tokenize=False, add_generation_prompt=True,
            ) + THINK_BLOCK
        else:
            prompt = f"### Human: {s['q']}\n\n### Assistant: {THINK_BLOCK}"
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(s["a"], add_special_tokens=False)["input_ids"] + [eos]
        # Truncate from the LEFT of the prompt so the answer is always intact.
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids[:]
        if len(ids) > max_len:
            overflow = len(ids) - max_len
            # never cut into the answer; cut oldest prompt tokens
            cut = min(overflow, max(0, len(p_ids) - 1))
            ids = ids[cut:][-max_len:]
            labels = labels[cut:][-max_len:]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}

    rows = [build(s) for s in samples]
    rows = [r for r in rows if any(l != -100 for l in r["labels"])]
    print(f"  Tokenized rows with non-empty answer span: {len(rows)}")
    return Dataset.from_list(rows)


class AnswerOnlyCollator:
    """Right-pad input_ids / attention_mask / labels (labels pad = -100)."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        import torch
        m = max(len(f["input_ids"]) for f in feats)
        ii, am, lb = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            ii.append(f["input_ids"] + [self.pad_id] * n)
            am.append(f["attention_mask"] + [0] * n)
            lb.append(f["labels"] + [-100] * n)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "attention_mask": torch.tensor(am, dtype=torch.long),
            "labels": torch.tensor(lb, dtype=torch.long),
        }


def train(samples, base_model, epochs, lr, batch, accum, max_len, optim="adamw_torch_fused"):
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    )

    print(f"  Base model: {base_model}")
    tok = AutoTokenizer.from_pretrained(base_model, token=HF_TOKEN)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = make_dataset(samples, tok, max_len)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        base_model, token=HF_TOKEN,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=lr,                 # full FT: ~1e-5, far lower than LoRA's 2e-4
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        bf16=use_bf16, fp16=not use_bf16,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,  # adamw_bnb_8bit cuts optimizer state ~12GB->~3GB on shared GPUs
        dataloader_num_workers=2,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=AnswerOnlyCollator(tok.pad_token_id),
    )
    print(f"  Full fine-tune: {sum(p.numel() for p in model.parameters())/1e9:.3f}B "
          f"params, all trainable, lr={lr}, epochs={epochs}")
    trainer.train()

    print(f"  Saving merged full-FT model to {OUTPUT_DIR}/ ...")
    trainer.save_model(OUTPUT_DIR)
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
                             commit_message=f"EvolAI full FT {tag}")
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
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--accum", type=int, default=16)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--optim", default="adamw_torch_fused",
                   help="adamw_torch_fused (default) or adamw_bnb_8bit "
                        "(needs bitsandbytes; saves ~9GB on a shared GPU)")
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
    print("  EvolAI FULL-parameter fine-tuner (answer-only loss)")
    print("=" * 60)
    print(f"  HF model  : {MODEL_NAME}")
    print(f"  Base model: {a.base_model}")
    print(f"  Dataset   : {DATASET}")
    print(f"  Track     : {TRACK} | Netuid: {NETUID} | Network: {NETWORK}")
    print(f"  epochs={a.epochs} lr={a.lr} batch={a.batch} accum={a.accum} "
          f"max_len={a.max_len}")
    print()

    samples = build_samples(a.max_samples)
    out = train(samples, a.base_model, a.epochs, a.lr, a.batch, a.accum,
                a.max_len, a.optim)
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
        print("  Watch: python diagnose_miner.py --uid 55")
    else:
        print("\n⚠ Registration failed — run manually:")
        print(f"  evolcli miner register {MODEL_NAME} "
              f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
              f"--track {TRACK} --revision {rev}")


if __name__ == "__main__":
    main()
