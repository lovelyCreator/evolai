#!/usr/bin/env python3
"""
EvolAI fully automated pipeline using Kaggle free GPU.

  python kaggle_pipeline.py           # run once
  python kaggle_pipeline.py --loop    # run every epoch (~72 min) forever

One-time setup:
  1. pip install kaggle
  2. Get API key: kaggle.com → Account → Create New API Token
     It downloads kaggle.json → place it at ~/.kaggle/kaggle.json
     chmod 600 ~/.kaggle/kaggle.json
  3. Set KAGGLE_USERNAME in .env
  4. python kaggle_pipeline.py --setup    ← creates the Kaggle dataset (run once)
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────

HF_TOKEN      = os.getenv("HF_TOKEN")
MODEL_NAME    = os.getenv("EVOLAI_MODEL_NAME")
TRACK         = os.getenv("EVOLAI_TRACK", "transformer")
NETUID        = int(os.getenv("EVOLAI_NETUID", "47"))
WALLET_NAME   = os.getenv("EVOLAI_WALLET_NAME")
HOTKEY        = os.getenv("EVOLAI_HOTKEY")
NETWORK       = os.getenv("BT_NETWORK", "finney")
KAGGLE_USER   = os.getenv("KAGGLE_USERNAME", "")
DATASET_SLUG  = os.getenv("KAGGLE_DATASET_SLUG", "evolai-data")
KERNEL_SLUG   = os.getenv("KAGGLE_KERNEL_SLUG", "evolai-train")

EPOCH_SECONDS = 72 * 60

# Working directories (git-ignored temp folders)
_DATASET_DIR = Path("_kaggle_dataset")
_KERNEL_DIR  = Path("_kaggle_kernel")
_OUTPUT_DIR  = Path("_kaggle_output")


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Step 1: Resolve miner UID ─────────────────────────────────────────────────

def get_miner_uid() -> int:
    import bittensor as bt

    subtensor = bt.Subtensor(NETWORK)
    metagraph = subtensor.metagraph(netuid=NETUID)
    wallet    = bt.wallet(name=WALLET_NAME, hotkey=HOTKEY)
    my_hotkey = wallet.hotkey.ss58_address

    for uid, hk in enumerate(metagraph.hotkeys):
        if hk == my_hotkey:
            subtensor.close()
            return uid

    subtensor.close()
    raise SystemExit("Hotkey not found in metagraph.")


# ── Step 2: Fetch challenge ───────────────────────────────────────────────────

def fetch_challenge(uid: int) -> tuple[dict, str]:
    out_path = f"challenge_uid{uid}.json"
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
        return json.load(f), out_path


# ── Step 3: Upload challenge + config to Kaggle dataset ──────────────────────

def _dataset_id() -> str:
    return f"{KAGGLE_USER}/{DATASET_SLUG}"


def _dataset_exists() -> bool:
    r = subprocess.run(
        ["kaggle", "datasets", "status", _dataset_id()],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def upload_dataset(challenge_path: str):
    _DATASET_DIR.mkdir(exist_ok=True)

    # challenge.json — epoch-specific indices
    shutil.copy(challenge_path, _DATASET_DIR / "challenge.json")

    # config.json — HF credentials (dataset is private, so this is safe)
    (_DATASET_DIR / "config.json").write_text(
        json.dumps({"hf_token": HF_TOKEN, "model_name": MODEL_NAME})
    )

    if not _dataset_exists():
        log("Creating Kaggle dataset for the first time ...")
        (_DATASET_DIR / "dataset-metadata.json").write_text(
            json.dumps({
                "title": "EvolAI Challenge Data",
                "id": _dataset_id(),
                "licenses": [{"name": "CC0-1.0"}],
                "isPrivate": True,
            }, indent=2)
        )
        subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(_DATASET_DIR), "--quiet"],
            check=True,
        )
    else:
        log("Updating Kaggle dataset ...")
        subprocess.run(
            [
                "kaggle", "datasets", "version",
                "-p", str(_DATASET_DIR),
                "-m", f"epoch-{datetime.now().strftime('%Y%m%d-%H%M')}",
                "--quiet",
            ],
            check=True,
        )


# ── Step 4: Push kernel to Kaggle ────────────────────────────────────────────

def _kernel_id() -> str:
    return f"{KAGGLE_USER}/{KERNEL_SLUG}"


def push_kernel():
    _KERNEL_DIR.mkdir(exist_ok=True)

    # Copy the training script that runs on Kaggle
    shutil.copy(
        Path(__file__).parent / "kaggle_kernel.py",
        _KERNEL_DIR / "train.py",
    )

    # Write Kaggle kernel metadata
    (_KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps({
            "id": _kernel_id(),
            "title": "EvolAI Train",
            "code_file": "train.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [_dataset_id()],
            "kernel_sources": [],
            "competition_sources": [],
        }, indent=2)
    )

    log("Pushing kernel to Kaggle ...")
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(_KERNEL_DIR)],
        check=True,
    )


# ── Step 5: Wait for kernel to finish ────────────────────────────────────────

def wait_for_kernel() -> bool:
    kid = _kernel_id()
    log(f"Waiting for kernel {kid} ...")

    for attempt in range(120):  # poll up to 60 min
        time.sleep(30)
        r = subprocess.run(
            ["kaggle", "kernels", "status", kid],
            capture_output=True, text=True,
        )
        status_line = r.stdout.strip().lower()
        log(f"  [{attempt * 30 // 60}m{(attempt * 30) % 60}s] {r.stdout.strip()}")

        if "complete" in status_line:
            return True
        if "error" in status_line or "cancel" in status_line:
            log("Kernel failed — fetching logs ...")
            subprocess.run(["kaggle", "kernels", "output", kid, "-p", str(_OUTPUT_DIR)])
            return False

    log("Timed out waiting for kernel (60 min).")
    return False


# ── Step 6: Download commit hash from kernel output ──────────────────────────

def get_commit_hash() -> str:
    kid = _kernel_id()
    _OUTPUT_DIR.mkdir(exist_ok=True)

    log("Downloading kernel output ...")
    subprocess.run(
        ["kaggle", "kernels", "output", kid, "-p", str(_OUTPUT_DIR)],
        check=True,
    )

    hash_file = _OUTPUT_DIR / "commit_hash.txt"
    if not hash_file.exists():
        raise RuntimeError(
            "commit_hash.txt not found in kernel output.\n"
            f"Check {_OUTPUT_DIR}/ for error logs."
        )

    return hash_file.read_text(encoding="utf-8").strip()


# ── Step 7: Re-register on-chain ─────────────────────────────────────────────

def re_register(revision: str) -> bool:
    log(f"Re-registering on-chain — revision: {revision[:20]}...")
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


# ── Setup (one-time) ──────────────────────────────────────────────────────────

def setup():
    """Create the Kaggle dataset for the first time."""
    log("Running first-time setup ...")

    # Create a dummy challenge to initialize the dataset
    dummy = {"uid": 0, "epoch": 0, "union": {}}
    dummy_path = "challenge_dummy.json"
    Path(dummy_path).write_text(json.dumps(dummy))

    upload_dataset(dummy_path)
    Path(dummy_path).unlink()

    log("✓ Setup complete! Kaggle dataset created.")
    log(f"  Dataset: https://www.kaggle.com/datasets/{_dataset_id()}")
    log("  Now run: python kaggle_pipeline.py")


# ── Single run ────────────────────────────────────────────────────────────────

def run_once():
    log("─" * 50)
    log("EvolAI Kaggle Pipeline")
    log("─" * 50)

    log("[1/6] Resolving miner UID ...")
    uid = get_miner_uid()
    log(f"      UID = {uid}")

    log("[2/6] Fetching challenge ...")
    challenge, challenge_path = fetch_challenge(uid)
    log(f"      epoch={challenge.get('epoch','?')}, "
        f"validators={challenge.get('validator_count', 0)}")

    log("[3/6] Uploading challenge to Kaggle ...")
    upload_dataset(challenge_path)

    log("[4/6] Pushing training kernel ...")
    push_kernel()

    log("[5/6] Waiting for Kaggle training to finish ...")
    ok = wait_for_kernel()
    if not ok:
        raise RuntimeError("Kaggle training failed or timed out.")

    revision = get_commit_hash()
    log(f"      Revision: {revision}")

    log("[6/6] Registering on-chain ...")
    ok = re_register(revision)
    if ok:
        log("✓ Done!")
    else:
        log(f"⚠ Registration failed. Manual command:")
        log(f"  evolcli miner register {MODEL_NAME} "
            f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
            f"--track {TRACK} --revision {revision}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EvolAI Kaggle pipeline")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously every epoch (~72 min)")
    parser.add_argument("--setup", action="store_true",
                        help="One-time setup: create Kaggle dataset")
    args = parser.parse_args()

    missing = [k for k, v in {
        "HF_TOKEN":           HF_TOKEN,
        "EVOLAI_MODEL_NAME":  MODEL_NAME,
        "EVOLAI_WALLET_NAME": WALLET_NAME,
        "EVOLAI_HOTKEY":      HOTKEY,
        "KAGGLE_USERNAME":    KAGGLE_USER,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing .env values: {', '.join(missing)}")

    if args.setup:
        setup()
        return

    if not args.loop:
        run_once()
        return

    log(f"Loop mode — running every {EPOCH_SECONDS // 60} min")
    while True:
        start = time.time()
        try:
            run_once()
        except SystemExit as e:
            log(f"STOPPED: {e}")
            break
        except Exception as e:
            log(f"ERROR (will retry next epoch): {e}")

        sleep_for = max(60, EPOCH_SECONDS - (time.time() - start))
        log(f"Sleeping {sleep_for / 60:.1f} min until next epoch ...")
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
