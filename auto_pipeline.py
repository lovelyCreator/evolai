#!/usr/bin/env python3
"""
EvolAI fully automated pipeline — runs entirely from your VPS.

  python auto_pipeline.py           # run once
  python auto_pipeline.py --loop    # run every epoch (~72 min) forever

Steps each run:
  1. Resolve your miner UID from the metagraph
  2. Fetch this epoch's challenge indices (evolcli)
  3. Train on a RunPod GPU (or local GPU if available)
  4. Push merged model to HuggingFace
  5. Re-register on-chain with the new commit hash

Requirements (one-time setup):
  - Add RUNPOD_API_KEY to .env
  - Add SSH_KEY_PATH to .env (private key matching the public key in your RunPod account)
  - pip install runpod paramiko
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Config from .env ──────────────────────────────────────────────────────────

HF_TOKEN       = os.getenv("HF_TOKEN")
MODEL_NAME     = os.getenv("EVOLAI_MODEL_NAME")
TRACK          = os.getenv("EVOLAI_TRACK", "transformer")
NETUID         = int(os.getenv("EVOLAI_NETUID", "47"))
WALLET_NAME    = os.getenv("EVOLAI_WALLET_NAME")
HOTKEY         = os.getenv("EVOLAI_HOTKEY")
NETWORK        = os.getenv("BT_NETWORK", "finney")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
SSH_KEY_PATH   = os.getenv("SSH_KEY_PATH", "~/.ssh/id_rsa")

EPOCH_SECONDS  = 72 * 60  # ~72 min per epoch; adjust if needed
RUNPOD_GPU     = os.getenv("RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 3080")
RUNPOD_IMAGE   = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


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
    raise SystemExit(
        f"Hotkey not found in metagraph (netuid={NETUID}).\n"
        "Make sure your miner is registered on the subnet."
    )


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
        data = json.load(f)

    return data, out_path


# ── Step 3a: Train locally (when VPS has a GPU) ───────────────────────────────

def has_local_gpu() -> bool:
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def train_local(challenge_path: str) -> str:
    # Import functions from train_and_submit.py
    sys.path.insert(0, str(Path(__file__).parent))
    from train_and_submit import load_training_data, fine_tune, push_to_hub

    with open(challenge_path, encoding="utf-8") as f:
        challenge = json.load(f)

    samples = load_training_data(challenge)
    if not samples:
        raise SystemExit("No training samples — validators may not have seeded yet.")

    output_dir = fine_tune(samples, base_model=MODEL_NAME)
    return push_to_hub(output_dir, repo_id=MODEL_NAME)


# ── Step 3b: Train on RunPod (when VPS has no GPU) ────────────────────────────

def train_on_runpod(challenge_path: str) -> str:
    try:
        import runpod as _rp
        import paramiko
    except ImportError as e:
        raise SystemExit(f"Missing dependency: {e}\nRun: pip install runpod paramiko")

    if not RUNPOD_API_KEY:
        raise SystemExit("RUNPOD_API_KEY is not set in .env")

    _rp.api_key = RUNPOD_API_KEY
    pod_id = None

    try:
        # ── Create pod ────────────────────────────────────────────────────────
        log(f"Creating RunPod pod ({RUNPOD_GPU}) ...")
        pod = _rp.create_pod(
            name=f"evolai-{int(time.time())}",
            image_name=RUNPOD_IMAGE,
            gpu_type_id=RUNPOD_GPU,
            cloud_type="SECURE",
            gpu_count=1,
            volume_in_gb=10,
            container_disk_in_gb=20,
            ports="22/tcp",
        )
        pod_id = pod["id"]
        log(f"Pod created: {pod_id}")

        # ── Wait for SSH port ─────────────────────────────────────────────────
        ssh_ip = ssh_port = None
        log("Waiting for pod to start ...")
        for attempt in range(60):
            time.sleep(10)
            info = _rp.get_pod(pod_id)
            for p in (info.get("runtime") or {}).get("ports") or []:
                if p.get("privatePort") == 22 and p.get("isIpPublic"):
                    ssh_ip   = p["ip"]
                    ssh_port = p["publicPort"]
                    break
            if ssh_ip:
                break
            log(f"  Still starting ... ({(attempt+1)*10}s)")

        if not ssh_ip:
            raise RuntimeError("Pod never became reachable via SSH (timeout 600s).")

        log(f"Pod ready at {ssh_ip}:{ssh_port} — waiting 15s for SSH daemon ...")
        time.sleep(15)

        # ── SSH connect ───────────────────────────────────────────────────────
        key_path = os.path.expanduser(SSH_KEY_PATH)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            ssh_ip, port=ssh_port,
            username="root",
            key_filename=key_path,
            timeout=30,
        )
        log("SSH connected")

        # ── Upload files ──────────────────────────────────────────────────────
        remote_script = Path(__file__).parent / "remote_train.py"
        sftp = ssh.open_sftp()
        sftp.put(challenge_path, "/root/challenge.json")
        sftp.put(str(remote_script), "/root/remote_train.py")
        sftp.close()
        log("Files uploaded")

        # ── Run training ──────────────────────────────────────────────────────
        cmd = (
            "pip install -q transformers==4.47.0 trl==0.12.0 peft==0.14.0 "
            "datasets huggingface_hub accelerate bitsandbytes && "
            f"HF_TOKEN={HF_TOKEN} MODEL_NAME={MODEL_NAME} "
            "python /root/remote_train.py /root/challenge.json"
        )
        log("Training started on RunPod ...")
        _, stdout, _ = ssh.exec_command(cmd, timeout=2400)

        commit_hash = None
        for line in iter(stdout.readline, ""):
            line = line.rstrip()
            print(f"  [pod] {line}", flush=True)
            if line.startswith("COMMIT_HASH:"):
                commit_hash = line.split(":", 1)[1].strip()

        exit_code = stdout.channel.recv_exit_status()
        ssh.close()

        if exit_code != 0 or not commit_hash:
            raise RuntimeError(
                f"Remote training failed (exit={exit_code}, hash={commit_hash})"
            )

        log(f"Training complete — revision: {commit_hash}")
        return commit_hash

    finally:
        if pod_id:
            try:
                _rp.terminate_pod(pod_id)
                log(f"Pod {pod_id} terminated")
            except Exception as e:
                log(f"Warning: could not terminate pod {pod_id}: {e}")


# ── Step 4: Re-register on-chain ─────────────────────────────────────────────

def re_register(revision: str) -> bool:
    log(f"Re-registering on-chain (revision: {revision[:20]}...)")
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


# ── Single run ────────────────────────────────────────────────────────────────

def run_once() -> bool:
    log("─" * 50)
    log("EvolAI Auto Pipeline")
    log("─" * 50)

    log("[1/4] Resolving miner UID ...")
    uid = get_miner_uid()
    log(f"      UID = {uid}")

    log("[2/4] Fetching challenge ...")
    challenge, challenge_path = fetch_challenge(uid)
    log(f"      epoch={challenge.get('epoch','?')}, "
        f"validators={challenge.get('validator_count', 0)}")

    log("[3/4] Training ...")
    if has_local_gpu():
        log("      Local GPU found — training locally")
        revision = train_local(challenge_path)
    else:
        log("      No local GPU — using RunPod")
        revision = train_on_runpod(challenge_path)

    log("[4/4] Registering on-chain ...")
    ok = re_register(revision)

    if ok:
        log("✓ Done!")
    else:
        log("⚠ Registration may have failed — check output above")
        log(f"  Manual command:")
        log(f"  evolcli miner register {MODEL_NAME} "
            f"--wallet-name {WALLET_NAME} --hotkey {HOTKEY} "
            f"--track {TRACK} --revision {revision}")

    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EvolAI auto pipeline")
    parser.add_argument(
        "--loop", action="store_true",
        help="Run continuously, once per epoch (~72 min)",
    )
    args = parser.parse_args()

    missing = [k for k, v in {
        "HF_TOKEN":           HF_TOKEN,
        "EVOLAI_MODEL_NAME":  MODEL_NAME,
        "EVOLAI_WALLET_NAME": WALLET_NAME,
        "EVOLAI_HOTKEY":      HOTKEY,
    }.items() if not v]
    if missing:
        raise SystemExit(f"Missing .env values: {', '.join(missing)}")

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

        elapsed  = time.time() - start
        sleep_for = max(60, EPOCH_SECONDS - elapsed)
        log(f"Sleeping {sleep_for / 60:.1f} min until next epoch ...")
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
