"""
Full incentive diagnostic for an EvolAI (netuid 47) miner.

Run on the VPS where bittensor is installed:

    python diagnose_miner.py --uid 55

It answers ONE question: *why is incentive still zero?* by separating the two
possible worlds —

  (A) NOT BEING EVALUATED  — trust=0, stale last_update, or the on-chain
      commitment points at the wrong model/revision / a non-public HF repo.
  (B) EVALUATED BUT NOT COMPETITIVE — trust>0 but you rank near the bottom of
      your size band, so normalized weight rounds to ~0 emission.

The fix is completely different for each, so this tells you which one you're in.
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="finney")
    ap.add_argument("--netuid", type=int, default=47)
    ap.add_argument("--uid", type=int, default=55)
    ap.add_argument("--hf-model", default="Roystar/evolai-qwen2.5-1.5b",
                    help="HF repo to verify is public / matches commitment")
    args = ap.parse_args()

    import bittensor as bt

    st = bt.Subtensor(network=args.network)
    mg = st.metagraph(args.netuid)
    block = st.block
    uid = args.uid

    if uid >= len(mg.hotkeys):
        raise SystemExit(f"UID {uid} not in subnet {args.netuid}")

    I = [float(x) for x in mg.I]
    T = [float(x) for x in mg.T]
    E = [float(x) for x in mg.E]

    print("=" * 64)
    print(f"EvolAI netuid {args.netuid} — UID {uid}   (block {block})")
    print("=" * 64)
    print(f"  hotkey      : {mg.hotkeys[uid]}")
    print(f"  stake       : {float(mg.S[uid]):.4f}")
    print(f"  trust       : {T[uid]:.6f}")
    print(f"  rank        : {float(mg.R[uid]):.6f}")
    print(f"  consensus   : {float(mg.C[uid]):.6f}")
    print(f"  incentive   : {I[uid]:.6f}")
    print(f"  emission    : {E[uid]:.6f}")
    try:
        lu = int(mg.last_update[uid])
        print(f"  last_update : block {lu}  ({block - lu} blocks ago)")
    except Exception:
        pass
    try:
        print(f"  active      : {bool(mg.active[uid])}")
    except Exception:
        pass

    # ── On-chain commitment: what model/revision are validators told to fetch? ──
    print("-" * 64)
    print("On-chain commitment (what validators actually download):")
    committed = None
    try:
        raw = st.get_commitment(args.netuid, uid)
        if not raw:
            print("  *** EMPTY — no commitment found for this UID. ***")
            print("  Validators have nothing to evaluate -> re-run register.")
        else:
            try:
                from evolai.utils.metadata import decompress_metadata
                committed = decompress_metadata(raw)
                print(f"  {committed}")
            except Exception:
                print(f"  raw: {raw!r}")
    except Exception as e:
        print(f"  (could not read commitment: {e})")

    # ── HF repo sanity: public? does committed revision resolve? ────────────────
    print("-" * 64)
    print(f"HuggingFace check for {args.hf_model}:")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(args.hf_model)  # raises if private/missing
        print(f"  public      : True   (last modified {info.lastModified})")
        print(f"  latest sha  : {info.sha}")
        if committed:
            for track in ("transformer", "mamba2"):
                if track in committed:
                    rev = committed[track]["revision"]
                    match = "MATCHES latest" if rev in (info.sha, "main") else \
                            "DOES NOT obviously match latest sha — verify!"
                    print(f"  committed rev ({track}): {rev}  -> {match}")
    except Exception as e:
        print(f"  *** HF check FAILED: {e}")
        print("  If this is a 401/404, the repo is private or renamed ->")
        print("  validators score 0. Make it public and re-register.")

    # ── Subnet context: are validators alive, and where do you rank? ───────────
    print("-" * 64)
    nonzero_inc = sum(1 for x in I if x > 0)
    nonzero_trust = sum(1 for x in T if x > 0)
    # validators = UIDs with validator_permit / vtrust
    try:
        permits = [bool(x) for x in mg.validator_permit]
        n_val = sum(permits)
    except Exception:
        n_val = "?"
    print(f"  validators w/ permit : {n_val}")
    print(f"  UIDs with trust>0    : {nonzero_trust} / {len(T)}")
    print(f"  UIDs with incentive>0: {nonzero_inc} / {len(I)}")

    # your rank among miners by incentive and trust
    rank_inc = sorted(range(len(I)), key=lambda i: I[i], reverse=True).index(uid) + 1
    rank_trust = sorted(range(len(T)), key=lambda i: T[i], reverse=True).index(uid) + 1
    print(f"  your incentive rank  : #{rank_inc} of {len(I)}")
    print(f"  your trust rank      : #{rank_trust} of {len(T)}")

    top = sorted(range(len(I)), key=lambda i: I[i], reverse=True)[:5]
    print("  top-5 incentive UIDs : " +
          ", ".join(f"{i}={I[i]:.4f}" for i in top))

    # ── Verdict ────────────────────────────────────────────────────────────────
    print("=" * 64)
    if T[uid] <= 0:
        print("VERDICT: scenario (A) — NOT BEING EVALUATED (trust=0).")
        print("  -> Check the commitment + HF lines above. Most likely the")
        print("     committed revision/model is wrong, the HF repo isn't public,")
        print("     or no validator has reached you yet. Fix + re-register.")
    elif I[uid] <= 0:
        print("VERDICT: scenario (B) — evaluated (trust>0) but score too low to")
        print("  earn emission. You're being graded; the model just isn't")
        print("  competitive in its size band. -> train a stronger model")
        print("  (more epochs / full fine-tune / better in-band base), re-push,")
        print("  re-register. A light LoRA pass on Qwen2.5-1.5B rarely earns here.")
    else:
        print(f"VERDICT: you ARE earning — incentive={I[uid]:.6f}, "
              f"emission={E[uid]:.6f}/block. Give it a tempo to compound.")
    print("=" * 64)


if __name__ == "__main__":
    main()
