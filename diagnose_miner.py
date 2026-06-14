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

    def vec(*names):
        """Return a metagraph vector by trying several attribute names."""
        for n in names:
            v = getattr(mg, n, None)
            if v is not None:
                try:
                    return [float(x) for x in v]
                except TypeError:
                    continue
        return [0.0] * len(mg.hotkeys)

    I = vec("I", "incentive", "Incentive")
    T = vec("T", "trust", "Trust")
    E = vec("E", "emission", "Emission")
    R = vec("R", "ranks", "rank", "Rank")
    C = vec("C", "consensus", "Consensus")
    S = vec("S", "stake", "total_stake", "Stake")

    print("=" * 64)
    print(f"EvolAI netuid {args.netuid} — UID {uid}   (block {block})")
    print("=" * 64)
    print(f"  hotkey      : {mg.hotkeys[uid]}")
    print(f"  stake       : {S[uid]:.4f}")
    print(f"  trust       : {T[uid]:.6f}")
    print(f"  rank        : {R[uid]:.6f}")
    print(f"  consensus   : {C[uid]:.6f}")
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
    permits = getattr(mg, "validator_permit", None)
    if permits is None:
        permits = getattr(mg, "validator_permits", None)
    try:
        n_val = sum(1 for x in permits if bool(x))
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
    # NOTE: netuid 47 does NOT populate the on-chain `trust` column (it is 0 for
    # every UID, including the top earner). Scores are computed off-chain by the
    # owner gate (evolai-gate.hf.space) and pushed as weights. So `trust` is
    # useless as an "evaluated?" signal here — read `incentive` directly.
    print("=" * 64)
    burn_uid, burn = (top[0], I[top[0]]) if I else (None, 0.0)
    commitment_ok = committed is not None
    if I[uid] > 0:
        print(f"VERDICT: you ARE earning — incentive={I[uid]:.6f}, "
              f"emission={E[uid]:.6f}/block. Give it a tempo to compound.")
    elif not commitment_ok:
        print("VERDICT: NO on-chain commitment found — validators have nothing")
        print("  to download. Re-run `evolcli miner register`.")
    else:
        print("VERDICT: committed correctly but NOT in the paying set "
              f"(incentive=0, rank #{rank_inc}/{len(I)}).")
        if burn_uid is not None and burn >= 0.5:
            field = sum(x for x in I) - burn
            print(f"  Context: UID {burn_uid} takes {burn:.1%} of emission "
                  f"(owner/burn); the whole field shares ~{field:.1%}.")
        print("  On this subnet trust=0 for everyone, so that's NOT 'unevaluated'.")
        print("  The lever is a STRONGER model (full fine-tune / more epochs /")
        print("  better in-band base / 3.5-3.8B band) — not a config change.")
        print("  A light merged-LoRA on Qwen2.5-1.5B does not break into the set.")
    print("=" * 64)


if __name__ == "__main__":
    main()
