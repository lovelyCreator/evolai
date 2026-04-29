import argparse

import bittensor as bt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check EvolAI miner status.")
    parser.add_argument("--network", default="finney", help="Bittensor network")
    parser.add_argument("--netuid", type=int, default=47, help="Subnet netuid")
    parser.add_argument("--uid", type=int, default=55, help="Miner UID")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subtensor = bt.Subtensor(network=args.network)
    metagraph = subtensor.metagraph(args.netuid)

    if args.uid < 0 or args.uid >= len(metagraph.hotkeys):
        raise SystemExit(f"UID {args.uid} is not in subnet {args.netuid}")

    print("=" * 50)
    print("EvolAI Miner Status")
    print("=" * 50)
    print("Network:", args.network)
    print("Netuid:", args.netuid)
    print("UID:", args.uid)
    print("Hotkey:", metagraph.hotkeys[args.uid])
    print("Block:", subtensor.block)
    print("-" * 50)
    print("Stake:", float(metagraph.S[args.uid]))
    print("Rank:", float(metagraph.R[args.uid]))
    print("Trust:", float(metagraph.T[args.uid]))
    print("Consensus:", float(metagraph.C[args.uid]))
    print("Incentive:", float(metagraph.I[args.uid]))
    print("Emission:", float(metagraph.E[args.uid]))
    print("=" * 50)


if __name__ == "__main__":
    main()
