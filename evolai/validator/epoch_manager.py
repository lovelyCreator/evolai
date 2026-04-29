
from __future__ import annotations

import hashlib
import json
import logging
import random
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


TRAIN_SALT = "train"
EVAL_SALT  = "eval"


_COMMITMENT_KEY = "ve"
_COMMITMENT_VERSION = 1


@dataclass
class EpochSeed:
    validator_uid: int
    validator_hotkey: str
    epoch: int
    seed: str


@dataclass
class EpochChallenge:
    uid: int
    epoch: int
    validator_uid: int

    datasets: Dict[str, List[int]]

    @property
    def all_indices_count(self) -> int:
        return sum(len(v) for v in self.datasets.values())


def current_epoch(block: int, epoch_blocks: int) -> int:
    return block // epoch_blocks


def generate_seed() -> str:
    return secrets.token_hex(16)


def commit_epoch_seed(
    wallet,
    subtensor,
    netuid: int,
    epoch: int,
    seed: str,
) -> tuple[bool, str]:
    try:
        payload = json.dumps(
            {_COMMITMENT_KEY: {"e": epoch, "s": seed, "v": _COMMITMENT_VERSION}},
            separators=(',', ':'),
        )
        logger.info(
            f"Committing epoch seed: epoch={epoch}, "
            f"payload={len(payload)} bytes, netuid={netuid}"
        )
        result = subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            data=payload,
        )

        if hasattr(result, 'success') and not result.success:
            msg = getattr(result, 'message', str(result))
            logger.warning(f"Seed commit rejected by chain for epoch {epoch}: {msg}")
            return False, f"chain rejected: {msg}"
        logger.info(f"Committed epoch seed for epoch {epoch} ({len(payload)} bytes)")
        return True, ""
    except Exception as exc:
        logger.warning(f"Failed to commit epoch seed for epoch {epoch}: {exc}", exc_info=True)
        return False, str(exc)


def read_all_validator_seeds(
    subtensor,
    netuid: int,
    metagraph,
    current_epoch_num: int,
    max_epoch_lag: int = 1,
) -> List[EpochSeed]:
    seeds: List[EpochSeed] = []
    min_epoch = current_epoch_num - max_epoch_lag

    for uid in range(len(metagraph.hotkeys)):
        hotkey = metagraph.hotkeys[uid]
        if not hotkey:
            continue
        try:
            commit_data = subtensor.get_commitment_metadata(netuid, hotkey)
            if not commit_data:
                continue

            fields = commit_data.get("info", {}).get("fields", [])
            if not (fields and fields[0] and fields[0][0]):
                continue

            raw_entry = fields[0][0]
            raw_key = next(
                (k for k in raw_entry if k.startswith("Raw") and k[3:].isdigit()),
                None,
            )
            if raw_key is None:
                continue

            raw_bytes = bytes(raw_entry[raw_key][0])
            payload = json.loads(raw_bytes)
            inner = payload.get(_COMMITMENT_KEY)
            if inner is None:

                continue

            epoch_num = inner.get("e", -1)
            seed_str = inner.get("s", "")
            if not seed_str or epoch_num < min_epoch:
                logger.debug(
                    f"UID {uid}: skipping seed — "
                    f"epoch={epoch_num} min_accepted={min_epoch}"
                )
                continue

            seeds.append(EpochSeed(
                validator_uid=uid,
                validator_hotkey=hotkey,
                epoch=epoch_num,
                seed=seed_str,
            ))

        except Exception as exc:
            logger.debug(f"UID {uid}: could not read seed: {exc}")
            continue

    logger.info(
        f"Read {len(seeds)} validator seed(s) for epoch(s) >= {min_epoch}"
    )
    return seeds


def derive_indices(
    seed: str,
    uid: int,
    dataset_name: str,
    dataset_size: int,
    n: int,
    salt: str = EVAL_SALT,
) -> List[int]:
    key = f"{seed}:{uid}:{dataset_name}:{salt}"
    digest = hashlib.sha256(key.encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    return sorted(rng.sample(range(dataset_size), min(n, dataset_size)))


def build_eval_challenge(
    seed: str,
    validator_uid: int,
    miner_uid: int,
    epoch: int,
    active_datasets: List[str],
    dataset_sizes: Dict[str, int],
    n_eval: int,
) -> EpochChallenge:
    datasets: Dict[str, List[int]] = {}
    for ds_name in active_datasets:
        ds_size = dataset_sizes.get(ds_name)
        if not ds_size:
            logger.warning(
                f"build_eval_challenge: unknown dataset size for {ds_name!r} — skipping"
            )
            continue
        datasets[ds_name] = derive_indices(
            seed, miner_uid, ds_name, ds_size, n_eval, EVAL_SALT
        )

    return EpochChallenge(
        uid=miner_uid,
        epoch=epoch,
        validator_uid=validator_uid,
        datasets=datasets,
    )


def build_training_hint(
    seed: str,
    validator_uid: int,
    miner_uid: int,
    epoch: int,
    active_datasets: List[str],
    dataset_sizes: Dict[str, int],
    n_train: int,
) -> EpochChallenge:
    datasets: Dict[str, List[int]] = {}
    for ds_name in active_datasets:
        ds_size = dataset_sizes.get(ds_name)
        if not ds_size:
            continue
        datasets[ds_name] = derive_indices(
            seed, miner_uid, ds_name, ds_size, n_train, TRAIN_SALT
        )

    return EpochChallenge(
        uid=miner_uid,
        epoch=epoch,
        validator_uid=validator_uid,
        datasets=datasets,
    )


def epoch_eval_order(
    validator_hotkey: str,
    epoch: int,
    uids: List[int],
) -> List[int]:
    key = f"{validator_hotkey}:{epoch}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    rng = random.Random(int.from_bytes(bytes.fromhex(digest[:16]), "big"))
    shuffled = list(uids)
    rng.shuffle(shuffled)
    return shuffled
