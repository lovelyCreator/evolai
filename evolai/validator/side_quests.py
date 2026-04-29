
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SideQuest:
    question: str
    answer: str
    answer_int: int


def _addition(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(100_000_000, 2_000_000_000)
    b = rng.randint(100_000_000, 2_000_000_000)
    return f"What is {a} + {b}?", a + b


def _subtraction(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(200_000_000, 2_000_000_000)
    b = rng.randint(0, a - 100_000)
    return f"What is {a} - {b}?", a - b


def _multiplication(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(20_000, 200_000)
    b = rng.randint(20_000, 200_000)
    return f"What is {a} * {b}?", a * b


def _integer_division(rng: random.Random) -> Tuple[str, int]:
    b = rng.randint(200, 20_000)
    result = rng.randint(100_000, 10_000_000)
    a = b * result
    return f"What is {a} / {b}? Give the integer result.", result


def _modulo(rng: random.Random) -> Tuple[str, int]:
    b = rng.randint(200_000, 2_000_000)
    q = rng.randint(100_000, 2_000_000)
    r = rng.randint(100_000, b - 1)
    a = (q * b) + r
    return f"What is {a} mod {b}?", r


def _power_small(rng: random.Random) -> Tuple[str, int]:
    base = rng.randint(80, 300)
    exp = rng.randint(4, 6)
    return f"What is {base} ^ {exp}?", base ** exp


def _two_step_mul_add(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(2_000, 30_000)
    b = rng.randint(2_000, 30_000)
    c = rng.randint(100_000, 5_000_000)
    return f"Compute ({a} * {b}) + {c}. Return only an integer.", (a * b) + c


def _two_step_sub_mul(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(1_000_000, 20_000_000)
    b = rng.randint(0, a - 100_000)
    c = rng.randint(200, 2_000)
    return f"Compute ({a} - {b}) * {c}. Return only an integer.", (a - b) * c


def _three_step_expr(rng: random.Random) -> Tuple[str, int]:
    a = rng.randint(20_000, 500_000)
    b = rng.randint(20_000, 500_000)
    c = rng.randint(200, 5_000)
    d = rng.randint(100_000, 5_000_000)
    return (
        f"Compute (({a} + {b}) * {c}) - {d}. Return only an integer.",
        ((a + b) * c) - d,
    )


_TASK_GENERATORS = [
    _addition,
    _subtraction,
    _multiplication,
    _integer_division,
    _modulo,
    _power_small,
    _two_step_mul_add,
    _two_step_sub_mul,
    _three_step_expr,
]


def _make_rng(block_hash: str, sample_index: int) -> random.Random:
    key = f"{block_hash}:{sample_index}:sidequest"
    digest = hashlib.sha256(key.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def generate_side_quests(
    block_hash: str,
    sample_index: int,
    n: int = 2,
) -> List[SideQuest]:
    rng = _make_rng(block_hash, sample_index)
    quests: List[SideQuest] = []
    for _ in range(n):
        gen = rng.choice(_TASK_GENERATORS)
        question_text, answer_int = gen(rng)
        quests.append(SideQuest(
            question=question_text,
            answer=str(answer_int),
            answer_int=answer_int,
        ))
    return quests


def shuffle_turn_order(
    block_hash: str,
    sample_index: int,
    n_turns: int = 3,
) -> List[int]:
    rng = _make_rng(block_hash, sample_index)


    for _ in range(20):
        rng.random()
    order = list(range(n_turns))
    rng.shuffle(order)
    return order


def check_side_quest_answer(generated_text: str, quest: SideQuest) -> bool:
    normalised = generated_text.replace(",", "").replace("_", "")
    target = quest.answer
    pattern = r"(?<!\d)" + re.escape(target) + r"(?!\d)"
    return bool(re.search(pattern, normalised))
