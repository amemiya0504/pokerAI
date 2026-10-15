"""Deterministic postflop spots for the GTO dataset.

Spot ``index`` under ``seed`` is always the same spot, so a dataset can be
rebuilt or resumed shard by shard. Ranges are synthetic: each player holds
the top X% of starting hands by the Chen formula (X drawn per spot), with a
few hands at reduced weight. They are not derived from a preflop
equilibrium; the dataset manifest says so.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np

from backend.agents.rule_based import chen_score
from backend.engine.cards import cards_str, derive_seed
from backend.solver.hands import COMBOS, N_CLASSES
from backend.solver.subgame import COMBO_INDEX, N, TreeConfig, board_mask

# Classes ordered strongest first by the Chen formula (ties by class index).
_CHEN = [chen_score(*COMBOS[i][0]) for i in range(N_CLASSES)]
CLASS_ORDER = sorted(range(N_CLASSES), key=lambda i: (-_CHEN[i], i))

STREET_CARDS = {"flop": 3, "turn": 4, "river": 5}


@dataclass(frozen=True)
class TreeSpec:
    """Bet abstraction per street, as pot fractions."""

    bet_sizes: dict[str, list[float]]
    raise_sizes: dict[str, list[float]]
    max_raises: int = 1
    allin: bool = True


DEFAULT_TREES = {
    "river": TreeSpec({"river": [0.33, 0.75, 1.5]}, {"river": [1.0]}),
    "turn": TreeSpec({"turn": [0.5, 1.0], "river": [0.5, 1.0]}, {"turn": [1.0], "river": [1.0]}),
}


@dataclass(frozen=True)
class Spot:
    index: int
    street: str
    board: str
    pot: float
    stack: float
    oop_width: float  # share of starting hands (by combos) in the OOP range
    ip_width: float
    oop_range: tuple[float, ...]
    ip_range: tuple[float, ...]

    def ranges(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.oop_range), np.array(self.ip_range)

    def describe(self) -> dict:
        d = asdict(self)
        d.pop("oop_range")
        d.pop("ip_range")
        return d


def top_range(width: float, rng: random.Random, board) -> np.ndarray:
    """Top ``width`` of combos by Chen order; ~15% of classes at random reduced weight."""
    w = np.zeros(N)
    target = width * 1326
    taken = 0
    for ci in CLASS_ORDER:
        if taken >= target:
            break
        weight = rng.uniform(0.25, 1.0) if rng.random() < 0.15 else 1.0
        for a, b in COMBOS[ci]:
            w[COMBO_INDEX[(min(a, b), max(a, b))]] = weight
        taken += len(COMBOS[ci])
    return w * board_mask(board)


def make_spot(seed: int, index: int, streets: dict[str, float]) -> Spot:
    """``streets``: probability of each street, e.g. {"river": 0.7, "turn": 0.3}."""
    rng = random.Random(derive_seed(seed, "spot", index))
    names = sorted(streets)
    street = rng.choices(names, [streets[s] for s in names])[0]
    board = tuple(rng.sample(range(52), STREET_CARDS[street]))
    spr = rng.choice([0.5, 1, 2, 4] if street == "river" else [1, 2, 4, 8])
    pot = 10.0
    oop_w, ip_w = round(rng.uniform(0.15, 0.6), 3), round(rng.uniform(0.15, 0.6), 3)
    oop = top_range(oop_w, rng, board)
    ip = top_range(ip_w, rng, board)
    return Spot(index, street, cards_str(board), pot, pot * spr, oop_w, ip_w,
                tuple(float(x) for x in oop), tuple(float(x) for x in ip))


def tree_config(spot: Spot, trees: dict[str, TreeSpec]) -> TreeConfig:
    t = trees[spot.street]
    return TreeConfig(spot.pot, spot.stack, dict(t.bet_sizes), dict(t.raise_sizes), t.max_raises, t.allin)
