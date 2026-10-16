"""Features for the policy network, shared by training-data preparation and
inference so both see exactly the same inputs.

Per node (public state): street, who acts, facing a bet, bet geometry relative
to the pot, legal-action mask, and both players' reach (normalised).
Per combo: hole and board cards, made-hand category, strength percentile among
all combos on this board, and exact all-in equity against the opponent's reach
(over every river card on the turn).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.engine.cards import parse_cards
from backend.engine.fast_eval import RunoutEquity, evaluate_many
from backend.solver.subgame import CARD_A, CARD_B, N, board_mask

VOCAB = ("f", "x", "c", "b33", "b50", "b75", "b100", "b150", "r100", "a")
SLOT = {a: i for i, a in enumerate(VOCAB)}
N_ACTIONS = len(VOCAB)
N_BINS = 10  # range-vs-range equity histogram bins per player
N_NODE = 8 + 2 * N_BINS  # length of node_numeric


def action_slots(actions: list[str]) -> list[int]:
    try:
        return [SLOT[a] for a in actions]
    except KeyError as e:
        raise ValueError(f"action {e.args[0]!r} is not in the policy vocabulary {VOCAB}") from None


@dataclass
class BoardFeatures:
    """Everything that depends only on the board (shared by a spot's nodes)."""

    board: tuple[int, ...]
    valid: np.ndarray  # (N,) combos not touching the board
    category: np.ndarray  # (N,) int8, 0..8, -1 if invalid
    strength: np.ndarray  # (N,) percentile in [0, 1] among valid combos
    runout: RunoutEquity  # equity against a range, over all river cards on the turn


def board_features(board_text: str) -> BoardFeatures:
    board = parse_cards(board_text)
    valid = board_mask(board) > 0
    idx = np.nonzero(valid)[0]
    score = np.full(N, -1, dtype=np.int64)
    hands = np.concatenate([CARD_A[idx, None], CARD_B[idx, None],
                            np.tile(np.array(board, dtype=np.int64), (len(idx), 1))], axis=1)
    score[idx] = evaluate_many(hands)
    s = score[idx]
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    # Average rank for ties, scaled to [0, 1].
    sorted_s = s[order]
    lo = np.searchsorted(sorted_s, sorted_s, "left")
    hi = np.searchsorted(sorted_s, sorted_s, "right") - 1
    ranks[order] = (lo + hi) / 2
    strength = np.zeros(N)
    strength[idx] = ranks / max(len(s) - 1, 1)
    category = np.full(N, -1, dtype=np.int8)
    category[idx] = (s >> 20).astype(np.int8)
    return BoardFeatures(board, valid, category, strength, RunoutEquity(board))


def range_histograms(bf: BoardFeatures, reach_actor: np.ndarray, reach_opp: np.ndarray,
                     eq_actor: np.ndarray | None = None) -> np.ndarray:
    """How each range's hands fare against the other range: reach-weighted
    histograms of equity (actor vs opponent's range, then opponent vs the
    actor's range), N_BINS bins each, each summing to 1. They summarise range
    and nut advantage, which drive bet sizing."""
    eq_a = bf.runout(reach_opp) if eq_actor is None else eq_actor
    eq_o = bf.runout(reach_actor)
    out = []
    for eq, w in ((eq_a, reach_actor), (eq_o, reach_opp)):
        h = np.bincount(np.minimum((eq * N_BINS).astype(np.int64), N_BINS - 1), weights=w, minlength=N_BINS)
        out.append(h / h.sum() if h.sum() > 0 else h)
    return np.concatenate(out).astype(np.float32)


def node_numeric(street: str, player: int, pot: float, stack: float,
                 inv: tuple[float, float], histograms: np.ndarray) -> np.ndarray:
    pot_now = pot + inv[0] + inv[1]
    to_call = max(inv[1 - player] - inv[player], 0.0)
    remaining = stack - inv[player]
    return np.array([
        1.0 if street == "river" else 0.0,
        float(player),
        1.0 if to_call > 0 else 0.0,
        to_call / pot_now,
        min(remaining / pot_now, 20.0) / 20.0,
        inv[player] / pot_now,
        inv[1 - player] / pot_now,
        pot / pot_now,
        *histograms,
    ], dtype=np.float32)


def normalise(reach: np.ndarray) -> np.ndarray:
    s = reach.sum()
    return (reach / s * 1000.0 if s > 0 else reach).astype(np.float32)


def equity(bf: BoardFeatures, reach_opp: np.ndarray) -> np.ndarray:
    return bf.runout(reach_opp)


def legal_mask(actions: list[str]) -> np.ndarray:
    m = np.zeros(N_ACTIONS, dtype=bool)
    m[action_slots(actions)] = True
    return m
