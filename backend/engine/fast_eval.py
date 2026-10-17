"""Vectorised 5-7 card evaluation with NumPy: same scores as ``evaluator.evaluate``.

Scores many hands at once (an (M, k) array of card ints) using rank counts,
per-suit rank bitmasks and lookup tables indexed by 13-bit rank masks.
"""

from __future__ import annotations

import numpy as np

from .evaluator import (
    FLUSH,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    HIGH_CARD,
    ONE_PAIR,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    _STRAIGHT_HIGH,
)

_MASKS = np.arange(1 << 13)
STRAIGHT_HIGH = np.array(_STRAIGHT_HIGH, dtype=np.int64)
# TOP[mask, i] = i-th highest rank in mask (0 when fewer ranks are present).
TOP = np.zeros((1 << 13, 5), dtype=np.int64)
for _m in range(1 << 13):
    _ranks = [r for r in range(12, -1, -1) if _m >> r & 1][:5]
    TOP[_m, : len(_ranks)] = _ranks
POPCOUNT = np.array([bin(m).count("1") for m in range(1 << 13)], dtype=np.int64)


def _pack(category, r0=0, r1=0, r2=0, r3=0, r4=0):
    return (category << 20) | (r0 << 16) | (r1 << 12) | (r2 << 8) | (r3 << 4) | r4


def _combo_cards():
    from itertools import combinations

    pairs = np.array(list(combinations(range(52), 2)), dtype=np.int64)
    card_combos = np.array([[i for i, (a, b) in enumerate(pairs) if c in (a, b)] for c in range(52)])
    return pairs[:, 0], pairs[:, 1], card_combos


CARD_A, CARD_B, CARD_COMBOS = _combo_cards()
N_COMBOS = len(CARD_A)


def runout_scores(board) -> np.ndarray:
    """Hand scores for all 1,326 combos on every completion of ``board`` to
    five cards (one row per river card on the turn, a single row on the
    river); -1 where the combo shares a card with that board."""
    board = list(board)
    if len(board) == 5:
        boards = [board]
    elif len(board) == 4:
        boards = [board + [c] for c in range(52) if c not in board]
    else:
        raise ValueError("runout_scores supports turn and river boards")
    B = len(boards)
    bmat = np.array(boards, dtype=np.int64)
    out = np.full((B, N_COMBOS), -1, dtype=np.int64)
    used = np.zeros((B, 52), dtype=bool)
    used[np.arange(B)[:, None], bmat] = True
    valid = ~(used[:, CARD_A] | used[:, CARD_B])
    bi, ci = np.nonzero(valid)
    hands = np.concatenate([CARD_A[ci, None], CARD_B[ci, None], bmat[bi]], axis=1)
    if len(hands) >= 20_000:
        # NumPy releases the GIL in most of evaluate_many: two threads halve
        # the time on a turn board (more threads contend for the GIL).
        out[bi, ci] = np.concatenate(list(_pool().map(evaluate_many, np.array_split(hands, 2))))
    else:
        out[bi, ci] = evaluate_many(hands)
    return out


def runout_scores_all(board) -> np.ndarray:
    """Like ``runout_scores`` but completes any 3- or 4-card board to five
    cards (all unordered runouts): one row per runout, -1 for combos that
    share a card with it."""
    from itertools import combinations

    board = list(board)
    rest = [c for c in range(52) if c not in board]
    runouts = [board + list(r) for r in combinations(rest, 5 - len(board))]
    bmat = np.array(runouts, dtype=np.int64)
    B = len(bmat)
    out = np.full((B, N_COMBOS), -1, dtype=np.int64)
    used = np.zeros((B, 52), dtype=bool)
    used[np.arange(B)[:, None], bmat] = True
    valid = ~(used[:, CARD_A] | used[:, CARD_B])
    bi, ci = np.nonzero(valid)
    hands = np.concatenate([CARD_A[ci, None], CARD_B[ci, None], bmat[bi]], axis=1)
    chunks = np.array_split(hands, max(1, len(hands) // 400_000))
    out[bi, ci] = np.concatenate(list(_pool().map(evaluate_many, chunks)))
    return out


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _POOL = ThreadPoolExecutor(2, thread_name_prefix="fast_eval")
    return _POOL


class _RankIndex:
    """For row-wise keys: the sort order and, for each query, how many keys are
    < it and <= it. Depends only on the keys, so it is built once per board."""

    def __init__(self, keys: np.ndarray, queries: np.ndarray) -> None:
        R, L = keys.shape
        self.order = np.argsort(keys, axis=1, kind="stable")
        ks = np.take_along_axis(keys, self.order, 1)
        off = (np.arange(R, dtype=np.int64) << 32)[:, None]
        flat = (ks + 1 + off).ravel()
        q = (queries + 1 + off).ravel()
        base = np.arange(R)[:, None] * L
        self.lt = np.searchsorted(flat, q, "left").reshape(queries.shape) - base
        self.le = np.searchsorted(flat, q, "right").reshape(queries.shape) - base

    def sums(self, weights: np.ndarray):
        ws = np.take_along_axis(weights, self.order, 1)
        cum = np.concatenate([np.zeros((len(ws), 1)), np.cumsum(ws, axis=1)], axis=1)
        return np.take_along_axis(cum, self.lt, 1), np.take_along_axis(cum, self.le, 1)


# POS_A[h] / POS_B[h]: position of combo h inside CARD_COMBOS[CARD_A[h]] / [CARD_B[h]].
POS_A = np.array([int(np.nonzero(CARD_COMBOS[CARD_A[h]] == h)[0][0]) for h in range(N_COMBOS)])
POS_B = np.array([int(np.nonzero(CARD_COMBOS[CARD_B[h]] == h)[0][0]) for h in range(N_COMBOS)])


class RunoutEquity:
    """Exact all-in equity of every combo against any opponent range on a turn
    or river board, with exact card removal: every (opponent combo, river card)
    pair compatible with the hero's combo counts with its weight; ties count
    half. Board-only work happens once in the constructor."""

    def __init__(self, board) -> None:
        self.scores = runout_scores(board)
        B = len(self.scores)
        self.valid = self.scores >= 0
        self.total_index = _RankIndex(self.scores, self.scores)
        card_scores = self.scores[:, CARD_COMBOS].reshape(B * 52, -1)
        # The hero combos holding card c are exactly the combos in its row, so each row queries itself.
        self.card_index = _RankIndex(card_scores, card_scores)
        rows = np.arange(B)[:, None] * 52
        self.a = (rows + CARD_A, POS_A)
        self.b = (rows + CARD_B, POS_B)

    def __call__(self, r_opp: np.ndarray) -> np.ndarray:
        B = len(self.scores)
        R = np.where(self.valid, np.asarray(r_opp, dtype=np.float64)[None, :], 0.0)
        lt_t, le_t = self.total_index.sums(R)
        rc = R[:, CARD_COMBOS].reshape(B * 52, -1)
        lt_c, le_c = self.card_index.sums(rc)
        weaker = lt_t - lt_c[self.a] - lt_c[self.b]
        not_stronger = le_t - le_c[self.a] - le_c[self.b] + R  # own combo subtracted twice
        ties = not_stronger - weaker
        card_tot = rc.sum(1).reshape(B, 52)
        compat = R.sum(1, keepdims=True) - card_tot[:, CARD_A] - card_tot[:, CARD_B] + R
        num = np.where(self.valid, weaker + 0.5 * ties, 0.0).sum(0)
        den = np.where(self.valid, compat, 0.0).sum(0)
        return np.where(den > 1e-12, num / np.maximum(den, 1e-30), 0.0)


def equity_vs_range(scores_or_board, r_opp: np.ndarray) -> np.ndarray:
    """One-off convenience wrapper around ``RunoutEquity``."""
    return RunoutEquity(scores_or_board)(r_opp)


def evaluate_many(cards: np.ndarray) -> np.ndarray:
    """Scores for an (M, k) int array of 5 to 7 cards per row."""
    cards = np.asarray(cards, dtype=np.int64)
    M, k = cards.shape
    if not 5 <= k <= 7:
        raise ValueError("evaluate_many needs 5 to 7 cards per hand")
    ranks, suits = cards >> 2, cards & 3
    bit = np.left_shift(1, ranks)
    row = np.arange(M)[:, None]
    counts = np.bincount((row * 13 + ranks).ravel(), minlength=M * 13).reshape(M, 13)
    # Ranks within one suit are distinct, so summing their bits gives the suit's rank mask.
    suit_masks = np.bincount((row * 4 + suits).ravel(), weights=bit.ravel(),
                             minlength=M * 4).reshape(M, 4).astype(np.int64)
    rank_mask = (counts > 0) @ (1 << np.arange(13))
    score = np.zeros(M, dtype=np.int64)
    done = np.zeros(M, dtype=bool)

    # Flush (and straight flush): with at most 7 cards it beats any quads / full house.
    suit_pop = POPCOUNT[suit_masks]
    fl = np.nonzero(suit_pop.max(1) >= 5)[0]
    if len(fl):
        fmask = suit_masks[fl, suit_pop[fl].argmax(1)]
        sf = STRAIGHT_HIGH[fmask]
        t = TOP[fmask]
        score[fl] = np.where(sf >= 0, _pack(STRAIGHT_FLUSH, np.maximum(sf, 0)),
                             _pack(FLUSH, *[t[:, i] for i in range(5)]))
        done[fl] = True

    # The two largest rank groups by (count, rank): key = count * 16 + rank.
    key = np.where(counts > 0, counts * 16 + np.arange(13), -1)
    i0 = key.argmax(1)
    k0 = key[np.arange(M), i0]
    key[np.arange(M), i0] = -1
    k1 = key.max(1)
    c0, r0, c1, r1 = k0 // 16, k0 % 16, k1 // 16, k1 % 16
    one = np.int64(1)

    sel = ~done & (c0 == 4)
    idx = np.nonzero(sel)[0]
    score[idx] = _pack(FOUR_OF_A_KIND, r0[idx], TOP[rank_mask[idx] & ~(one << r0[idx]), 0])
    done[idx] = True

    idx = np.nonzero(~done & (c0 == 3) & (c1 >= 2))[0]
    score[idx] = _pack(FULL_HOUSE, r0[idx], r1[idx])
    done[idx] = True

    st = STRAIGHT_HIGH[rank_mask]
    idx = np.nonzero(~done & (st >= 0))[0]
    score[idx] = _pack(STRAIGHT, st[idx])
    done[idx] = True

    idx = np.nonzero(~done & (c0 == 3))[0]
    t = TOP[rank_mask[idx] & ~(one << r0[idx])]
    score[idx] = _pack(THREE_OF_A_KIND, r0[idx], t[:, 0], t[:, 1])
    done[idx] = True

    idx = np.nonzero(~done & (c0 == 2) & (c1 == 2))[0]
    t = TOP[rank_mask[idx] & ~(one << r0[idx]) & ~(one << r1[idx])]
    score[idx] = _pack(TWO_PAIR, r0[idx], r1[idx], t[:, 0])
    done[idx] = True

    idx = np.nonzero(~done & (c0 == 2))[0]
    t = TOP[rank_mask[idx] & ~(one << r0[idx])]
    score[idx] = _pack(ONE_PAIR, r0[idx], t[:, 0], t[:, 1], t[:, 2])
    done[idx] = True

    idx = np.nonzero(~done)[0]
    t = TOP[rank_mask[idx]]
    score[idx] = _pack(HIGH_CARD, *[t[:, i] for i in range(5)])
    return score
