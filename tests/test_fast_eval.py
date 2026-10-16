"""Vectorised evaluator and exact equity against a range."""

from __future__ import annotations

import random

import numpy as np
import pytest

from backend.engine.evaluator import evaluate
from backend.engine.fast_eval import CARD_A, CARD_B, RunoutEquity, evaluate_many, runout_scores
from backend.solver.subgame import Showdown, board_mask


@pytest.mark.parametrize("k", [5, 6, 7])
def test_matches_scalar_evaluator(k):
    rng = random.Random(k)
    hands = np.array([rng.sample(range(52), k) for _ in range(30000)])
    assert (evaluate_many(hands) == np.array([evaluate(h) for h in hands])).all()


def test_edge_hands():
    from backend.engine.cards import parse_cards

    for text in ["Ah2c3d4s5h9c9d", "Ah2h3h4h5h9c9d", "AhAcAdKhKcKd2s", "AhAcKdKs2c2d9h",
                 "2h3h4h5h7h6c8d", "AhAcAdAs2c3d9h", "KhKcKdKs5c"]:
        cards = parse_cards(text)
        assert evaluate_many(np.array([cards]))[0] == evaluate(cards), text


def _reach(rng, board, p=0.3):
    return np.array([rng.random() if rng.random() < p else 0.0 for _ in range(1326)]) * board_mask(board)


def test_river_equity_matches_showdown():
    rng = random.Random(1)
    for _ in range(5):
        board = tuple(rng.sample(range(52), 5))
        r = _reach(rng, board)
        np.testing.assert_allclose(RunoutEquity(board)(r), Showdown(board).equity(r), atol=1e-12)


def test_turn_equity_matches_brute_force():
    rng = random.Random(2)
    board = tuple(rng.sample(range(52), 4))
    r = _reach(rng, board, 0.08)
    eq = RunoutEquity(board)(r)
    heroes = rng.sample([i for i in range(1326) if board_mask(board)[i]], 6)
    for h in heroes:
        hc = {int(CARD_A[h]), int(CARD_B[h])}
        num = den = 0.0
        for o in np.nonzero(r)[0]:
            oc = {int(CARD_A[o]), int(CARD_B[o])}
            if hc & oc:
                continue
            for c in range(52):
                if c in board or c in hc or c in oc:
                    continue
                sh = evaluate((*hc, *board, c))
                so = evaluate((*oc, *board, c))
                num += r[o] * (1.0 if sh > so else 0.5 if sh == so else 0.0)
                den += r[o]
        assert eq[h] == pytest.approx(num / den, abs=1e-12)


def test_runout_scores_shape_and_invalid_combos():
    board = (0, 5, 10, 20)
    s = runout_scores(board)
    assert s.shape == (48, 1326)
    assert (s[:, board_mask(board) == 0] == -1).all()
    with pytest.raises(ValueError):
        runout_scores((0, 1, 2))
