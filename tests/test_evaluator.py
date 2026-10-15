import os
import random
from collections import Counter
from itertools import combinations

import pytest

from backend.engine.cards import parse_cards
from backend.engine.evaluator import (
    FLUSH,
    FULL_HOUSE,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TWO_PAIR,
    category,
    evaluate,
)

N_SAMPLES = int(os.environ.get("POKER_EVAL_SAMPLES", "100000"))


def reference5(cards):
    """Independent, sort-based 5-card evaluator: (category, tiebreak tuple)."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    flush = len({c & 3 for c in cards}) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_high = 3
    groups = sorted(Counter(ranks).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in groups]
    order = tuple(r for r, _ in groups)
    if straight_high is not None and flush:
        return (8, (straight_high,))
    if shape == [4, 1]:
        return (7, order)
    if shape == [3, 2]:
        return (6, order)
    if flush:
        return (5, tuple(ranks))
    if straight_high is not None:
        return (4, (straight_high,))
    if shape == [3, 1, 1]:
        return (3, order)
    if shape == [2, 2, 1]:
        return (2, order)
    if shape == [2, 1, 1, 1]:
        return (1, order)
    return (0, tuple(ranks))


def reference7(cards):
    return max(reference5(combo) for combo in combinations(cards, 5))


def test_matches_brute_force_reference():
    rng = random.Random(2026)
    hands = [rng.sample(range(52), 7) for _ in range(N_SAMPLES)]
    for i in range(0, N_SAMPLES - 1, 2):
        a, b = hands[i], hands[i + 1]
        ra, rb = reference7(a), reference7(b)
        sa, sb = evaluate(a), evaluate(b)
        assert category(sa) == ra[0], a
        assert (sa > sb) == (ra > rb) and (sa == sb) == (ra == rb), (a, b)


@pytest.mark.parametrize(
    "cards, cat",
    [
        ("Ah2c3d4s5h9c9d", STRAIGHT),  # wheel beats a pair
        ("Ah2h3h4h5h9c9d", STRAIGHT_FLUSH),  # steel wheel
        ("AhAcAdKhKcKd2s", FULL_HOUSE),  # two sets of trips
        ("AhAcKdKs2c2d9h", TWO_PAIR),  # three pairs
        ("2h3h4h5h7h6c8d", FLUSH),  # flush beats a straight
    ],
)
def test_categories(cards, cat):
    assert category(evaluate(parse_cards(cards))) == cat


def test_ordering_edge_cases():
    def ev(s):
        return evaluate(parse_cards(s))

    assert ev("Ah2c3d4s5h") < ev("2c3d4s5h6c")  # wheel is the lowest straight
    assert ev("AhAcAdKhKcKd2s") == ev("AhAcAdKhKc2s3s")  # third king plays as pair
    assert ev("AhAcKdKs2c2dQh") > ev("AhAcKdKs3c3dJh")  # best kicker, not third pair
    assert ev("AhAcAdAs2c3d9h") > ev("AhAcAdAs2c3d8h")  # quads kicker
    assert ev("KhKcKdKs5c") == ev("KhKcKdKs5d")  # suits never break ties


def test_rejects_wrong_card_counts():
    with pytest.raises(ValueError):
        evaluate(parse_cards("AhKh"))


def test_agrees_with_pokerkit():
    pokerkit = pytest.importorskip("pokerkit")
    from backend.engine.cards import cards_str

    rng = random.Random(7)
    for _ in range(3000):
        cards = rng.sample(range(52), 9)
        board = cards[4:]
        a, b = cards[:2] + board, cards[2:4] + board
        pa = pokerkit.StandardHighHand.from_game(cards_str(cards[:2]), cards_str(board))
        pb = pokerkit.StandardHighHand.from_game(cards_str(cards[2:4]), cards_str(board))
        sa, sb = evaluate(a), evaluate(b)
        assert (sa > sb) == (pa > pb) and (sa == sb) == (pa == pb), cards_str(cards)
