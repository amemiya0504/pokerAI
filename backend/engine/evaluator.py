"""Direct 5-7 card hand evaluation.

Uses rank counts and per-suit rank bitmasks; no lookup of 5-card
combinations. Scores are plain ints: a higher score is a stronger hand and
equal scores tie. Layout: ``category << 20`` followed by up to five 4-bit
tie-break ranks.
"""

from __future__ import annotations

from collections.abc import Sequence

HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = (
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
)

_WHEEL = 0b1_0000_0000_1111  # A-2-3-4-5


def _straight_high(mask: int) -> int:
    for high in range(12, 3, -1):
        window = 0b11111 << (high - 4)
        if mask & window == window:
            return high
    if mask & _WHEEL == _WHEEL:
        return 3  # five-high
    return -1


_STRAIGHT_HIGH = tuple(_straight_high(m) for m in range(1 << 13))


def _top_ranks(mask: int, k: int) -> list[int]:
    ranks = []
    for r in range(12, -1, -1):
        if mask >> r & 1:
            ranks.append(r)
            if len(ranks) == k:
                break
    return ranks


def _pack(category: int, ranks: Sequence[int]) -> int:
    score = category
    for i in range(5):
        score = score << 4 | (ranks[i] if i < len(ranks) else 0)
    return score


def evaluate(cards: Sequence[int]) -> int:
    """Score the best 5-card hand among 5 to 7 cards."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("evaluate needs 5 to 7 cards")
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suits[c & 3] |= 1 << r
        mask |= 1 << r

    # With at most 7 cards a flush excludes quads and full houses, so the
    # flush branch can return immediately.
    for suit_mask in suits:
        if suit_mask.bit_count() >= 5:
            high = _STRAIGHT_HIGH[suit_mask]
            if high >= 0:
                return _pack(STRAIGHT_FLUSH, (high,))
            return _pack(FLUSH, _top_ranks(suit_mask, 5))

    quads: list[int] = []
    trips: list[int] = []
    pairs: list[int] = []
    singles: list[int] = []
    for r in range(12, -1, -1):
        n = counts[r]
        if n == 4:
            quads.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 1:
            singles.append(r)

    if quads:
        q = quads[0]
        kicker = max(r for r in range(13) if counts[r] and r != q)
        return _pack(FOUR_OF_A_KIND, (q, kicker))
    if trips and (len(trips) > 1 or pairs):
        pair = max(trips[1] if len(trips) > 1 else -1, pairs[0] if pairs else -1)
        return _pack(FULL_HOUSE, (trips[0], pair))
    high = _STRAIGHT_HIGH[mask]
    if high >= 0:
        return _pack(STRAIGHT, (high,))
    if trips:
        return _pack(THREE_OF_A_KIND, (trips[0], *singles[:2]))
    if len(pairs) >= 2:
        kicker = max(pairs[2:] + singles[:1])
        return _pack(TWO_PAIR, (pairs[0], pairs[1], kicker))
    if pairs:
        return _pack(ONE_PAIR, (pairs[0], *singles[:3]))
    return _pack(HIGH_CARD, singles[:5])


def category(score: int) -> int:
    return score >> 20


def category_name(score: int) -> str:
    return CATEGORY_NAMES[score >> 20]
