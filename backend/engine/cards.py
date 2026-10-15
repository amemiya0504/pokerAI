"""Cards, deterministic decks and seed derivation.

A card is an int in 0..51: ``rank = card >> 2`` (0 = deuce .. 12 = ace) and
``suit = card & 3`` (0..3 = c, d, h, s).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

RANKS = "23456789TJQKA"
SUITS = "cdhs"


def card_rank(card: int) -> int:
    return card >> 2


def card_suit(card: int) -> int:
    return card & 3


def make_card(rank: int, suit: int) -> int:
    return rank << 2 | suit


def card_str(card: int) -> str:
    if not 0 <= card < 52:
        raise ValueError(f"invalid card {card}")
    return RANKS[card >> 2] + SUITS[card & 3]


def cards_str(cards: Iterable[int]) -> str:
    return "".join(card_str(c) for c in cards)


def parse_card(text: str) -> int:
    if len(text) != 2 or text[0] not in RANKS or text[1] not in SUITS:
        raise ValueError(f"invalid card {text!r}")
    return make_card(RANKS.index(text[0]), SUITS.index(text[1]))


def parse_cards(text: str) -> tuple[int, ...]:
    """Parse ``"AsKd"`` or ``"As Kd"`` into card ints."""
    text = text.replace(" ", "")
    if len(text) % 2:
        raise ValueError(f"invalid card string {text!r}")
    return tuple(parse_card(text[i : i + 2]) for i in range(0, len(text), 2))


def derive_seed(master: int, *labels: object) -> int:
    """Derive an independent 63-bit sub-seed from a master seed.

    Every random source (deck, agents, scheduling) gets its own sub-seed so
    that changing one consumer never shifts the random stream of another.
    63 bits so a seed fits a signed BIGINT column.
    """
    digest = hashlib.sha256(repr((master, *labels)).encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


class Deck:
    """A 52-card deck shuffled by ``random.Random(seed)``.

    The same seed always yields the same order.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._cards = list(range(52))
        random.Random(seed).shuffle(self._cards)
        self._pos = 0

    @property
    def remaining(self) -> int:
        return 52 - self._pos

    def deal(self, n: int = 1) -> tuple[int, ...]:
        if n > self.remaining:
            raise ValueError("deck exhausted")
        cards = tuple(self._cards[self._pos : self._pos + n])
        self._pos += n
        return cards

    def burn(self) -> None:
        self.deal(1)
