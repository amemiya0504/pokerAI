"""Seats, button movement and starting new hands.

Simplified moving button: each hand the button moves to the next seated
player who is in the hand. No dead button and no missed-blind tracking.
"""

from __future__ import annotations

from dataclasses import dataclass

from .actions import Action, LegalActions
from .cards import Deck, derive_seed
from .hand import MAX_PLAYERS, HandState


@dataclass
class Seat:
    player_id: str
    stack: int
    sitting_out: bool = False
    leave_pending: bool = False

    @property
    def away(self) -> bool:
        """Away players are auto-checked or auto-folded when it is their turn."""
        return self.sitting_out or self.leave_pending


class Table:
    def __init__(self, small_blind: int, big_blind: int, *, seed: int,
                 max_seats: int = MAX_PLAYERS) -> None:
        if not 2 <= max_seats <= MAX_PLAYERS:
            raise ValueError("max_seats must be 2 to 9")
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.seed = seed
        self.max_seats = max_seats
        self.seats: list[Seat | None] = [None] * max_seats
        self.button: int | None = None
        self.hand_no = 0
        self.hand: HandState | None = None
        self.last_hand: HandState | None = None

    @property
    def hand_running(self) -> bool:
        """True from start_hand until finish_hand has settled the stacks."""
        return self.hand is not None

    def _in_hand(self, seat: int) -> bool:
        return self.hand_running and seat in self.hand.seats

    def _seat(self, seat: int) -> Seat:
        s = self.seats[seat]
        if s is None:
            raise ValueError(f"seat {seat} is empty")
        return s

    def sit(self, seat: int, player_id: str, stack: int) -> None:
        if self.seats[seat] is not None:
            raise ValueError(f"seat {seat} is taken")
        if stack <= 0:
            raise ValueError("stack must be positive")
        self.seats[seat] = Seat(player_id, stack)

    def leave(self, seat: int) -> int | None:
        """Leave the table. Mid-hand the leave takes effect when the hand ends;
        returns the stack taken away, or None if the leave is pending."""
        s = self._seat(seat)
        if self._in_hand(seat):
            s.leave_pending = True
            return None
        self.seats[seat] = None
        return s.stack

    def sit_out(self, seat: int) -> None:
        self._seat(seat).sitting_out = True

    def sit_in(self, seat: int) -> None:
        s = self._seat(seat)
        if s.stack <= 0:
            raise ValueError("add chips before sitting back in")
        s.sitting_out = False

    def add_chips(self, seat: int, amount: int) -> None:
        if self._in_hand(seat):
            raise ValueError("cannot add chips during a hand")
        if amount <= 0:
            raise ValueError("amount must be positive")
        self._seat(seat).stack += amount

    def is_away(self, seat: int) -> bool:
        s = self.seats[seat]
        return s is None or s.away

    @staticmethod
    def away_action(legal: LegalActions) -> Action:
        return Action.check() if legal.can_check else Action.fold()

    def eligible_seats(self) -> list[int]:
        return [i for i, s in enumerate(self.seats)
                if s is not None and not s.away and s.stack > 0]

    def start_hand(self) -> HandState:
        if self.hand_running:
            raise RuntimeError("a hand is already running")
        eligible = self.eligible_seats()
        if len(eligible) < 2:
            raise RuntimeError("need at least 2 eligible players")
        if self.button is None:
            self.button = eligible[0]
        else:
            self.button = next((i for i in eligible if i > self.button), eligible[0])
        self.hand_no += 1
        deck = Deck(derive_seed(self.seed, "deck", self.hand_no))
        self.hand = HandState(
            {i: self.seats[i].stack for i in eligible},
            self.button,
            self.small_blind,
            self.big_blind,
            deck,
        )
        return self.hand

    def finish_hand(self) -> None:
        if self.hand is None or not self.hand.is_over:
            raise RuntimeError("no finished hand to settle")
        for seat, stack in self.hand.result().final_stacks.items():
            s = self._seat(seat)
            s.stack = stack
            if stack == 0:
                s.sitting_out = True
        for i, s in enumerate(self.seats):
            if s is not None and s.leave_pending:
                self.seats[i] = None
        self.last_hand, self.hand = self.hand, None
