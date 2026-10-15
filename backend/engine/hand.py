"""Single-hand No-Limit Hold'em state machine for 2-9 players.

Rule decisions (see the design doc, section 7.2):

* All chip amounts are ints.
* Preflop the amount to call is the full big blind even if the big blind
  posted short; the minimum raise increment starts at one big blind on
  every street.
* An all-in raise smaller than a full raise does not change the minimum
  raise increment. A player who already acted may re-raise only if the
  bet has grown by at least one full raise since their last action
  (TDA cumulative rule).
* Nobody may raise when no other live player could put in more than the
  current bet (everyone else is all-in or has at most the current bet).
* Heads-up: the button posts the small blind, acts first preflop and last
  after the flop.
* Uncalled chips are returned at the end of each betting round.
* Side pots are cut at each live player's total contribution; odd chips go
  one at a time to winners starting left of the button.
* With at most one player able to act and bets matched, the rest of the
  board is dealt automatically.
* Everyone who reaches showdown shows their cards.
* Antes are not supported yet (the parameter is reserved).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from .actions import Action, ActionType, IllegalActionError, LegalActions
from .cards import Deck
from .evaluator import category_name, evaluate

MAX_PLAYERS = 9


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3

    def __str__(self) -> str:
        return self.name.lower()


class EventType(StrEnum):
    POST_BLIND = "post_blind"
    ACTION = "action"
    DEAL_BOARD = "deal_board"
    RETURN_UNCALLED = "return_uncalled"
    SHOWDOWN = "showdown"
    AWARD = "award"


@dataclass(frozen=True)
class Event:
    """A public event. ``amount`` is chips moved, or the "raise to" total
    for BET/RAISE; ``cards`` are board cards or cards shown at showdown."""

    type: EventType
    street: Street
    seat: int | None = None
    action: ActionType | None = None
    amount: int = 0
    cards: tuple[int, ...] = ()


@dataclass(frozen=True)
class SeatView:
    seat: int
    stack: int
    street_bet: int
    total_bet: int
    folded: bool
    all_in: bool
    shown_cards: tuple[int, ...] | None


@dataclass(frozen=True)
class PlayerView:
    """Everything one seat may legally see. Rebuilt on every query."""

    seat: int | None  # None for a spectator
    hole_cards: tuple[int, ...]
    board: tuple[int, ...]
    street: Street
    button: int
    small_blind: int
    big_blind: int
    pot: int
    current_bet: int
    to_act: int | None
    seats: tuple[SeatView, ...]
    history: tuple[Event, ...]
    is_over: bool


@dataclass(frozen=True)
class PotResult:
    amount: int
    eligible: tuple[int, ...]
    winners: tuple[int, ...]
    shares: tuple[tuple[int, int], ...]  # (seat, chips)


@dataclass(frozen=True)
class HandResult:
    board: tuple[int, ...]
    pots: tuple[PotResult, ...]
    payouts: Mapping[int, int]
    final_stacks: Mapping[int, int]
    net: Mapping[int, int]
    showdown: Mapping[int, tuple[int, ...]]  # seat -> shown hole cards
    hand_names: Mapping[int, str]


class _Player:
    __slots__ = ("seat", "start_stack", "stack", "hole", "folded", "street_bet",
                 "total_bet", "acted", "raise_ref")

    def __init__(self, seat: int, stack: int) -> None:
        self.seat = seat
        self.start_stack = stack
        self.stack = stack
        self.hole: tuple[int, ...] = ()
        self.folded = False
        self.street_bet = 0
        self.total_bet = 0
        self.acted = False
        # Bet level this player last acted at; used for the reopen rule.
        self.raise_ref = 0

    @property
    def live(self) -> bool:
        return not self.folded

    @property
    def can_act(self) -> bool:
        return not self.folded and self.stack > 0


class HandState:
    """One hand of NLHE. Seats are table seat numbers (0..8)."""

    def __init__(
        self,
        stacks: Mapping[int, int],
        button: int,
        small_blind: int,
        big_blind: int,
        deck: Deck,
        *,
        ante: int = 0,
    ) -> None:
        if ante:
            raise NotImplementedError("antes are not supported in Milestone 1")
        if not 2 <= len(stacks) <= MAX_PLAYERS:
            raise ValueError("a hand needs 2 to 9 players")
        if not 0 < small_blind <= big_blind:
            raise ValueError("blinds must satisfy 0 < small_blind <= big_blind")
        if button not in stacks:
            raise ValueError("button must be a seat in the hand")
        for seat, stack in stacks.items():
            if not 0 <= seat < MAX_PLAYERS:
                raise ValueError(f"invalid seat {seat}")
            if stack <= 0:
                raise ValueError(f"seat {seat} has no chips")

        self.small_blind = small_blind
        self.big_blind = big_blind
        self.button = button
        self.deck_seed = deck.seed
        self._deck = deck
        self._order = sorted(stacks)
        self._players = {s: _Player(s, stacks[s]) for s in self._order}
        self.board: list[int] = []
        self.street = Street.PREFLOP
        self.current_bet = 0
        self.last_full_raise = big_blind
        self.to_act: int | None = None
        self.is_over = False
        self.events: list[Event] = []
        self.actions: list[tuple[int, Action]] = []
        self._result: HandResult | None = None

        if len(self._order) == 2:
            self.sb_seat = button
            self.bb_seat = self._next_seat(button)
        else:
            self.sb_seat = self._next_seat(button)
            self.bb_seat = self._next_seat(self.sb_seat)

        first = self._next_seat(button)
        dealt: dict[int, list[int]] = {s: [] for s in self._order}
        for _ in range(2):
            for seat in self._seats_from(first):
                dealt[seat].extend(deck.deal(1))
        for seat, cards in dealt.items():
            self._players[seat].hole = tuple(cards)

        self._post_blind(self.sb_seat, small_blind)
        self._post_blind(self.bb_seat, big_blind)
        self.current_bet = big_blind
        self._start_round(self._next_seat(self.bb_seat))

    # ------------------------------------------------------------------ queries

    @property
    def seats(self) -> tuple[int, ...]:
        return tuple(self._order)

    @property
    def pot(self) -> int:
        """Chips in the middle; 0 once the hand is over and the pot is paid out."""
        if self.is_over:
            return 0
        return sum(p.total_bet for p in self._players.values())

    def stack(self, seat: int) -> int:
        return self._players[seat].stack

    def street_bet(self, seat: int) -> int:
        return self._players[seat].street_bet

    def total_bet(self, seat: int) -> int:
        return self._players[seat].total_bet

    def is_folded(self, seat: int) -> bool:
        return self._players[seat].folded

    def hole_cards(self, seat: int) -> tuple[int, ...]:
        """Privileged access for the runner and hand records, never for agents."""
        return self._players[seat].hole

    def starting_stacks(self) -> dict[int, int]:
        return {s: self._players[s].start_stack for s in self._order}

    def legal_actions(self) -> LegalActions:
        if self.is_over or self.to_act is None:
            raise IllegalActionError("the hand is over")
        p = self._players[self.to_act]
        to_call = self.current_bet - p.street_bet
        call_amount = min(max(to_call, 0), p.stack)
        can_check = to_call <= 0

        raise_type = min_to = max_to = None
        max_total = p.street_bet + p.stack
        reopened = not p.acted or self.current_bet - p.raise_ref >= self.last_full_raise
        # A raise needs someone who could put in more than the current bet.
        opponent_can_call_more = any(
            q.can_act and q.street_bet + q.stack > self.current_bet
            for q in self._players.values()
            if q is not p
        )
        if max_total > self.current_bet and reopened and opponent_can_call_more:
            raise_type = ActionType.BET if self.current_bet == 0 else ActionType.RAISE
            max_to = max_total
            min_to = min(self.current_bet + self.last_full_raise, max_total)

        return LegalActions(
            seat=p.seat,
            can_fold=not can_check,
            can_check=can_check,
            call_amount=call_amount,
            raise_type=raise_type,
            min_raise_to=min_to,
            max_raise_to=max_to,
        )

    def all_in_action(self) -> Action:
        """Translate "all-in" into the matching CALL / BET / RAISE."""
        legal = self.legal_actions()
        if legal.raise_type is not None and legal.max_raise_to is not None:
            return Action(legal.raise_type, legal.max_raise_to)
        if legal.can_call:
            return Action.call()
        raise IllegalActionError("all-in is not possible here")

    def observation(self, seat: int | None) -> PlayerView:
        """What ``seat`` may see; ``None`` is a spectator with no hole cards."""
        if seat is not None and seat not in self._players:
            raise ValueError(f"seat {seat} is not in this hand")
        shown = self._result.showdown if self._result else {}
        seat_views = tuple(
            SeatView(
                seat=q.seat,
                stack=q.stack,
                street_bet=q.street_bet,
                total_bet=q.total_bet,
                folded=q.folded,
                all_in=q.live and q.stack == 0,
                shown_cards=shown.get(q.seat),
            )
            for q in (self._players[s] for s in self._order)
        )
        return PlayerView(
            seat=seat,
            hole_cards=self._players[seat].hole if seat is not None else (),
            board=tuple(self.board),
            street=self.street,
            button=self.button,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            pot=self.pot,
            current_bet=self.current_bet,
            to_act=self.to_act,
            seats=seat_views,
            history=tuple(self.events),
            is_over=self.is_over,
        )

    def result(self) -> HandResult:
        if self._result is None:
            raise RuntimeError("the hand is not over")
        return self._result

    # ------------------------------------------------------------------ actions

    def apply(self, action: Action) -> list[Event]:
        legal = self.legal_actions()
        if not legal.is_legal(action):
            raise IllegalActionError(
                f"seat {legal.seat}: {action} is not legal "
                f"(legal: {[str(t) for t in legal.types()]}, "
                f"raise to {legal.min_raise_to}..{legal.max_raise_to})"
            )
        start = len(self.events)
        p = self._players[legal.seat]
        self.actions.append((p.seat, action))

        if action.type is ActionType.FOLD:
            p.folded = True
            self._emit(EventType.ACTION, p.seat, action=action.type)
        elif action.type is ActionType.CHECK:
            self._emit(EventType.ACTION, p.seat, action=action.type)
        elif action.type is ActionType.CALL:
            self._commit(p, legal.call_amount)
            self._emit(EventType.ACTION, p.seat, action=action.type, amount=legal.call_amount)
        else:
            self._commit(p, action.amount - p.street_bet)
            increment = action.amount - self.current_bet
            if increment >= self.last_full_raise:
                self.last_full_raise = increment
            self.current_bet = action.amount
            self._emit(EventType.ACTION, p.seat, action=action.type, amount=action.amount)
        p.acted = True
        p.raise_ref = self.current_bet

        live = sum(1 for q in self._players.values() if q.live)
        nxt = self._find_actor(self._next_seat(p.seat))
        if live == 1 or nxt is None:
            self._end_round()
        else:
            self.to_act = nxt
        return self.events[start:]

    # ----------------------------------------------------------------- internals

    def _next_seat(self, seat: int) -> int:
        i = self._order.index(seat)
        return self._order[(i + 1) % len(self._order)]

    def _seats_from(self, seat: int) -> list[int]:
        i = self._order.index(seat)
        return self._order[i:] + self._order[:i]

    def _emit(self, type: EventType, seat: int | None = None, **kw) -> None:
        self.events.append(Event(type, self.street, seat, **kw))

    def _commit(self, p: _Player, chips: int) -> None:
        assert 0 <= chips <= p.stack
        p.stack -= chips
        p.street_bet += chips
        p.total_bet += chips

    def _post_blind(self, seat: int, blind: int) -> None:
        p = self._players[seat]
        chips = min(blind, p.stack)
        self._commit(p, chips)
        self._emit(EventType.POST_BLIND, seat, amount=chips)

    def _needs_action(self, p: _Player) -> bool:
        if not p.can_act:
            return False
        if p.street_bet < self.current_bet:
            return True
        if p.acted:
            return False
        # Matched but not yet acted: only worth asking if someone could respond.
        return any(q.can_act for q in self._players.values() if q is not p)

    def _find_actor(self, start: int) -> int | None:
        for seat in self._seats_from(start):
            if self._needs_action(self._players[seat]):
                return seat
        return None

    def _start_round(self, first: int) -> None:
        nxt = self._find_actor(first)
        if nxt is None:
            self._end_round()
        else:
            self.to_act = nxt

    def _end_round(self) -> None:
        self.to_act = None
        self._return_uncalled()
        if sum(1 for q in self._players.values() if q.live) == 1:
            self._finish(showdown=False)
            return
        if self.street is Street.RIVER:
            self._finish(showdown=True)
            return

        self.street = Street(self.street + 1)
        self._deck.burn()
        cards = self._deck.deal(3 if self.street is Street.FLOP else 1)
        self.board.extend(cards)
        self._emit(EventType.DEAL_BOARD, cards=cards)
        for q in self._players.values():
            q.street_bet = 0
            q.acted = False
            q.raise_ref = 0
        self.current_bet = 0
        self.last_full_raise = self.big_blind
        self._start_round(self._next_seat(self.button))

    def _return_uncalled(self) -> None:
        bets = sorted((q.street_bet for q in self._players.values()), reverse=True)
        if len(bets) < 2 or bets[0] == bets[1]:
            return
        top = next(q for q in self._players.values() if q.street_bet == bets[0])
        refund = bets[0] - bets[1]
        top.street_bet -= refund
        top.total_bet -= refund
        top.stack += refund
        self._emit(EventType.RETURN_UNCALLED, top.seat, amount=refund)

    def _pots(self) -> list[tuple[int, tuple[int, ...]]]:
        players = list(self._players.values())
        live = [q for q in players if q.live]
        pots: list[tuple[int, tuple[int, ...]]] = []
        prev = 0
        for level in sorted({q.total_bet for q in live}):
            amount = sum(min(q.total_bet, level) - min(q.total_bet, prev) for q in players)
            eligible = tuple(s for s in self._order
                             if self._players[s].live and self._players[s].total_bet >= level)
            prev = level
            if amount == 0:
                continue
            if pots and pots[-1][1] == eligible:
                pots[-1] = (pots[-1][0] + amount, eligible)
            else:
                pots.append((amount, eligible))
        leftover = sum(max(q.total_bet - prev, 0) for q in players)
        if leftover:
            amount, eligible = pots[-1]
            pots[-1] = (amount + leftover, eligible)
        return pots

    def _finish(self, showdown: bool) -> None:
        players = self._players
        live = [s for s in self._order if players[s].live]
        payout_order = self._seats_from(self._next_seat(self.button))
        shown: dict[int, tuple[int, ...]] = {}
        names: dict[int, str] = {}
        scores: dict[int, int] = {}
        if showdown:
            for s in live:
                shown[s] = players[s].hole
                scores[s] = evaluate(players[s].hole + tuple(self.board))
                names[s] = category_name(scores[s])
                self._emit(EventType.SHOWDOWN, s, cards=players[s].hole)

        payouts = {s: 0 for s in self._order}
        pot_results = []
        for amount, eligible in self._pots():
            if showdown:
                best = max(scores[s] for s in eligible)
                winners = tuple(s for s in payout_order if s in eligible and scores[s] == best)
            else:
                winners = (live[0],)
            share, odd = divmod(amount, len(winners))
            shares = tuple((s, share + (1 if i < odd else 0)) for i, s in enumerate(winners))
            for s, chips in shares:
                payouts[s] += chips
                self._emit(EventType.AWARD, s, amount=chips)
            pot_results.append(PotResult(amount, eligible, winners, shares))

        for s in self._order:
            players[s].stack += payouts[s]
        final = {s: players[s].stack for s in self._order}
        self._result = HandResult(
            board=tuple(self.board),
            pots=tuple(pot_results),
            payouts=payouts,
            final_stacks=final,
            net={s: final[s] - players[s].start_stack for s in self._order},
            showdown=shown,
            hand_names=names,
        )
        self.is_over = True
        self.to_act = None
