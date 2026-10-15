"""Shared test helpers (tests/ is on sys.path via conftest)."""

from __future__ import annotations

from collections.abc import Mapping

from backend.engine.actions import Action
from backend.engine.cards import Deck, parse_cards
from backend.engine.hand import EventType, HandState


class StackedDeck(Deck):
    """A deck whose top cards are fixed; the rest follow in index order."""

    def __init__(self, top: list[int]) -> None:
        super().__init__(0)
        rest = [c for c in range(52) if c not in top]
        self._cards = list(top) + rest
        self.seed = None


def make_hand(
    stacks: Mapping[int, int],
    button: int,
    holes: Mapping[int, str],
    board: str = "",
    small_blind: int = 50,
    big_blind: int = 100,
) -> HandState:
    """Build a hand with chosen hole cards and board (burns use filler cards).

    Seats missing from ``holes`` get filler cards.
    """
    order = sorted(stacks)
    i = order.index(button)
    deal_order = order[i + 1 :] + order[: i + 1]
    chosen = {s: parse_cards(h) for s, h in holes.items()}
    board_cards = parse_cards(board)
    used = {c for cs in chosen.values() for c in cs} | set(board_cards)
    filler = iter(c for c in range(52) if c not in used)
    hole = {s: chosen.get(s) or (next(filler), next(filler)) for s in order}

    top = [hole[s][0] for s in deal_order] + [hole[s][1] for s in deal_order]
    streets = [board_cards[:3], board_cards[3:4], board_cards[4:5]]
    for cards in streets:
        top.append(next(filler))  # burn
        top.extend(cards)
    while len(top) < len(deal_order) * 2 + 8:
        top.append(next(filler))
    return HandState(dict(stacks), button, small_blind, big_blind, StackedDeck(top))


def play(state: HandState, *actions: str) -> None:
    """Apply actions written as 'f', 'x', 'c', 'b300', 'r300', 'allin'."""
    for a in actions:
        if a == "f":
            state.apply(Action.fold())
        elif a == "x":
            state.apply(Action.check())
        elif a == "c":
            state.apply(Action.call())
        elif a == "allin":
            state.apply(state.all_in_action())
        elif a[0] == "b":
            state.apply(Action.bet(int(a[1:])))
        elif a[0] == "r":
            state.apply(Action.raise_to(int(a[1:])))
        else:
            raise ValueError(a)


def check_invariants(state: HandState, total_chips: int) -> None:
    """Invariants that must hold after every action."""
    seats = state.seats
    if state.is_over:
        final = state.result().final_stacks
        assert sum(final.values()) == total_chips, "chips not conserved"
        assert all(v >= 0 for v in final.values())
    else:
        assert state.pot >= 0
        assert all(state.stack(s) >= 0 for s in seats)
        assert sum(state.stack(s) for s in seats) + state.pot == total_chips, "chips not conserved"
        actor = state.to_act
        assert actor is not None and not state.is_folded(actor) and state.stack(actor) > 0
        legal = state.legal_actions()
        assert legal.seat == actor
        assert legal.can_fold != legal.can_check
        if legal.can_raise:
            assert legal.min_raise_to <= legal.max_raise_to
            assert legal.max_raise_to == state.street_bet(actor) + state.stack(actor)

    shown = state.result().showdown if state.is_over else {}
    for seat in (*seats, None):
        view = state.observation(seat)
        assert view.hole_cards == (state.hole_cards(seat) if seat is not None else ())
        hidden = {c for s in seats if s != seat and s not in shown for c in state.hole_cards(s)}
        visible = set(view.hole_cards) | set(view.board)
        for sv in view.seats:
            if sv.shown_cards:
                visible |= set(sv.shown_cards)
        for e in view.history:
            visible |= set(e.cards)
            if e.type is EventType.SHOWDOWN:
                assert e.seat in shown
        assert not visible & hidden, f"seat {seat} can see hidden cards"
