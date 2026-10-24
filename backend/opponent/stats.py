"""Opponent statistics from public hand histories (Milestone 8).

Only public information is used: blinds, stacks, the action sequence and
the cards shown at showdown. Each statistic is a *group* of outcomes that
share one denominator (design doc section 6: keep the opportunities, not
percentages), so a posterior can be rebuilt from raw counts.

Two kinds of groups:

* decision contexts: every decision a player makes falls into exactly one
  context, e.g. ``flop_vs_bet`` (facing the first bet on the flop) with
  outcomes fold / call / raise. A context is only counted when every one of
  its outcomes was legal (a player facing an all-in cannot raise), so the
  frequency means "when all options are open". These are what range
  tracking and an exploit engine need.
* summary statistics (HUD style): VPIP, PFR, 3-bet, fold to 3-bet, c-bet,
  fold to c-bet, went to showdown, won at showdown; binary yes / no.

The public hand is replayed through the engine itself (any deck: betting
legality does not depend on the cards), so contexts and legal actions are
exactly the engine's. ``decision_context`` is also what style agents use to
pick their action, so an agent's parameters and the measured statistics
refer to the same situations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from backend.engine.actions import Action, ActionType, LegalActions
from backend.engine.cards import Deck
from backend.engine.hand import EventType, HandState, PlayerView, Street

POSTFLOP = ("flop", "turn", "river")

# Context groups: outcome names in a fixed order.
CONTEXTS: dict[str, tuple[str, ...]] = {
    "pf_open": ("fold", "call", "raise"),  # no raise yet, chips to call (first in, or behind limpers)
    "pf_option": ("check", "raise"),  # no raise yet, nothing to call (big blind after limps)
    "pf_vs_raise": ("fold", "call", "raise"),  # facing an open raise
    "pf_vs_3bet": ("fold", "call", "raise"),  # facing a re-raise
    "pf_vs_4bet": ("fold", "call", "raise"),  # facing three or more raises
}
for _s in POSTFLOP:
    CONTEXTS[f"{_s}_unbet"] = ("check", "bet")
    CONTEXTS[f"{_s}_vs_bet"] = ("fold", "call", "raise")
    CONTEXTS[f"{_s}_vs_raise"] = ("fold", "call", "raise")

SUMMARY: dict[str, str] = {
    "vpip": "put chips in voluntarily preflop (of hands with a preflop decision)",
    "pfr": "raised preflop (of hands with a preflop decision)",
    "three_bet": "re-raised when first facing a single raise preflop",
    "fold_to_three_bet": "folded when the own open raise was re-raised",
    "cbet_flop": "bet the flop as the preflop aggressor when checked to (or first to act)",
    "fold_to_cbet_flop": "folded to the preflop aggressor's first flop bet",
    "wtsd": "went to showdown (of hands where the player saw the flop)",
    "wsd": "won chips at showdown (of showdowns)",
}
BINARY = ("yes", "no")

GROUPS: dict[str, tuple[str, ...]] = {**CONTEXTS, **{k: BINARY for k in SUMMARY}}


def group_street(group: str) -> str:
    if group.startswith("pf_") or group in ("vpip", "pfr", "three_bet", "fold_to_three_bet"):
        return "preflop"
    if group in ("wtsd", "wsd"):
        return "all"
    return group.split("_")[0] if group.split("_")[0] in POSTFLOP else "flop"


# ----------------------------------------------------------------------------- positions


_EARLY = ("UTG", "UTG1", "UTG2", "LJ", "HJ", "CO")


def positions(seats: Iterable[int], button: int) -> dict[int, str]:
    """Position names for the seats dealt in. Heads-up: BTN (small blind) and BB."""
    order = sorted(seats)
    i = order.index(button)
    clockwise = order[i:] + order[:i]  # button first
    n = len(order)
    if n == 2:
        return {clockwise[0]: "BTN", clockwise[1]: "BB"}
    names = ["BTN", "SB", "BB"] + list(_EARLY[len(_EARLY) - (n - 3):] if n > 3 else ())
    return dict(zip(clockwise, names))


# ----------------------------------------------------------------------------- contexts


def _street_name(street: Street) -> str:
    return str(street)


def _raises_this_street(view: PlayerView) -> int:
    n = 0
    for e in view.history:
        if e.type is EventType.ACTION and e.street is view.street and e.action in (ActionType.BET,
                                                                                    ActionType.RAISE):
            n += 1
    return n


def decision_context(view: PlayerView, legal: LegalActions) -> tuple[str, tuple[str, ...]]:
    """The context group of the decision ``view`` is at, and which of its
    outcomes are legal here."""
    raises = _raises_this_street(view)
    can_raise = legal.raise_type is not None
    if view.street is Street.PREFLOP:
        if raises == 0:
            if legal.can_check:
                group = "pf_option"
            else:
                group = "pf_open"
        else:
            group = {1: "pf_vs_raise", 2: "pf_vs_3bet"}.get(raises, "pf_vs_4bet")
    else:
        s = _street_name(view.street)
        group = f"{s}_unbet" if raises == 0 else (f"{s}_vs_bet" if raises == 1 else f"{s}_vs_raise")
    legal_outcomes = []
    for o in GROUPS[group]:
        if o == "fold" and legal.can_fold or o == "check" and legal.can_check or o == "call" and legal.can_call \
                or o in ("raise", "bet") and can_raise:
            legal_outcomes.append(o)
    return group, tuple(legal_outcomes)


def outcome_of(action: Action) -> str:
    return {ActionType.FOLD: "fold", ActionType.CHECK: "check", ActionType.CALL: "call",
            ActionType.BET: "bet", ActionType.RAISE: "raise"}[action.type]


# ----------------------------------------------------------------------------- public hands


@dataclass(frozen=True)
class PublicHand:
    """What anyone watching the table knows once the hand is over."""

    button: int
    small_blind: int
    big_blind: int
    start_stacks: Mapping[int, int]
    actions: tuple[tuple[int, Action], ...]  # (seat, action) in order
    showdown: frozenset[int]  # seats that reached showdown
    won_at_showdown: frozenset[int]  # seats awarded chips at showdown

    @classmethod
    def from_view(cls, view: PlayerView) -> PublicHand:
        """From the final observation of any seat (or a spectator)."""
        if not view.is_over:
            raise ValueError("the hand is not over")
        contributed = {s.seat: 0 for s in view.seats}
        awarded = {s.seat: 0 for s in view.seats}
        street_bet = {s.seat: 0 for s in view.seats}
        actions: list[tuple[int, Action]] = []
        showdown: set[int] = set()
        won: set[int] = set()
        for e in view.history:
            if e.type is EventType.DEAL_BOARD:
                street_bet = {s: 0 for s in street_bet}
            elif e.type is EventType.POST_BLIND:
                contributed[e.seat] += e.amount
                street_bet[e.seat] += e.amount
            elif e.type is EventType.RETURN_UNCALLED:
                contributed[e.seat] -= e.amount
                street_bet[e.seat] -= e.amount
            elif e.type is EventType.ACTION:
                if e.action is ActionType.CALL:
                    added = e.amount
                    actions.append((e.seat, Action.call()))
                elif e.action in (ActionType.BET, ActionType.RAISE):
                    added = e.amount - street_bet[e.seat]
                    actions.append((e.seat, Action(e.action, e.amount)))
                else:
                    added = 0
                    actions.append((e.seat, Action(e.action)))
                contributed[e.seat] += added
                street_bet[e.seat] += added
            elif e.type is EventType.SHOWDOWN:
                showdown.add(e.seat)
            elif e.type is EventType.AWARD:
                awarded[e.seat] += e.amount
                if showdown and e.amount > 0:
                    won.add(e.seat)
        start = {s.seat: s.stack - awarded[s.seat] + contributed[s.seat] for s in view.seats}
        return cls(view.button, view.small_blind, view.big_blind, start, tuple(actions),
                   frozenset(showdown), frozenset(won))

    @classmethod
    def from_state(cls, state: HandState) -> PublicHand:
        return cls.from_view(state.observation(None))

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PublicHand:
        """From a JSON hand record (``backend.engine.records``); hole cards are ignored."""
        from backend.engine.records import replay

        return cls.from_state(replay(record))


@dataclass(frozen=True)
class Observation:
    seat: int
    group: str
    outcome: str
    position: str


def observations(hand: PublicHand) -> list[Observation]:
    """Every statistic's opportunities in one hand, with their outcome."""
    state = HandState(dict(hand.start_stacks), hand.button, hand.small_blind, hand.big_blind, Deck(0))
    pos = positions(hand.start_stacks, hand.button)
    out: list[Observation] = []

    def add(seat: int, group: str, outcome: str) -> None:
        out.append(Observation(seat, group, outcome, pos[seat]))

    pf_decided: set[int] = set()
    vpip: set[int] = set()
    pfr: set[int] = set()
    first_vs_raise: set[int] = set()
    opener: int | None = None
    opener_faced_3bet = False
    pf_aggressor: int | None = None
    pf_folded: set[int] = set()
    cbet_chance_done = False
    cbet_made = False
    fold_to_cbet_seen: set[int] = set()
    for seat, action in hand.actions:
        if state.to_act != seat:
            raise ValueError("actions do not replay")
        view = state.observation(seat)
        legal = state.legal_actions()
        group, legal_outcomes = decision_context(view, legal)
        outcome = outcome_of(action)
        if set(legal_outcomes) == set(GROUPS[group]):
            add(seat, group, outcome)
        raises = _raises_this_street(view)
        if view.street is Street.PREFLOP:
            pf_decided.add(seat)
            if action.type is ActionType.FOLD:
                pf_folded.add(seat)
            if action.type in (ActionType.CALL, ActionType.RAISE, ActionType.BET):
                vpip.add(seat)
            if action.type in (ActionType.RAISE, ActionType.BET):
                pfr.add(seat)
                pf_aggressor = seat
                if raises == 0 and opener is None:
                    opener = seat
            if raises == 1 and seat not in first_vs_raise and seat != opener:
                first_vs_raise.add(seat)
                add(seat, "three_bet", "yes" if action.type is ActionType.RAISE else "no")
            if raises == 2 and seat == opener and not opener_faced_3bet:
                opener_faced_3bet = True
                add(seat, "fold_to_three_bet", "yes" if action.type is ActionType.FOLD else "no")
        elif view.street is Street.FLOP:
            if seat == pf_aggressor and not cbet_chance_done and raises == 0:
                cbet_chance_done = True
                cbet_made = action.type is ActionType.BET
                add(seat, "cbet_flop", "yes" if cbet_made else "no")
            elif cbet_made and raises == 1 and seat != pf_aggressor and seat not in fold_to_cbet_seen:
                fold_to_cbet_seen.add(seat)
                add(seat, "fold_to_cbet_flop", "yes" if action.type is ActionType.FOLD else "no")
            if raises == 0 and seat != pf_aggressor and action.type is ActionType.BET:
                cbet_chance_done = True  # someone led into the aggressor: no c-bet chance
        state.apply(action)
    if not state.is_over:
        raise ValueError("the hand did not finish")
    flop_seen = set(hand.start_stacks) - pf_folded if len(state.board) >= 3 else set()
    for seat in sorted(pf_decided):
        add(seat, "vpip", "yes" if seat in vpip else "no")
        add(seat, "pfr", "yes" if seat in pfr else "no")
    for seat in sorted(flop_seen):
        add(seat, "wtsd", "yes" if seat in hand.showdown else "no")
    for seat in sorted(hand.showdown):
        add(seat, "wsd", "yes" if seat in hand.won_at_showdown else "no")
    return out
