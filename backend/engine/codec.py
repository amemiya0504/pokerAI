"""Plain-data (JSON-compatible) encoding of what an agent may see and do.

Agents behind a serialization boundary receive only data rebuilt from these
dicts, never objects shared with the engine.
"""

from __future__ import annotations

from typing import Any

from .actions import Action, ActionType, LegalActions
from .hand import Event, EventType, PlayerView, SeatView, Street


_EVENT = {e.value: e for e in EventType}
_ACTION = {a.value: a for a in ActionType}
_STREET = tuple(Street)


def view_to_dict(v: PlayerView) -> dict[str, Any]:
    return {
        "seat": v.seat,
        "hole_cards": list(v.hole_cards),
        "board": list(v.board),
        "street": int(v.street),
        "button": v.button,
        "small_blind": v.small_blind,
        "big_blind": v.big_blind,
        "pot": v.pot,
        "current_bet": v.current_bet,
        "to_act": v.to_act,
        "seats": [
            [s.seat, s.stack, s.street_bet, s.total_bet, s.folded, s.all_in,
             list(s.shown_cards) if s.shown_cards is not None else None]
            for s in v.seats
        ],
        "history": [
            [str(e.type), int(e.street), e.seat, str(e.action) if e.action else None,
             e.amount, list(e.cards)]
            for e in v.history
        ],
        "is_over": v.is_over,
    }


def view_from_dict(d: dict[str, Any]) -> PlayerView:
    return PlayerView(
        seat=d["seat"],
        hole_cards=tuple(d["hole_cards"]),
        board=tuple(d["board"]),
        street=Street(d["street"]),
        button=d["button"],
        small_blind=d["small_blind"],
        big_blind=d["big_blind"],
        pot=d["pot"],
        current_bet=d["current_bet"],
        to_act=d["to_act"],
        seats=tuple(
            SeatView(seat, stack, sb, tb, folded, all_in,
                     tuple(shown) if shown is not None else None)
            for seat, stack, sb, tb, folded, all_in, shown in d["seats"]
        ),
        history=tuple(
            Event(_EVENT[t], _STREET[st], seat, _ACTION[a] if a else None, amount, tuple(cards))
            for t, st, seat, a, amount, cards in d["history"]
        ),
        is_over=d["is_over"],
    )


def legal_to_dict(l: LegalActions) -> dict[str, Any]:
    return {
        "seat": l.seat,
        "can_fold": l.can_fold,
        "can_check": l.can_check,
        "call_amount": l.call_amount,
        "raise_type": str(l.raise_type) if l.raise_type else None,
        "min_raise_to": l.min_raise_to,
        "max_raise_to": l.max_raise_to,
    }


def legal_from_dict(d: dict[str, Any]) -> LegalActions:
    return LegalActions(
        seat=d["seat"],
        can_fold=d["can_fold"],
        can_check=d["can_check"],
        call_amount=d["call_amount"],
        raise_type=ActionType(d["raise_type"]) if d["raise_type"] else None,
        min_raise_to=d["min_raise_to"],
        max_raise_to=d["max_raise_to"],
    )


def action_to_dict(a: Action) -> dict[str, Any]:
    return {"type": str(a.type), "amount": a.amount}


def action_from_dict(d: dict[str, Any]) -> Action:
    return Action(ActionType(d["type"]), int(d["amount"]))
