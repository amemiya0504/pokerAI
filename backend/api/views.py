"""JSON views sent to browsers. Built only from ``PlayerView`` (never from
``HandState`` internals) so a client can never see another seat's cards."""

from __future__ import annotations

from typing import Any

from backend.engine.actions import LegalActions
from backend.engine.cards import card_str
from backend.engine.evaluator import category_name, evaluate
from backend.engine.hand import EventType, PlayerView


def cards(cs) -> list[str]:
    return [card_str(c) for c in cs]


def event_json(e) -> dict[str, Any]:
    return {
        "type": str(e.type),
        "street": str(e.street),
        "seat": e.seat,
        "action": str(e.action) if e.action else None,
        "amount": e.amount,
        "cards": cards(e.cards),
    }


def legal_json(legal: LegalActions) -> dict[str, Any]:
    return {
        "can_fold": legal.can_fold,
        "can_check": legal.can_check,
        "call_amount": legal.call_amount,
        "raise_type": str(legal.raise_type) if legal.raise_type else None,
        "min_raise_to": legal.min_raise_to,
        "max_raise_to": legal.max_raise_to,
    }


def view_json(view: PlayerView) -> dict[str, Any]:
    """The hand as one seat sees it."""
    shown = {sv.seat: sv.shown_cards for sv in view.seats if sv.shown_cards}
    hand_names = {
        str(s): category_name(evaluate(c + view.board)) for s, c in shown.items()
    } if len(view.board) == 5 else {}
    awards: dict[int, int] = {}
    for e in view.history:
        if e.type is EventType.AWARD:
            awards[e.seat] = awards.get(e.seat, 0) + e.amount
    return {
        "hole_cards": cards(view.hole_cards),
        "board": cards(view.board),
        "street": str(view.street),
        "button": view.button,
        "pot": view.pot,
        "current_bet": view.current_bet,
        "to_act": view.to_act,
        "is_over": view.is_over,
        "seats": {
            str(sv.seat): {
                "stack": sv.stack,
                "street_bet": sv.street_bet,
                "total_bet": sv.total_bet,
                "folded": sv.folded,
                "all_in": sv.all_in,
                "shown_cards": cards(sv.shown_cards) if sv.shown_cards else None,
            }
            for sv in view.seats
        },
        "events": [event_json(e) for e in view.history],
        "awards": {str(s): a for s, a in awards.items()},
        "hand_names": hand_names,
    }
