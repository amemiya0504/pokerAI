"""JSON hand records and replay.

The source of truth for a hand is (deck seed, button, blinds, starting
stacks, action sequence). Everything else in a record is derived and is
kept only for readability and analysis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .actions import Action, ActionType
from .cards import Deck, cards_str
from .hand import EventType, HandState, Street

FORMAT = "poker-ai-hand/1"


def hand_record(
    state: HandState,
    *,
    hand_no: int = 0,
    decisions: Sequence[Mapping[str, Any] | None] | None = None,
    player_ids: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Serialize a finished hand. ``decisions[i]`` holds the optional
    decision metadata (strategy_source, action_probs, reason) for action i."""
    result = state.result()
    starts = state.starting_stacks()
    streets = _action_streets(state)
    actions = []
    for i, (seat, action) in enumerate(state.actions):
        meta = dict(decisions[i] or {}) if decisions else {}
        actions.append({
            "seq": i,
            "seat": seat,
            "street": str(streets[i]),
            "type": str(action.type),
            "amount": action.amount,
            "strategy_source": meta.get("strategy_source"),
            "action_probs": meta.get("action_probs"),
            "reason": meta.get("reason"),
        })
    return {
        "format": FORMAT,
        "hand_no": hand_no,
        "deck_seed": state.deck_seed,
        "button": state.button,
        "small_blind": state.small_blind,
        "big_blind": state.big_blind,
        "players": [
            {
                "seat": s,
                "player_id": (player_ids or {}).get(s),
                "start_stack": starts[s],
                "hole_cards": cards_str(state.hole_cards(s)),
            }
            for s in state.seats
        ],
        "actions": actions,
        "board": cards_str(result.board),
        "result": {
            "pots": [
                {
                    "amount": p.amount,
                    "eligible": list(p.eligible),
                    "winners": list(p.winners),
                    "shares": {str(s): c for s, c in p.shares},
                }
                for p in result.pots
            ],
            "final_stacks": {str(s): v for s, v in result.final_stacks.items()},
            "net": {str(s): v for s, v in result.net.items()},
            "showdown": {
                str(s): {"cards": cards_str(c), "hand": result.hand_names[s]}
                for s, c in result.showdown.items()
            },
        },
    }


def replay(record: Mapping[str, Any]) -> HandState:
    """Rebuild a hand from its seed and action sequence."""
    if record.get("format") != FORMAT:
        raise ValueError(f"unsupported record format {record.get('format')!r}")
    state = HandState(
        {p["seat"]: p["start_stack"] for p in record["players"]},
        record["button"],
        record["small_blind"],
        record["big_blind"],
        Deck(record["deck_seed"]),
    )
    for a in record["actions"]:
        if state.to_act != a["seat"]:
            raise ValueError(f"action {a['seq']}: expected seat {state.to_act}, got {a['seat']}")
        state.apply(Action(ActionType(a["type"]), a["amount"]))
    if not state.is_over:
        raise ValueError("record ends before the hand is over")
    return state


def _action_streets(state: HandState) -> list[Street]:
    return [e.street for e in state.events if e.type is EventType.ACTION]
