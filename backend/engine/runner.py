"""Minimal loop wiring a Table to agents. Milestone 3 grows this into Arena.

Agents are duck-typed (``act(view, legal) -> decision`` where the decision
has ``.action`` and optionally ``.strategy_source``, ``.action_probs`` and
``.reason``) so that the engine does not import the agents package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .records import hand_record
from .table import Table


def run_hand(table: Table, agents: Mapping[int, Any]) -> dict[str, Any]:
    """Play one hand to the end, settle it, and return its JSON record."""
    state = table.start_hand()
    decisions: list[dict[str, Any] | None] = []
    while not state.is_over:
        seat = state.to_act
        legal = state.legal_actions()
        if table.is_away(seat):
            state.apply(table.away_action(legal))
            decisions.append({"reason": "away: auto check/fold"})
            continue
        decision = agents[seat].act(state.observation(seat), legal)
        state.apply(decision.action)
        source = getattr(decision, "strategy_source", None)
        decisions.append({
            "strategy_source": str(source) if source is not None else None,
            "action_probs": getattr(decision, "action_probs", None),
            "reason": getattr(decision, "reason", None),
        })
    player_ids = {s: table.seats[s].player_id for s in state.seats}
    record = hand_record(state, hand_no=table.hand_no, decisions=decisions,
                         player_ids=player_ids)
    table.finish_hand()
    return record


def run_session(table: Table, agents: Mapping[int, Any], n_hands: int) -> list[dict[str, Any]]:
    """Play up to ``n_hands``, stopping early when fewer than 2 can play."""
    records = []
    for _ in range(n_hands):
        if len(table.eligible_seats()) < 2:
            break
        records.append(run_hand(table, agents))
    return records
