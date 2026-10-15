"""Preflop agent backed by the solved push/fold datasets (Milestone 4).

Uses the solved strategy only when the situation is inside the solved game:
preflop, every earlier action a fold, an all-in push, or a call of the first
push, and an effective stack close to a solved depth. Everything else
(deeper stacks, limps, non-all-in raises, postflop, cells the solver rarely
reached) falls back to RuleBasedAgent and is labelled HEURISTIC with the
reason, so no solver probability is ever claimed where there is none.
"""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend.engine.actions import Action, ActionType, LegalActions
from backend.engine.hand import EventType, PlayerView, Street
from backend.solver.hands import CLASS_NAMES, class_index
from backend.solver.pushfold import PushFoldGame
from backend.solver.store import PUSHFOLD_DIR, load_all

from .base import Decision, StrategySource
from .rule_based import RuleBasedAgent

MAX_DEPTH_GAP = 0.25  # use a dataset only within 25% of its stack depth


@lru_cache(maxsize=4)
def _datasets(directory: str) -> dict[tuple[int, float], dict[str, Any]]:
    return load_all(Path(directory))


def preflop_order(view: PlayerView) -> list[int]:
    """Seats in push/fold order: UTG .. BTN, SB, BB (heads-up: SB/BTN, BB)."""
    seats = [s.seat for s in view.seats]
    i = seats.index(view.button)
    clockwise = seats[i + 1:] + seats[:i + 1]  # starts left of the button, ends on it
    if len(seats) == 2:
        return [view.button, clockwise[0]]
    sb, bb = clockwise[0], clockwise[1]
    return clockwise[2:] + [sb, bb]


class PushFoldAgent:
    def __init__(self, seed: int = 0, data_dir: str | Path = PUSHFOLD_DIR) -> None:
        self._rng = random.Random(seed)
        self._data = _datasets(str(data_dir))
        self._fallback = RuleBasedAgent(seed)

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        found = self._lookup(view)
        if isinstance(found, str):
            d = self._fallback.act(view, legal)
            return Decision(d.action, StrategySource.HEURISTIC,
                            reason=f"no solved data ({found}); {d.reason}")
        dataset, infoset, q, label = found
        aggress = self._rng.random() < q
        facing_push = "vs" in infoset
        if aggress:
            if facing_push:
                action = Action.call() if legal.can_call else Action.check()
            elif legal.raise_type is not None:
                action = Action(legal.raise_type, legal.max_raise_to)
            else:
                action = Action.call() if legal.can_call else Action.check()
        else:
            action = Action.check() if legal.can_check else Action.fold()
        name = "call" if facing_push else "push"
        return Decision(action, StrategySource.SOLVER_APPROX,
                        action_probs={"fold": round(1 - q, 4), name: q},
                        reason=f"{label} {infoset} {CLASS_NAMES[class_index(*view.hole_cards)]}")

    def _lookup(self, view: PlayerView):
        """(dataset, infoset name, aggress prob, label) or a reason string."""
        if view.street is not Street.PREFLOP:
            return "postflop"
        order = preflop_order(view)
        n = len(order)
        bb = view.big_blind
        start = {s.seat: s.stack + s.total_bet for s in view.seats}
        me = view.seat
        others = max(start[s] for s in order if s != me)
        eff = min(start[me], others) / bb
        depths = sorted(d for (p, d) in self._data if p == n)
        if not depths:
            return f"no {n}-handed dataset"
        depth = min(depths, key=lambda d: abs(d - eff))
        if abs(depth - eff) > MAX_DEPTH_GAP * depth:
            return f"effective stack {eff:.1f}bb outside solved depths {depths}"

        pos = {seat: i for i, seat in enumerate(order)}
        first, mask = -1, 0
        for e in view.history:
            if e.type is not EventType.ACTION or e.street is not Street.PREFLOP:
                continue
            # Preflop a player who moved all-in still has 0 behind.
            all_in = next(s for s in view.seats if s.seat == e.seat).stack == 0
            if e.action is ActionType.FOLD:
                continue
            if first < 0 and e.action in (ActionType.RAISE, ActionType.BET) and all_in:
                first = pos[e.seat]
            elif first >= 0 and e.action is ActionType.CALL:
                mask |= 1 << pos[e.seat]
            else:
                return f"off-tree action: {e.action} by seat {e.seat}"
        game = PushFoldGame(n, depth)
        name = game.infoset_name((pos[me], first, mask))
        data = self._data[(n, depth)]
        cell = data["strategy"].get(name)
        if cell is None:
            return f"infoset {name} not in dataset"
        q = cell["aggress_prob"][class_index(*view.hole_cards)]
        if q is None:
            return f"{name} not converged for this hand"
        return data, name, q, f"pushfold {n}max {depth:g}bb (eff {eff:.1f}bb)"
