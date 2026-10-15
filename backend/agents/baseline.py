"""Baseline agents for tests and stress simulation only."""

from __future__ import annotations

import random

from backend.engine.actions import Action, ActionType, LegalActions
from backend.engine.hand import PlayerView

from .base import Decision, StrategySource


class RandomAgent:
    """Picks a legal action type uniformly, then a uniform raise size.

    ``all_in_weight`` biases raises toward all-in so that side pots and
    incomplete raises show up often in fuzz tests.
    """

    def __init__(self, seed: int, all_in_weight: float = 0.2) -> None:
        self._rng = random.Random(seed)
        self._all_in_weight = all_in_weight

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        rng = self._rng
        kind = rng.choice(legal.types())
        if kind in (ActionType.BET, ActionType.RAISE):
            assert legal.min_raise_to is not None and legal.max_raise_to is not None
            if rng.random() < self._all_in_weight:
                amount = legal.max_raise_to
            else:
                amount = rng.randint(legal.min_raise_to, legal.max_raise_to)
            action = Action(kind, amount)
        else:
            action = Action(kind)
        return Decision(action, StrategySource.RANDOM)


class CallingStationAgent:
    """Always checks or calls; never folds or raises."""

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        action = Action.check() if legal.can_check else Action.call()
        return Decision(action, StrategySource.HEURISTIC, reason="calling station")
