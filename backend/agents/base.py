"""Agent interface: an agent sees only a PlayerView and returns a Decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from backend.engine.actions import Action, LegalActions
from backend.engine.hand import PlayerView


class StrategySource(StrEnum):
    """Where a decision's strategy comes from, strongest guarantee first.

    Only VERIFIED_EQUILIBRIUM may be called exact GTO.
    """

    VERIFIED_EQUILIBRIUM = "verified_equilibrium"
    SOLVER_APPROX = "solver_approx"
    NEURAL_APPROX = "neural_approx"
    HEURISTIC = "heuristic"
    RANDOM = "random"


@dataclass(frozen=True)
class Decision:
    action: Action
    strategy_source: StrategySource
    action_probs: dict[str, float] | None = None
    action_ev: dict[str, float] | None = None
    reason: str | None = None


class Agent(Protocol):
    def act(self, view: PlayerView, legal: LegalActions) -> Decision: ...
