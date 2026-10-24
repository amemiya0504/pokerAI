"""Style agents with known action frequencies (Milestone 8 acceptance).

In every decision context of ``backend.opponent.stats`` the agent picks an
outcome with fixed probabilities that do not depend on its cards, so the
true frequency of each context statistic is exactly the parameter (when
every outcome is legal; otherwise the legal ones are renormalised, and the
statistics do not count that decision). They are test opponents for the
opponent model, not strategies: the decisions are labelled HEURISTIC.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from backend.engine.actions import Action, ActionType, LegalActions
from backend.engine.hand import PlayerView, Street
from backend.opponent.stats import CONTEXTS, POSTFLOP, decision_context

from .base import Decision, StrategySource


def _style(open_, option, vs_raise, vs_3bet, vs_4bet, unbet, vs_bet, vs_reraise) -> dict[str, tuple[float, ...]]:
    p = {"pf_open": open_, "pf_option": option, "pf_vs_raise": vs_raise, "pf_vs_3bet": vs_3bet,
         "pf_vs_4bet": vs_4bet}
    for s in POSTFLOP:
        p[f"{s}_unbet"], p[f"{s}_vs_bet"], p[f"{s}_vs_raise"] = unbet, vs_bet, vs_reraise
    return p


# Outcome order follows ``CONTEXTS``: (fold, call, raise), (check, raise), (check, bet).
STYLES: dict[str, dict[str, tuple[float, ...]]] = {
    "nit": _style((0.80, 0.08, 0.12), (0.90, 0.10), (0.75, 0.20, 0.05), (0.70, 0.20, 0.10), (0.60, 0.30, 0.10),
                  (0.70, 0.30), (0.55, 0.40, 0.05), (0.60, 0.35, 0.05)),
    "tag": _style((0.55, 0.05, 0.40), (0.70, 0.30), (0.55, 0.30, 0.15), (0.50, 0.35, 0.15), (0.45, 0.35, 0.20),
                  (0.45, 0.55), (0.45, 0.40, 0.15), (0.50, 0.35, 0.15)),
    "lag": _style((0.25, 0.10, 0.65), (0.50, 0.50), (0.35, 0.35, 0.30), (0.35, 0.40, 0.25), (0.40, 0.40, 0.20),
                  (0.35, 0.65), (0.30, 0.45, 0.25), (0.40, 0.40, 0.20)),
    "station": _style((0.20, 0.75, 0.05), (0.95, 0.05), (0.15, 0.82, 0.03), (0.15, 0.82, 0.03), (0.20, 0.78, 0.02),
                      (0.85, 0.15), (0.10, 0.85, 0.05), (0.15, 0.82, 0.03)),
    "maniac": _style((0.05, 0.10, 0.85), (0.20, 0.80), (0.10, 0.30, 0.60), (0.10, 0.35, 0.55), (0.15, 0.35, 0.50),
                     (0.15, 0.85), (0.10, 0.40, 0.50), (0.15, 0.40, 0.45)),
}


class StyleAgent:
    def __init__(self, seed: int = 0, style: str = "tag",
                 probabilities: Mapping[str, tuple[float, ...]] | None = None) -> None:
        if probabilities is None:
            if style not in STYLES:
                raise ValueError(f"unknown style {style!r}; known: {sorted(STYLES)}")
            probabilities = STYLES[style]
        for g, p in probabilities.items():
            if len(p) != len(CONTEXTS[g]) or abs(sum(p) - 1) > 1e-9 or min(p) < 0:
                raise ValueError(f"bad probabilities for {g}: {p}")
        missing = set(CONTEXTS) - set(probabilities)
        if missing:
            raise ValueError(f"missing contexts: {sorted(missing)}")
        self.style, self.probabilities = style, dict(probabilities)
        self._rng = random.Random(seed)

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        group, legal_outcomes = decision_context(view, legal)
        outs = CONTEXTS[group]
        probs = {o: p for o, p in zip(outs, self.probabilities[group]) if o in legal_outcomes}
        total = sum(probs.values())
        if total <= 0:  # every outcome with weight is illegal: take the passive one
            probs = {o: 1.0 for o in legal_outcomes if o in ("check", "call", "fold")}
            total = sum(probs.values())
        probs = {o: p / total for o, p in probs.items()}
        pick = self._rng.choices(list(probs), weights=list(probs.values()))[0]
        return Decision(self._action(pick, view, legal), StrategySource.HEURISTIC,
                        action_probs={o: round(p, 4) for o, p in probs.items()},
                        reason=f"style agent '{self.style}' in context {group} (frequencies independent of cards)")

    @staticmethod
    def _action(outcome: str, view: PlayerView, legal: LegalActions) -> Action:
        if outcome == "fold":
            return Action.fold()
        if outcome == "check":
            return Action.check()
        if outcome == "call":
            return Action.call()
        if view.street is Street.PREFLOP:
            to = 5 * view.big_blind // 2 if view.current_bet <= view.big_blind else 3 * view.current_bet
        elif view.current_bet == 0:
            to = round(view.pot * 2 / 3)
        else:
            to = 3 * view.current_bet
        to = min(max(to, legal.min_raise_to), legal.max_raise_to)
        return Action(legal.raise_type or ActionType.BET, to)
