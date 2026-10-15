"""A simple deterministic rule-based agent (HEURISTIC, not GTO).

Preflop it ranks hands with the Chen formula; after the flop it uses the
made hand (only counting what the hole cards add to the board) and pot odds.
"""

from __future__ import annotations

from backend.engine.actions import Action, LegalActions
from backend.engine.evaluator import ONE_PAIR, TWO_PAIR, category, evaluate
from backend.engine.hand import PlayerView, Street

from .base import Decision, StrategySource

_CHEN_HIGH = (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10)  # 2..A


def chen_score(c1: int, c2: int) -> int:
    r1, r2 = sorted((c1 >> 2, c2 >> 2), reverse=True)
    if r1 == r2:
        score = max(5.0, _CHEN_HIGH[r1] * 2)
    else:
        score = _CHEN_HIGH[r1]
        if c1 & 3 == c2 & 3:
            score += 2
        gap = r1 - r2 - 1
        score -= (0, 1, 2, 4)[gap] if gap < 4 else 5
        if gap <= 1 and r1 < 10:  # both below a queen
            score += 1
    return int(-(-score // 1))  # round half up to an integer


def _board_category(board: tuple[int, ...]) -> int:
    """Pair-type category the board makes on its own (flush/straight only on 5 cards)."""
    if len(board) == 5:
        return category(evaluate(board))
    counts = sorted((sum(1 for c in board if c >> 2 == r) for r in {c >> 2 for c in board}), reverse=True)
    if counts[0] >= 3:
        return 3
    if counts[0] == 2:
        return TWO_PAIR if len(counts) > 1 and counts[1] == 2 else ONE_PAIR
    return 0


def _flush_draw(hole: tuple[int, ...], board: tuple[int, ...]) -> bool:
    if len(board) >= 5:
        return False
    for suit in {c & 3 for c in hole}:
        if sum(1 for c in hole + board if c & 3 == suit) == 4:
            return True
    return False


class RuleBasedAgent:
    def __init__(self, seed: int = 0, open_threshold: int = 8) -> None:
        self.open_threshold = open_threshold

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        if view.street is Street.PREFLOP:
            action, reason = self._preflop(view, legal)
        else:
            action, reason = self._postflop(view, legal)
        return Decision(action, StrategySource.HEURISTIC, reason=reason)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _raise(legal: LegalActions, to: float) -> Action | None:
        if legal.raise_type is None:
            return None
        amount = min(max(int(to), legal.min_raise_to), legal.max_raise_to)
        return Action(legal.raise_type, amount)

    @staticmethod
    def _passive(legal: LegalActions, call_ok: bool) -> Action:
        if legal.can_check:
            return Action.check()
        return Action.call() if call_ok else Action.fold()

    def _preflop(self, v: PlayerView, legal: LegalActions) -> tuple[Action, str]:
        score = chen_score(*v.hole_cards)
        players = sum(1 for s in v.seats if not s.folded)
        threshold = self.open_threshold + (1 if players > 6 else 0)
        bb = v.big_blind
        raised = v.current_bet > bb
        if not raised:
            if score >= threshold:
                limpers = sum(1 for s in v.seats if s.street_bet == bb) - 1
                act = self._raise(legal, bb * (3 + max(limpers, 0)))
                if act:
                    return act, f"chen {score}: open raise"
            return self._passive(legal, score >= threshold - 2), f"chen {score}: limp/check"
        if score >= 11:
            act = self._raise(legal, v.current_bet * 3)
            if act:
                return act, f"chen {score}: 3-bet"
            return Action.call(), f"chen {score}: call (cannot raise)"
        me = next(s for s in v.seats if s.seat == v.seat)
        cheap = legal.call_amount <= 0.15 * (me.stack + me.street_bet)
        return self._passive(legal, score >= threshold + 1 and cheap), f"chen {score}: facing raise"

    def _postflop(self, v: PlayerView, legal: LegalActions) -> tuple[Action, str]:
        hole, board = v.hole_cards, v.board
        made = category(evaluate(hole + board))
        uses_hole = made > _board_category(board)
        top_board = max(c >> 2 for c in board)
        hole_ranks = [c >> 2 for c in hole]
        if uses_hole and made >= TWO_PAIR:
            tier = "strong"
        elif uses_hole and made == ONE_PAIR and (
            hole_ranks[0] == hole_ranks[1] and hole_ranks[0] > top_board or top_board in hole_ranks
        ):
            tier = "medium"  # overpair or top pair
        elif _flush_draw(hole, board):
            tier = "draw"
        else:
            tier = "weak"

        pot = v.pot
        if legal.can_check:
            if tier == "strong":
                act = self._raise(legal, v.current_bet + 0.75 * pot)
                return act or Action.check(), "value bet"
            if tier == "medium" and sum(1 for s in v.seats if not s.folded) <= 3:
                act = self._raise(legal, v.current_bet + 0.33 * pot)
                return act or Action.check(), "thin value bet"
            return Action.check(), f"{tier}: check"
        price = legal.call_amount / (pot + legal.call_amount)
        if tier == "strong":
            act = self._raise(legal, v.current_bet * 3)
            return act or Action.call(), "strong: raise"
        if tier == "medium":
            return self._passive(legal, price <= 0.33), f"medium: price {price:.2f}"
        if tier == "draw":
            return self._passive(legal, price <= 0.25), f"draw: price {price:.2f}"
        return Action.fold(), "weak: fold"
