"""A simple deterministic rule-based agent (HEURISTIC, not GTO).

Preflop it ranks hands with the Chen formula; after the flop it uses the
made hand (only counting what the hole cards add to the board) and pot odds.

``board_aware=True`` (the live snapshot API's fallback, 2026-11-12) fixes what
live hands showed: a pair on a paired board is one pair, not two pair (As5s on
Qc Qd 5d raised as "strong"); top pair with a kicker below a ten calls the
flop but folds bigger turn and river bets; a third card of a suit on the board
downgrades a pair. Off by default: experiments use this agent as an opponent
and as a fallback, and their accepted results depend on it as it was.
It also (2026-11-14) mixes a small probe with thin hands and a semi-bluff with
flush draws when checked to heads-up in position on an unpaired flop or turn,
seeded per decision, and checks / calls instead of inflating a bet below the
legal minimum. Its draws count straight outs too (an open-ender is a draw, a
lone gutshot is not), turn draws call only what their outs pay for (a pair's
two-pair and trips cards included), and a flop
combo draw (12+ outs) jams at SPR 1.5 or less with at most three players.
"""

from __future__ import annotations

import random

from backend.engine.actions import Action, LegalActions
from backend.engine.evaluator import FLUSH, ONE_PAIR, STRAIGHT, THREE_OF_A_KIND, TWO_PAIR, category, evaluate
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


TEN = 8  # rank index: 2..A are 0..12

# shortcut: engineering priors, not solver output; calibrate on live replays before raising them.
PROBE_THIN = 0.35  # thin made hand: bet 25% pot
SEMI_BLUFF_DRAW = 0.33  # flush draw / open-ender: bet 40% pot
DRAW_OUTS = 8  # an open-ender or a flush draw; a lone gutshot stays weak
COMBO_OUTS = 12  # e.g. a flush draw plus a gutshot: about 45% with two cards to come
JAM_SPR = 1.5  # stack behind / pot: at or below it a flop combo draw is committed anyway, so it jams


def _draw_outs(hole: tuple[int, ...], board: tuple[int, ...]) -> int:
    """Unseen cards that give the hole cards a straight or better they do not have yet (flop and turn).
    shortcut: counts non-nut and board-pairing outs as clean; discount them if overcalls show up."""
    if len(board) >= 5 or category(evaluate(hole + board)) >= STRAIGHT:
        return 0
    known = set(hole + board)
    outs = 0
    for c in range(52):
        if c in known:
            continue
        made = category(evaluate(hole + board + (c,)))
        if made >= STRAIGHT and (len(board) < 4 or made > category(evaluate(board + (c,)))):
            outs += 1
    return outs


def _pair_outs(hole: tuple[int, ...], board: tuple[int, ...]) -> int:
    """Unseen cards of a hole rank that lift our own pair to two pair or trips (JT on K Q 5 J: 5)."""
    made = category(evaluate(hole + board))
    if made != ONE_PAIR or made <= _board_category(board):
        return 0
    known = set(hole + board)
    return sum(1 for c in range(52) if c not in known and c >> 2 in {h >> 2 for h in hole}
               and category(evaluate(hole + board + (c,))) in (TWO_PAIR, THREE_OF_A_KIND))


def _spr(v: PlayerView) -> float:
    me = next(s for s in v.seats if s.seat == v.seat)
    return me.stack / max(1, v.pot)


def _last_to_act(v: PlayerView) -> bool:
    """No live seat that can still act sits after the hero this street (postflop: clockwise from the
    button); False when the button is unknown."""
    if v.button is None:
        return False
    order = lambda seat: (seat - v.button - 1) % 1000  # noqa: E731
    return all(order(s.seat) < order(v.seat) for s in v.seats
               if s.seat != v.seat and not s.folded and not s.all_in)


def _board_aware_tier(hole: tuple[int, ...], board: tuple[int, ...]) -> str:
    """strong / medium / thin / draw / weak from what the hole cards really add."""
    made = category(evaluate(hole + board))
    board_ranks = [c >> 2 for c in board]
    hole_ranks = [c >> 2 for c in hole]
    tier = "weak"
    if made > _board_category(board):
        board_paired = len(set(board_ranks)) < len(board_ranks)
        if made >= THREE_OF_A_KIND or made == TWO_PAIR and not board_paired:
            tier = "strong"
        else:  # one pair of our own (a paired board's second pair is not ours)
            mine = max(r for r in hole_ranks if r in board_ranks or hole_ranks.count(r) == 2)
            top = max(board_ranks)
            if hole_ranks[0] == hole_ranks[1] and mine > top:
                tier = "medium"  # overpair
            elif mine == top:
                kicker = max(r for r in hole_ranks if r != mine) if len(set(hole_ranks)) == 2 else mine
                tier = "medium" if kicker >= TEN else "thin"
    if tier == "weak" and _draw_outs(hole, board) >= DRAW_OUTS:
        return "draw"
    suited = max(sum(1 for c in board if c & 3 == s) for s in range(4))
    if len(board) >= 4 and suited >= 3 and made < FLUSH:  # a flush can be there and we do not have one
        tier = {"medium": "thin", "thin": "weak"}.get(tier, tier)
    return tier


class RuleBasedAgent:
    def __init__(self, seed: int = 0, open_threshold: int = 8, board_aware: bool = False) -> None:
        self.open_threshold = open_threshold
        self.board_aware = board_aware
        self._rng = random.Random(seed)

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        if view.street is Street.PREFLOP:
            action, reason = self._preflop(view, legal)
        else:
            action, reason = self._postflop(view, legal)
        return Decision(action, StrategySource.HEURISTIC, reason=reason)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _raise(legal: LegalActions, to: float, strict: bool = False) -> Action | None:
        """``strict``: None when ``to`` is below the legal minimum (a short all-in still counts)."""
        if legal.raise_type is None:
            return None
        if strict and int(to) < min(legal.min_raise_to, legal.max_raise_to):
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
        if self.board_aware:
            return self._postflop_board_aware(v, legal)
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

    def _postflop_board_aware(self, v: PlayerView, legal: LegalActions) -> tuple[Action, str]:
        tier = _board_aware_tier(v.hole_cards, v.board)
        pot = v.pot
        outs = _draw_outs(v.hole_cards, v.board) if tier == "draw" else 0
        live = sum(1 for s in v.seats if not s.folded)
        # 2026-11-14 h43: AcJc on Th 4c Qc, SPR 0.8, checked and called off; a jam adds fold equity
        committed_combo = (tier == "draw" and v.street is Street.FLOP and outs >= COMBO_OUTS and live <= 3
                           and _spr(v) <= JAM_SPR)
        if committed_combo:
            act = self._raise(legal, legal.max_raise_to or 0, strict=True)
            if act:
                return act, f"combo draw ({outs} outs), SPR {_spr(v):.2f}: jam"
            if not legal.can_check:
                return Action.call(), f"combo draw ({outs} outs), SPR {_spr(v):.2f}: call (cannot raise)"
        if legal.can_check:
            if tier == "strong":
                act = self._raise(legal, v.current_bet + 0.75 * pot, strict=True)
                return act or Action.check(), "value bet"
            if tier == "medium" and live <= 3:
                act = self._raise(legal, v.current_bet + 0.33 * pot, strict=True)
                return act or Action.check(), "thin value bet"
            if (tier in ("thin", "draw") and live == 2 and v.street is not Street.RIVER
                    and len({c >> 2 for c in v.board}) == len(v.board) and _last_to_act(v)):
                freq, frac = (PROBE_THIN, 0.25) if tier == "thin" else (SEMI_BLUFF_DRAW, 0.40)
                if self._rng.random() < freq:
                    act = self._raise(legal, v.current_bet + frac * pot, strict=True)
                    if act:
                        return act, f"{tier}: heads-up in position, bet {frac:.0%} pot (mix {freq})"
                return Action.check(), f"{tier}: check (heads-up in position, mix {freq})"
            return Action.check(), f"{tier}: check"
        price = legal.call_amount / (pot + legal.call_amount)
        if tier == "strong":
            act = self._raise(legal, v.current_bet * 3, strict=True)
            return act or Action.call(), "strong: raise"
        if tier == "medium":
            return self._passive(legal, price <= 0.33), f"medium: price {price:.2f}"
        if tier == "thin":  # weak kicker / scare card: the flop, or a small bet later
            limit = 0.33 if v.street is Street.FLOP else 0.2
            return self._passive(legal, price <= limit), f"thin: price {price:.2f}"
        if tier == "draw":
            # one card to come: the price has to be covered by the outs alone, a pair's improvers included
            # (2026-11-14 h158: JT on K Q 5 J folded 13 outs at 0.25 counting only the 8 straight cards)
            if v.street is not Street.FLOP:
                outs += _pair_outs(v.hole_cards, v.board)
            limit = 0.25 if v.street is Street.FLOP else min(0.25, outs / 46)
            return self._passive(legal, price <= limit), f"draw ({outs} outs): price {price:.2f}"
        return Action.fold(), "weak: fold"
