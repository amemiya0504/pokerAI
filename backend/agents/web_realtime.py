"""Web-only policy: frozen preflop, existing HU solver, multiway heuristic."""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from backend.engine.actions import Action, LegalActions
from backend.engine.cards import derive_seed
from backend.engine.hand import PlayerView, Street
from backend.solver.hands import class_index

from .base import Decision, StrategySource
from .pushfold import PushFoldAgent
from .realtime import RealtimeAgent, strength_percentile
from .web_preflop import effective_bb, lookup, ranges, tables


@lru_cache(maxsize=64)
def _board_strength(board: tuple[int, ...]) -> np.ndarray:
    return strength_percentile(board)


def _public_ranges(view: PlayerView) -> dict[int, np.ndarray]:
    """Public-action-conditioned combo weights for every seat."""
    from .realtime import heuristic_likelihood, replay

    weights = ranges(view, PushFoldAgent(0))
    actions, _, _ = replay(view)
    for action in actions:
        if action.seat != view.seat and not next(s for s in view.seats if s.seat == action.seat).folded:
            likelihood = heuristic_likelihood(action.action, _board_strength(tuple(action.before.board)))
            weights[action.seat] *= likelihood
    return weights


def equity(view: PlayerView, samples: int = 192) -> float:
    """Range-vs-range CPU Monte Carlo with card blockers and split-pot equity."""
    from backend.engine.fast_eval import CARD_A, CARD_B, evaluate_many

    opponents = [s.seat for s in view.seats if not s.folded and s.seat != view.seat]
    if not opponents:
        return 1.0
    weights = _public_ranges(view)
    missing = 5 - len(view.board)
    rng = np.random.default_rng(derive_seed(37, *view.hole_cards, *view.board, len(opponents)))
    holes = np.empty((samples, len(opponents), 2), dtype=np.int64)
    runouts = np.empty((samples, missing), dtype=np.int64)
    known = set(view.board + view.hole_cards)
    for sample in range(samples):
        used = set(known)
        order = rng.permutation(len(opponents))
        for index in order:
            valid = np.flatnonzero(~np.isin(CARD_A, tuple(used)) & ~np.isin(CARD_B, tuple(used)))
            mass = weights[opponents[index]][valid]
            if mass.sum() <= 0:
                mass = np.ones(len(valid))
            combo = int(rng.choice(valid, p=mass / mass.sum()))
            a, b = int(CARD_A[combo]), int(CARD_B[combo])
            holes[sample, index] = (a, b)
            used.update((a, b))
        available = np.array([c for c in range(52) if c not in used], dtype=np.int64)
        if missing:
            runouts[sample] = rng.choice(available, size=missing, replace=False)

    board = np.concatenate([np.tile(view.board, (samples, 1)), runouts], axis=1)
    hero = evaluate_many(np.concatenate([np.tile(view.hole_cards, (samples, 1)), board], axis=1))
    other = np.stack([evaluate_many(np.concatenate([holes[:, i], board], axis=1))
                      for i in range(len(opponents))], axis=1)
    best = np.maximum(hero, other.max(axis=1))
    return float(np.mean((hero == best) / (1 + (other == hero[:, None]).sum(axis=1))))


def _multiway_bluff(view: PlayerView, legal: LegalActions, eq: float) -> dict[str, float] | None:
    if not view.board or not legal.can_check or not legal.can_raise or view.pot <= 0:
        return None

    from backend.engine.fast_eval import CARD_A, CARD_B

    me = next(s for s in view.seats if s.seat == view.seat)
    to = min(legal.max_raise_to, max(legal.min_raise_to, me.street_bet + round(.70 * view.pot)))
    bet = to - me.street_bet
    if bet <= 0:
        return None

    board = tuple(view.board)
    on_board = ~np.isin(CARD_A, board) & ~np.isin(CARD_B, board)
    unblocked = on_board.copy()
    for card in view.hole_cards:
        unblocked &= (CARD_A != card) & (CARD_B != card)

    strength = _board_strength(board)
    call_threshold = bet / (view.pot + 2 * bet)
    ranges_by_seat = _public_ranges(view)
    opponents = [s.seat for s in view.seats
                 if s.seat != view.seat and not s.folded and not s.all_in and s.stack > 0]
    if not opponents:
        return None

    fold_equity = fold_equity_without_blockers = 1.0
    for seat in opponents:
        weights = ranges_by_seat[seat]
        total = float(weights[unblocked].sum())
        base_total = float(weights[on_board].sum())
        if total <= 0 or base_total <= 0:
            return None
        calls = float(weights[unblocked & (strength >= call_threshold)].sum()) / total
        base_calls = float(weights[on_board & (strength >= call_threshold)].sum()) / base_total
        fold_equity *= 1 - calls
        fold_equity_without_blockers *= 1 - base_calls

    # shortcut: made-hand percentile proxies call equity and opponents fold independently; upgrade with calibrated joint response ranges.
    # shortcut: showdown equity proxies check EV; future betting is unmodeled.
    check_ev = eq * view.pot
    called_ev = eq * (view.pot + 2 * bet) - bet
    ev = fold_equity * view.pot + (1 - fold_equity) * called_ev
    improvement_room = view.pot - check_ev
    # shortcut: EV advantage is a non-equilibrium mix proxy; replace with a solved response model before enabling.
    mix = min(1.0, max(0.0, (ev - check_ev) / improvement_room)) if improvement_room > 0 else 0.0
    return {"ev": ev, "check_ev": check_ev, "mix": mix, "fold_equity": fold_equity,
            "blocker_delta": fold_equity - fold_equity_without_blockers,
            "bet": float(bet)}


def heuristic(view: PlayerView, legal: LegalActions, rng: random.Random | None = None,
              allow_bluff: bool = False) -> Decision:
    n = sum(not s.folded for s in view.seats)
    eq = equity(view)
    price = legal.call_amount / max(1, view.pot + legal.call_amount)
    # shortcut: call EV assumes showdown equity realizes fully; upgrade when multi-street realization is modeled.
    call_ev = eq * (view.pot + legal.call_amount) - legal.call_amount
    bluff = None
    if legal.can_check:
        value = eq > max(.50, 1 / n + .13)
        raising = float(value)
        if allow_bluff and not value:
            bluff = _multiway_bluff(view, legal, eq)
            raising = bluff["mix"] if bluff else 0.0
        fold, passive = 0.0, 1 - raising
    else:
        # A call has non-negative chip EV only when range equity covers the pot odds.
        continuing = float(call_ev > 0)
        raising = 0.0
        fold, passive = 1 - continuing, continuing
    if not legal.can_raise:
        passive += raising
        raising = 0.0
    if fold == 1:
        action = Action.fold()
    elif raising >= 1 or (raising > 0 and (rng or random.Random(0)).random() < raising):
        me = next(s for s in view.seats if s.seat == view.seat)
        to = me.street_bet + legal.call_amount + round(.70 * (view.pot + legal.call_amount))
        action = Action(legal.raise_type, min(legal.max_raise_to, max(legal.min_raise_to, to)))
    else:
        action = Action.check() if legal.can_check else Action.call()
    action_ev = {"check": bluff["check_ev"], "raise": bluff["ev"]} if bluff else None
    reason = f"HEURISTIC multiway ({n} live): range equity {eq:.3f}, price {price:.3f}, call EV {call_ev:.1f}"
    if bluff:
        reason += (f"; estimated bluff mix {raising:.3f}, fold equity {bluff['fold_equity']:.3f}, "
                   f"blocker delta {bluff['blocker_delta']:.3f}, EV {bluff['ev']:.1f}")
    return Decision(action, StrategySource.HEURISTIC, action_ev=action_ev,
                    action_probs={"fold": fold, "check" if legal.can_check else "call": passive, "raise": raising},
                    reason=reason)


class _WebHeadsUp(RealtimeAgent):
    def _update_ranges(self, view, m, actions, oop, ip, budget):
        if not m.ranges:
            m.ranges.update(ranges(view, self._preflop))
            for seat in (oop, ip):
                if seat != view.seat and self._model(seat) is not None:
                    m.selfplay[seat] = m.selfplay_eps[seat] = m.ranges[seat].copy()
            m.notes.append("preflop ranges: frozen web tables, public action likelihoods")
        return super()._update_ranges(view, m, actions, oop, ip, budget)

    def _network_decision(self, view, legal, st, ranges, me, oop, ip, hero, street, why):
        if self._network() is None or street == "flop":
            from .realtime import heuristic_likelihood, strength_percentile

            d = heuristic(view, legal)
            self._mem.own_likelihood.append(heuristic_likelihood(d.action.type, strength_percentile(tuple(view.board))))
            return Decision(d.action, StrategySource.HEURISTIC, d.action_probs,
                            reason=f"HU HEURISTIC fallback ({why}); {d.reason}")
        return super()._network_decision(view, legal, st, ranges, me, oop, ip, hero, street, why)


class WebRealtimeAgent:
    def __init__(self, seed: int = 0, seconds: float = 10.0, mode: str = "solve",
                 opponent_models: dict | None = None, multiway_bluff: bool = False) -> None:
        if mode not in ("solve", "network"):
            raise ValueError("mode must be 'solve' or 'network'")
        self.seed, self.seconds, self.mode = seed, seconds, mode
        self.opponent_models = opponent_models if opponent_models is not None else {}
        self.multiway_bluff = multiway_bluff
        self._rng = random.Random(seed)
        self._pushfold = PushFoldAgent(seed)
        self._hu = None

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        if view.street is Street.PREFLOP:
            if 5 <= effective_bb(view) <= 20 and not isinstance(self._pushfold._lookup(view), str):
                return self._pushfold.act(view, legal)
            probs, table, key = lookup(view, legal)
            row = probs[class_index(*view.hole_cards)]
            q = self._rng.random()
            if q < row[0]:
                action = Action.fold()
            elif q < row[0] + row[1]:
                action = Action.check() if legal.can_check else Action.call()
            else:
                limpers = sum(s.street_bet == view.big_blind for s in view.seats) - 1
                to = (view.current_bet * table["raise_multiple"] if view.current_bet > view.big_blind else
                      view.big_blind * (table["raise_bb"] + max(limpers, 0)))
                action = Action(legal.raise_type, min(legal.max_raise_to, max(legal.min_raise_to, round(to))))
            return Decision(action, StrategySource.HEURISTIC,
                            action_probs={"fold": float(row[0]), "check" if legal.can_check else "call": float(row[1]), "raise": float(row[2])},
                            reason=f"frozen preflop {tables()['version']} {key}; HEURISTIC, not GTO")
        if len(view.seats) != 2:
            return heuristic(view, legal, self._rng, self.multiway_bluff)
        if self._hu is None:
            # M9's reference describes its old preflop policy; it cannot calibrate this policy.
            self._hu = _WebHeadsUp(self.seed, seconds=self.seconds, mode=self.mode,
                                  opponent_models=self.opponent_models, exploit=False, reference_model_path=None)
        return self._hu.act(view, legal)

    def hand_over(self, view: PlayerView) -> None:
        from backend.opponent.model import OpponentModel
        from backend.opponent.stats import PublicHand

        hand = PublicHand.from_view(view)
        for seat in hand.start_stacks:
            if seat != view.seat:
                self.opponent_models.setdefault(seat, OpponentModel()).observe(hand, seat)
