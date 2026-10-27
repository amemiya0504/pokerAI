"""Web-only policy: frozen preflop, existing HU solver, multiway heuristic."""

from __future__ import annotations

import random
from functools import lru_cache

import numpy as np

from backend.engine.actions import Action, ActionType, LegalActions
from backend.engine.cards import derive_seed
from backend.engine.hand import EventType, PlayerView, Street
from backend.solver.hands import class_index

from .base import Decision, StrategySource
from .pushfold import PushFoldAgent
from .realtime import RealtimeAgent, strength_percentile
from .web_preflop import effective_bb, lookup, ranges, tables

# Style adaptation (A8.1, opt-in): thresholds fixed before its gate; HEURISTIC, not an exploit solver.
# opponent_models[STYLE_KEY][seat] = [iso chances, isos, squeeze chances, squeezes]
STYLE_KEY = "preflop_style"
# A8's station range switch failed its gate; kept behind station_ranges=True only to reproduce A8.
STATION_MIN_HANDS, STATION_VPIP, STATION_PFR = 30, .40, .08
AGGRESSIVE_MIN_CHANCES, AGGRESSIVE_RATE = 8, .50


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


def equity(view: PlayerView, samples: int = 192, uniform_seats: frozenset = frozenset()) -> float:
    """Range-vs-range CPU Monte Carlo with card blockers and split-pot equity."""
    from backend.engine.fast_eval import CARD_A, CARD_B, evaluate_many

    opponents = [s.seat for s in view.seats if not s.folded and s.seat != view.seat]
    if not opponents:
        return 1.0
    weights = _public_ranges(view)
    for seat in uniform_seats:  # their actions say little about their cards (A8)
        weights[seat] = np.ones(1326)
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
              allow_bluff: bool = False, uniform_seats: frozenset = frozenset()) -> Decision:
    n = sum(not s.folded for s in view.seats)
    eq = equity(view, uniform_seats=uniform_seats) if uniform_seats else equity(view)
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
    if uniform_seats:
        reason += f"; uniform ranges for station-like seats {sorted(uniform_seats)}"
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


def preflop_style_chances(view: PlayerView) -> list[tuple[int, str, bool]]:
    """(seat, "iso" | "squeeze", took it) for each first preflop decision facing limpers / an open plus callers."""
    out, raises, limpers, callers, acted = [], 0, 0, 0, set()
    for e in view.history:
        if e.type is not EventType.ACTION or e.street is not Street.PREFLOP:
            continue
        aggressive = e.action in (ActionType.BET, ActionType.RAISE)
        if e.seat not in acted:
            if raises == 0 and limpers:
                out.append((e.seat, "iso", aggressive))
            elif raises == 1 and callers:
                out.append((e.seat, "squeeze", aggressive))
        acted.add(e.seat)
        if aggressive:
            raises, callers = raises + 1, 0
        elif e.action is ActionType.CALL:
            limpers += raises == 0
            callers += raises == 1
    return out


class WebRealtimeAgent:
    def __init__(self, seed: int = 0, seconds: float = 10.0, mode: str = "solve",
                 opponent_models: dict | None = None, multiway_bluff: bool = False, adaptive: bool = False,
                 station_ranges: bool = False) -> None:
        if mode not in ("solve", "network"):
            raise ValueError("mode must be 'solve' or 'network'")
        self.seed, self.seconds, self.mode = seed, seconds, mode
        self.opponent_models = opponent_models if opponent_models is not None else {}
        self.multiway_bluff = multiway_bluff
        self.adaptive = adaptive
        self.station_ranges = station_ranges
        self._rng = random.Random(seed)
        self._pushfold = PushFoldAgent(seed)
        self._hu = None

    def _aggressive(self, seat: int, kind: str) -> bool:
        counts = self.opponent_models.get(STYLE_KEY, {}).get(seat)
        if counts is None:
            return False
        chances, took = counts[0:2] if kind == "iso" else counts[2:4]
        return chances >= AGGRESSIVE_MIN_CHANCES and (took + 1) / (chances + 2) >= AGGRESSIVE_RATE

    def _station_seats(self, view: PlayerView) -> frozenset:
        out = set()
        for s in view.seats:
            model = self.opponent_models.get(s.seat) if s.seat != view.seat and not s.folded else None
            if model is None:
                continue
            vpip_yes, vpip_no = model.total("vpip")
            pfr_yes, pfr_no = model.total("pfr")
            hands = vpip_yes + vpip_no
            if (hands >= STATION_MIN_HANDS and vpip_yes / hands >= STATION_VPIP
                    and pfr_yes / max(1, pfr_yes + pfr_no) <= STATION_PFR):
                out.add(s.seat)
        return frozenset(out)

    def _defend(self, view: PlayerView, legal: LegalActions) -> bool:
        """Use v8.2's capped-range defence only against a raiser seen to iso / squeeze often, and only
        while nobody else has called that raise (A8.3: multiway, weak defenders folded postflop)."""
        from .web_preflop_v8 import DEFEND_GROUPS, context

        group = context(view, legal, defend=True).split(":")[0]
        if group not in DEFEND_GROUPS:
            return False
        actions = [e for e in view.history if e.type is EventType.ACTION and e.street is Street.PREFLOP]
        last = max(i for i, e in enumerate(actions) if e.action in (ActionType.BET, ActionType.RAISE))
        if any(e.action is ActionType.CALL for e in actions[last + 1:]):
            return False
        return self._aggressive(actions[last].seat, "iso" if group == "limp_vs_raise" else "squeeze")

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        if view.street is Street.PREFLOP:
            if 5 <= effective_bb(view) <= 20 and not isinstance(self._pushfold._lookup(view), str):
                return self._pushfold.act(view, legal)
            if self.adaptive and "defend_groups" in tables():
                probs, table, key = lookup(view, legal, defend=self._defend(view, legal))
            else:
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
                me = next(s for s in view.seats if s.seat == view.seat)
                if "jam_fraction" in table and to >= table["jam_fraction"] * (me.stack + me.street_bet):
                    to = legal.max_raise_to  # v8: a raise committing this much of the stack is a jam
                action = Action(legal.raise_type, min(legal.max_raise_to, max(legal.min_raise_to, round(to))))
            return Decision(action, StrategySource.HEURISTIC,
                            action_probs={"fold": float(row[0]), "check" if legal.can_check else "call": float(row[1]), "raise": float(row[2])},
                            reason=f"frozen preflop {tables()['version']} {key}; HEURISTIC, not GTO")
        if len(view.seats) != 2:
            stations = self._station_seats(view) if self.adaptive and self.station_ranges else frozenset()
            if stations:
                return heuristic(view, legal, self._rng, self.multiway_bluff, uniform_seats=stations)
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
        style = self.opponent_models.setdefault(STYLE_KEY, {})
        for seat, kind, took in preflop_style_chances(view):
            if seat != view.seat:
                counts = style.setdefault(seat, [0, 0, 0, 0])
                i = 0 if kind == "iso" else 2
                counts[i] += 1
                counts[i + 1] += took
