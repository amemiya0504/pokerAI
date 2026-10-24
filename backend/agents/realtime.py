"""Heads-up agent that solves the current subgame at decision time (Phase A).

Design: docs/realtime-agent.md. On the turn and river the agent tracks both
players' ranges, re-solves the subgame rooted at the actual state (actual bet
sizes, so no action translation) on the GPU within a time budget, and plays
its combo's strategy (SOLVER_APPROX). If solving is not possible it falls
back to the policy network (NEURAL_APPROX). On the flop it solves a
depth-limited tree (flop and turn betting, river dealt without betting;
SOLVER_APPROX, the reason says so). Preflop uses the push/fold agent (solved when short,
rule-based otherwise); multi-way pots use the rule-based agent (HEURISTIC).
Every decision says which applied and why.

Range tracking (public beliefs, no private information):

* preflop: a heuristic top-X% range by the player's preflop line (labelled);
* postflop, own actions: the exact strategy the agent used;
* postflop, opponent actions: the subgame is re-solved at the opponent's
  decision (with his actual bet size added to the tree) and the range is
  multiplied by each combo's probability of that action. This assumes the
  opponent plays the equilibrium (self-play assumption, replaced by M8's
  opponent model later).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.engine.actions import Action, ActionType, LegalActions
from backend.engine.cards import cards_str
from backend.engine.hand import EventType, PlayerView, Street

from .base import Decision, StrategySource
from .pushfold import PushFoldAgent
from .rule_based import RuleBasedAgent

N = 1326
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "out" / "policy" / "run-v1v2" / "model.pt"
# Statistics of the GTO baseline's own play (M9): exploitation is relative to them.
DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "out" / "m9" / "models" / "gto.json"
RANGE_FLOOR = 0.02  # every combo keeps a little weight: the heuristics can be wrong
FLOP_DEPTH = 2  # streets with betting in the flop subgame

# Bet abstraction for real-time trees (pot fractions), per street of the subgame.
TREES = {
    # Flop: depth-limited to two streets (river dealt without betting). Depth 1
    # (turn and river both dealt) was far off the full solve: see README.
    "flop": ({"flop": [0.33, 0.75], "turn": [0.75]}, {"flop": [1.0], "turn": [1.0]}, 2),
    "river": ({"river": [0.33, 0.75, 1.5]}, {"river": [1.0]}, 2),
    "turn": ({"turn": [0.5, 1.0], "river": [0.5, 1.0]}, {"turn": [1.0], "river": [1.0]}, 1),
}


def _street_name(board) -> str:
    return {3: "flop", 4: "turn", 5: "river"}[len(board)]


# ----------------------------------------------------------------------------- public replay


@dataclass
class StreetState:
    """Public state just before an action, measured from the start of its street."""

    street: Street
    board: tuple[int, ...]
    pot_start: float  # chips in the pot when the street began
    behind_start: dict[int, float]  # each seat's stack when the street began
    bets: dict[int, float]  # chips each seat put in on this street so far
    raises: int = 0
    oop_checked: bool = False


@dataclass
class PublicAction:
    seat: int
    action: ActionType
    added: float  # chips this action put in
    to: float  # the seat's street total after the action
    all_in: bool
    before: StreetState


def replay(view: PlayerView) -> tuple[list[PublicAction], StreetState, int]:
    """All postflop actions with the state before each, the current state,
    and the out-of-position seat (heads-up: the non-button seat)."""
    seats = [s.seat for s in view.seats]
    oop = next(s for s in seats if s != view.button)
    start = {s.seat: s.stack + s.total_bet for s in view.seats}
    total = {s: 0.0 for s in seats}
    bets = {s: 0.0 for s in seats}
    street, board, raises, checked = Street.PREFLOP, (), 0, False
    st = None
    actions: list[PublicAction] = []

    def state() -> StreetState:
        return StreetState(street, board, sum(total.values()) - sum(bets.values()),
                           {s: start[s] - (total[s] - bets[s]) for s in seats}, dict(bets), raises, checked)

    for e in view.history:
        if e.type is EventType.DEAL_BOARD:
            street, board = e.street, board + tuple(e.cards)
            bets = {s: 0.0 for s in seats}
            raises, checked = 0, False
        elif e.type is EventType.POST_BLIND:
            total[e.seat] += e.amount
            bets[e.seat] += e.amount
        elif e.type is EventType.RETURN_UNCALLED:
            total[e.seat] -= e.amount
            bets[e.seat] -= e.amount
        elif e.type is EventType.ACTION:
            before = state()
            if e.action is ActionType.CALL:
                added = e.amount
            elif e.action in (ActionType.BET, ActionType.RAISE):
                added = e.amount - bets[e.seat]
            else:
                added = 0.0
            total[e.seat] += added
            bets[e.seat] += added
            if street is not Street.PREFLOP:
                actions.append(PublicAction(e.seat, e.action, added, bets[e.seat],
                                            start[e.seat] - total[e.seat] <= 1e-9, before))
            if e.action is ActionType.RAISE:
                raises += 1
            if e.action is ActionType.CHECK and e.seat == oop:
                checked = True
    st = state()
    return actions, st, oop


# ----------------------------------------------------------------------------- range heuristics


def _class_order():
    from backend.dataset.spots import CLASS_ORDER

    return CLASS_ORDER


def top_range(width: float) -> np.ndarray:
    from backend.solver.hands import COMBOS
    from backend.solver.subgame import COMBO_INDEX

    w = np.zeros(N)
    taken = 0
    for ci in _class_order():
        if taken >= width * 1326:
            break
        for a, b in COMBOS[ci]:
            w[COMBO_INDEX[(min(a, b), max(a, b))]] = 1.0
        taken += len(COMBOS[ci])
    return w


def preflop_range(view: PlayerView, seat: int) -> tuple[np.ndarray, str]:
    """Heuristic range from a seat's preflop line (HEURISTIC: no solved deep-stack preflop)."""
    acts = [e.action for e in view.history if e.type is EventType.ACTION and e.street is Street.PREFLOP
            and e.seat == seat]
    raises = sum(a in (ActionType.RAISE, ActionType.BET) for a in acts)
    called_raise = False
    level = 0
    for e in view.history:
        if e.type is not EventType.ACTION or e.street is not Street.PREFLOP:
            continue
        if e.action is ActionType.RAISE:
            level += 1
        if e.seat == seat and e.action is ActionType.CALL and level > 0:
            called_raise = True
    if raises >= 2:
        width, line = 0.06, "4-bet+"
    elif raises == 1 and level >= 2:
        width, line = 0.08, "3-bet"
    elif raises == 1:
        width, line = 0.20, "raise"
    elif called_raise:
        width, line = 0.32, "call a raise"
    else:
        width, line = 0.70, "limp/check"
    return np.maximum(top_range(width), RANGE_FLOOR), line


def preflop_context(view: PlayerView, seat: int) -> tuple[str, str] | None:
    """(context group, outcome) of ``seat``'s last preflop action, or None."""
    bb = None
    for e in view.history:
        if e.type is EventType.POST_BLIND and e.amount == view.big_blind:
            bb = e.seat
    raises, last = 0, None
    for e in view.history:
        if e.type is not EventType.ACTION or e.street is not Street.PREFLOP:
            continue
        if e.seat == seat:
            if raises == 0:
                group = "pf_option" if seat == bb else "pf_open"
            else:
                group = {1: "pf_vs_raise", 2: "pf_vs_3bet"}.get(raises, "pf_vs_4bet")
            outcome = {ActionType.CHECK: "check", ActionType.CALL: "call", ActionType.FOLD: "fold"}.get(
                e.action, "raise")
            last = (group, outcome)
        if e.action in (ActionType.RAISE, ActionType.BET):
            raises += 1
    return last


def modelled_preflop_range(view: PlayerView, seat: int, model, strength: float) -> tuple[np.ndarray, str]:
    """Like ``preflop_range``, but the width comes from the opponent model's
    frequencies in the context of the seat's last preflop action, shrunk toward
    the heuristic (``backend.opponent.ranges``). Without observations in that
    context it is exactly ``preflop_range``."""
    from backend.opponent.ranges import CALLER_TOP_WEIGHT, preflop_width

    ctx = preflop_context(view, seat)
    if ctx is None or ctx[1] == "fold":
        return preflop_range(view, seat)
    width, top, n = preflop_width(model, ctx[0], ctx[1], strength)
    if n == 0:
        return preflop_range(view, seat)
    r = top_range(width)
    if top > 0:
        r = r - (1 - CALLER_TOP_WEIGHT) * top_range(top)
    return np.maximum(r, RANGE_FLOOR), (f"{ctx[0]} {ctx[1]}: top {width:.0%}"
                                        + (f" (top {top:.0%} at {CALLER_TOP_WEIGHT})" if top else "")
                                        + f", opponent model ({n} observations)")


def strength_percentile(board: tuple[int, ...]) -> np.ndarray:
    """Made-hand strength percentile of every combo on the current board."""
    from backend.engine.fast_eval import CARD_A, CARD_B, evaluate_many
    from backend.solver.subgame import board_mask

    valid = board_mask(board) > 0
    idx = np.nonzero(valid)[0]
    hands = np.concatenate([CARD_A[idx, None], CARD_B[idx, None],
                            np.tile(np.array(board, dtype=np.int64), (len(idx), 1))], axis=1)
    s = evaluate_many(hands)
    order = np.argsort(np.argsort(s, kind="stable"), kind="stable")
    out = np.zeros(N)
    out[idx] = order / max(len(idx) - 1, 1)
    return out


def heuristic_likelihood(action: ActionType, strength: np.ndarray) -> np.ndarray:
    """P(action | combo) up to a constant, from made-hand strength (flop only)."""
    s = strength
    if action in (ActionType.BET, ActionType.RAISE):
        return 0.1 + 0.9 * s**3
    if action is ActionType.CALL:
        return 0.15 + 0.85 * s
    if action is ActionType.CHECK:
        return 1.0 - 0.6 * s**4
    return 1.0 - s  # fold


# ----------------------------------------------------------------------------- the agent


@dataclass
class _HandMemory:
    hole: tuple[int, ...]
    ranges: dict[int, np.ndarray] = field(default_factory=dict)
    processed: int = 0  # postflop actions already applied to the ranges
    cards_seen: int = 0
    own_likelihood: list[np.ndarray] = field(default_factory=list)  # one per own postflop action
    notes: list[str] = field(default_factory=list)
    # The latest solve and the index (in ``replay``'s actions) of its root
    # action: safe re-solving takes the opponent's values at the next node from it.
    last: tuple | None = None
    # With an opponent model: the opponent's range as the self-play tracker
    # would have it (same re-solves, no tilt, heuristic preflop), for diagnostics.
    selfplay: dict[int, np.ndarray] = field(default_factory=dict)
    # Ablation: self-play plus the model's mistake share (EPSILON), no model.
    selfplay_eps: dict[int, np.ndarray] = field(default_factory=dict)

    def keep(self, cfr, root_index: int) -> None:
        self.drop()
        cfr.freeze()
        self.last = (cfr, root_index)

    def drop(self) -> None:
        if self.last is not None:
            self.last[0].release()
            self.last = None


class RealtimeAgent:
    """``mode``: "solve" (real-time solving, network fallback) or "network"
    (network only, with the same range tracking; for comparison).

    ``resolving``: "unsafe" re-solves with the tracked ranges; "safe" adds the
    re-solve gadget (``backend.solver.subgame_gpu.Gadget``) with the opponent's
    values at the current node from the previous solve, so a misread opponent
    range cannot hand any of his combos more than that solve gave him (falls
    back to unsafe when no previous solve covers the node, and says so)."""

    def __init__(self, seed: int = 0, seconds: float = 10.0, target_pct_pot: float = 0.5,
                 mode: str = "solve", device: str | None = None, model_path: str | Path = DEFAULT_MODEL,
                 min_iterations: int = 20, resolving: str = "safe", opponent_models: dict | None = None,
                 use_opponent_model: bool = True, model_strength: float = 20.0, exploit: bool = True,
                 exploit_max: float = 0.8, exploit_strength: float = 50.0,
                 opponent_model=None, opponent_model_path: str | Path | None = None,
                 reference_model=None, reference_model_path: str | Path | None = DEFAULT_REFERENCE) -> None:
        if mode not in ("solve", "network"):
            raise ValueError("mode must be 'solve' or 'network'")
        if resolving not in ("unsafe", "safe"):
            raise ValueError("resolving must be 'unsafe' or 'safe'")
        self.resolving = resolving
        # Opponent models by seat (``backend.opponent.model.OpponentModel``),
        # updated by ``hand_over``; pass a shared dict to keep them across
        # agent instances (the web session builds a new agent every hand).
        self.opponent_models = opponent_models if opponent_models is not None else {}
        self.use_opponent_model, self.model_strength = use_opponent_model, model_strength
        # M9: exploit the modelled tendencies (data-biased response); the
        # confidence of a context is exploit_max * n / (n + exploit_strength).
        self.exploit, self.exploit_max, self.exploit_strength = exploit, exploit_max, exploit_strength
        if opponent_model is None and opponent_model_path is not None:
            import json

            from backend.opponent.model import OpponentModel

            opponent_model = OpponentModel.from_json(json.loads(Path(opponent_model_path).read_text()))
        # Model used for any opponent seat without its own (e.g. a known opponent's history).
        self.default_opponent_model = opponent_model
        if reference_model is None and reference_model_path is not None and Path(reference_model_path).exists():
            import json

            from backend.opponent.model import OpponentModel

            reference_model = OpponentModel.from_json(json.loads(Path(reference_model_path).read_text()))
        # Without a reference there is no exploitation: absolute frequencies mislead (README, M9).
        self.reference_model = reference_model
        self.seconds, self.target, self.mode = seconds, target_pct_pot, mode
        self.device, self.model_path, self.min_iterations = device, Path(model_path), min_iterations
        self._rng = random.Random(seed)
        self._preflop = PushFoldAgent(seed)
        self._rule = RuleBasedAgent(seed)
        self._runner = None
        self._mem: _HandMemory | None = None
        if mode == "solve":
            # Import PyTorch and initialise the device now: done lazily inside
            # the first decision it would cost seconds of that decision's budget.
            import torch

            from backend.solver import subgame_gpu  # noqa: F401

            dev = device or subgame_gpu.default_device()
            torch.zeros(1, device=dev)

    # ------------------------------------------------------------------ helpers

    def hand_over(self, view: PlayerView) -> None:
        """A hand this agent sat in is over: add its public information to the
        opponent models (no hole cards except those shown at showdown)."""
        from backend.opponent.model import OpponentModel
        from backend.opponent.stats import PublicHand

        hand = PublicHand.from_view(view)
        for seat in hand.start_stacks:
            if seat != view.seat:
                if seat not in self.opponent_models:
                    base = self.default_opponent_model
                    self.opponent_models[seat] = (OpponentModel.from_json(base.to_json()) if base is not None
                                                  else OpponentModel())
                self.opponent_models[seat].observe(hand, seat)

    def _model(self, seat: int):
        if not self.use_opponent_model:
            return None
        return self.opponent_models.get(seat, self.default_opponent_model)

    def _range_model(self, seat: int, group: str | None = None, threshold: float = 1 / 3):
        """M9 applies range changes only where its public evidence supports them."""
        model = self._model(seat)
        if not self.exploit or self.reference_model is None:
            return model
        from backend.opponent.exploit import overfold_evidence, wide_range_evidence

        if wide_range_evidence(model, self.reference_model) or (
            group is not None and overfold_evidence(model, self.reference_model, group, threshold)
        ):
            return model
        return None

    def _exploit_spec(self, seat: int, solver_player: int):
        """``Exploit`` for the opponent in ``seat`` from his model, or None."""
        from backend.opponent.exploit import exploit_from_model

        if not self.exploit or self.reference_model is None:
            return None
        return exploit_from_model(self._model(seat), solver_player, self.exploit_max, self.exploit_strength,
                                  reference=self.reference_model)

    def _model_note(self, seat: int) -> str:
        model = self._model(seat)
        if model is None:
            return "opponent ranges: self-play assumption"
        return f"opponent ranges: opponent model ({model.hands} hands)"

    def range_report(self) -> dict[str, dict[int, np.ndarray]]:
        """The current hand's tracked ranges: "model" (used for decisions),
        "selfplay" (the same re-solves without the opponent model) and
        "selfplay_eps" (that plus the model's 2% mistake share: an ablation)."""
        m = self._mem
        if m is None:
            return {"model": {}, "selfplay": {}, "selfplay_eps": {}}
        return {"model": dict(m.ranges), "selfplay": {**m.ranges, **m.selfplay},
                "selfplay_eps": {**m.ranges, **m.selfplay_eps}}

    def _network(self):
        """The policy network, or None when no trained model file exists."""
        if self._runner is None and self.model_path.exists():
            from backend.policy.infer import PolicyRunner

            self._runner = PolicyRunner(self.model_path, device="cpu")
        return self._runner

    def _memory(self, view: PlayerView) -> _HandMemory:
        m = self._mem
        if m is None or m.hole != view.hole_cards or len(view.history) < m.processed:
            if m is not None:
                m.drop()
            m = self._mem = _HandMemory(tuple(view.hole_cards))
        return m

    @staticmethod
    def _cfg(st: StreetState, oop: int, ip: int):
        from backend.solver.subgame import TreeConfig

        bets, raises, max_raises = TREES[_street_name(st.board)]
        stack = min(st.behind_start[oop], st.behind_start[ip])
        return TreeConfig(st.pot_start, stack, bets, raises, max_raises, True)

    @staticmethod
    def _root(st: StreetState, actor: int, oop: int, ip: int, extra: tuple[float, ...] = (),
              allow_raise: bool = False):
        """``allow_raise``: the observed action was a bet/raise, so the root must
        offer one even if the abstraction's raise cap is already reached."""
        from backend.solver.subgame_gpu import RootState

        raises = st.raises
        if allow_raise:
            max_raises = TREES[_street_name(st.board)][2]
            raises = min(raises, max_raises - 1)
        return RootState((st.bets[oop], st.bets[ip]), 0 if actor == oop else 1, raises,
                         actor == ip and st.oop_checked and st.bets[oop] == st.bets[ip] == 0, extra)

    def _solve(self, st: StreetState, ranges, actor: int, oop: int, ip: int, deadline: float,
               extra: tuple[float, ...] = (), allow_raise: bool = False, need_exploitability: bool = True,
               gadget=None, exploit=None):
        """Solve from ``st`` with ``actor`` to act until the exploitability
        target or the absolute ``deadline`` (tree building included);
        returns (cfr, info) or raises. Without ``need_exploitability`` (range
        updates) the first check is skipped when it would overrun the deadline,
        and info's exploitability is then None."""
        import torch

        from backend.solver.subgame_gpu import BatchedSubgameCFR

        t0 = time.perf_counter()
        cfr = BatchedSubgameCFR(st.board, (ranges[oop], ranges[ip]), self._cfg(st, oop, ip),
                                device=self.device, dtype=torch.float32,
                                root=self._root(st, actor, oop, ip, extra, allow_raise),
                                checkdown=FLOP_DEPTH if len(st.board) == 3 else False, gadget=gadget,
                                exploit=exploit)
        build = time.perf_counter() - t0
        done, evals, ex = 0, 0, None
        timing = getattr(self, "_timing", None)

        def run(n: int) -> float:
            nonlocal done
            c0 = time.perf_counter()
            cfr.iterate(n)
            done += n
            dt = time.perf_counter() - c0
            if timing is not None:
                timing.append((n, round(dt, 2), cfr.template_nodes, round(cfr.memory_gb, 2)))
            return dt

        def check() -> float:
            nonlocal ex, evals
            c0 = time.perf_counter()
            ex = cfr.exploitability()["exploitability_pct_pot"]  # exact; costs ~3 tree passes
            evals += 1
            return time.perf_counter() - c0

        warm = run(10)  # warm-up and CUDA graph capture: slow, not used for timing
        # Decisions need min_iterations anyway; range updates shrink to the deadline.
        n = 25 if need_exploitability else max(5, min(25, int((deadline - time.perf_counter()) / (warm / 10))))
        per_it = run(n) / n
        if not need_exploitability and deadline - time.perf_counter() < 20 * per_it:  # a check ~ 20 iterations
            return cfr, {"iterations": done, "exploitability_pct_pot": None, "build_seconds": build,
                         "checks": 0, "seconds": time.perf_counter() - t0}
        eval_cost = check()
        # A check costs ~3 tree passes, so check only where the target may have
        # been reached: DCFR's exploitability falls roughly like t^-1.5 here
        # (measured on flop and turn trees), which predicts the iterations
        # still needed. Each chunk is at least half the iterations done so far,
        # and the last one is whatever still fits before the deadline.
        while ex > self.target:
            left = deadline - time.perf_counter() - eval_cost
            fits = int(left / per_it)
            if fits < 5:
                break
            need = int(done * (ex / self.target) ** (1 / 1.5)) - done
            n = min(max(need, done // 2, 5), fits)
            per_it = run(n) / n
            eval_cost = check()
        return cfr, {"iterations": done, "exploitability_pct_pot": ex, "build_seconds": build,
                     "checks": evals, "seconds": time.perf_counter() - t0}

    @staticmethod
    def _child_add(node, i: int) -> float:
        return node.children[i].inv[node.player] - node.inv[node.player]

    @staticmethod
    def _root_child_add(cfr, i: int) -> float:
        return RealtimeAgent._child_add(cfr.root, i)

    def _match_at(self, node, act: PublicAction) -> int | None:
        """Index of ``node``'s action matching an observed public action (the
        nearest size for bets and raises), or None if it has no such action."""
        labels = node.actions
        want = {ActionType.FOLD: "f", ActionType.CHECK: "x", ActionType.CALL: "c"}.get(act.action)
        if want is not None:
            return labels.index(want) if want in labels else None
        aggressive = [i for i, a in enumerate(labels) if a[0] in "bra"]
        if not aggressive:
            return None
        return min(aggressive, key=lambda i: abs(self._child_add(node, i) - act.added))

    def _match(self, cfr, act: PublicAction) -> int:
        """Index of the root action matching an observed public action."""
        return self._match_at(cfr.root, act)

    def _safe_targets(self, m: _HandMemory, actions, me: int, oop: int, board) -> tuple[np.ndarray | None, str]:
        """The opponent's counterfactual values at the current node from the
        latest solve, or (None, why not)."""
        if m.last is None:
            return None, "no previous solve this hand"
        cfr, start = m.last
        node = cfr.root
        for act in actions[start:]:
            if node.kind == "chance":
                node = node.children[0]
            if node.kind != "decision":
                return None, "previous solve's tree ends before this node"
            i = self._match_at(node, act)
            if i is None:
                return None, "an action since the previous solve is not in its tree"
            pot_now = cfr.cfg.pot + node.inv[0] + node.inv[1]
            if act.action in (ActionType.BET, ActionType.RAISE) and \
                    abs(self._child_add(node, i) - act.added) > max(1.0, 0.02 * pot_now):
                return None, "a bet size since the previous solve is not in its tree"
            node = node.children[i]
        if node.kind == "chance":
            node = node.children[0]
        if node.kind != "decision" or node.player != (0 if me == oop else 1):
            return None, "previous solve's tree has no decision of ours here"
        rows = [tuple(int(c) for c in b) for b in node.boards]
        if tuple(board) not in rows:
            return None, "previous solve's tree does not have this board"
        opp = 1 - node.player
        try:
            v, _, _ = cfr.counterfactual_values(opp, node.path, tuple(board))
        except KeyError as e:
            return None, f"node not found in the previous solve ({e})"
        return v, "targets from the previous solve"

    @staticmethod
    def _fraction(act: PublicAction, oop: int, ip: int) -> float:
        st = act.before
        other = ip if act.seat == oop else oop
        to_call = st.bets[other] - st.bets[act.seat]
        pot_now = st.pot_start + st.bets[oop] + st.bets[ip]
        if to_call <= 0:
            return act.added / pot_now
        return (act.added - to_call) / (pot_now + to_call)

    # ------------------------------------------------------------------ range tracking

    def _update_ranges(self, view: PlayerView, m: _HandMemory, actions, oop: int, ip: int,
                       budget: float) -> float:
        """Apply new public information within ``budget`` seconds in total;
        returns the time spent."""
        from backend.solver.subgame import board_mask

        t0 = time.perf_counter()
        if not m.ranges:
            for s in (oop, ip):
                m.ranges[s], line = preflop_range(view, s)
                if s != view.seat and self._model(s) is not None:
                    m.selfplay[s] = m.selfplay_eps[s] = m.ranges[s]
                    model = self._range_model(s)
                    if model is not None:
                        m.ranges[s], line = modelled_preflop_range(view, s, model, self.model_strength)
                m.notes.append(f"preflop range seat {s}: {line} (heuristic)")
        if len(view.board) != m.cards_seen:
            mask = board_mask(view.board)
            for s in m.ranges:
                m.ranges[s] = m.ranges[s] * mask
            for s in m.selfplay:
                m.selfplay[s] = m.selfplay[s] * mask
                m.selfplay_eps[s] = m.selfplay_eps[s] * mask
            m.cards_seen = len(view.board)
        own = [a for a in actions if a.seat == view.seat]
        for k in range(m.processed, len(actions)):
            act = actions[k]
            seat = act.seat
            if seat == view.seat:
                lik = m.own_likelihood[own.index(act)] if own.index(act) < len(m.own_likelihood) else None
                if lik is not None:
                    m.ranges[seat] = m.ranges[seat] * lik
                continue
            st = act.before
            lik = None
            if self.mode == "solve":
                try:
                    extra = ()
                    if act.action in (ActionType.BET, ActionType.RAISE) and not act.all_in:
                        extra = (round(self._fraction(act, oop, ip), 4),)
                    pending = sum(1 for a in actions[k:] if a.seat != view.seat)
                    deadline = t0 + budget * (1 - (pending - 1) / max(pending, 1)) \
                        if pending > 1 else t0 + budget
                    aggressive = act.action in (ActionType.BET, ActionType.RAISE)
                    cfr, _ = self._solve(st, m.ranges, seat, oop, ip, deadline, extra, allow_raise=aggressive,
                                         need_exploitability=False)
                    sigma = cfr._avg(cfr.root)[:, 0, :].double().cpu().numpy()
                    i = self._match(cfr, act)
                    lik = sigma[i]
                    from backend.opponent.ranges import EPSILON, modelled_strategy, postflop_group

                    other = ip if seat == oop else oop
                    facing = st.bets[other] > st.bets[seat]
                    group = postflop_group(_street_name(st.board), facing, st.raises)
                    threshold = max(1 / 3, (st.bets[other] - st.bets[seat]) /
                                    max(st.pot_start + sum(st.bets.values()), 1e-30))
                    model = self._range_model(seat, group, threshold)
                    if seat in m.selfplay:
                        eps_lik = (1 - EPSILON) * lik + EPSILON / sigma.shape[0] if model is not None else lik
                        m.selfplay[seat] = np.maximum(m.selfplay[seat] * lik,
                                                      1e-9 * (m.selfplay[seat] > 0))
                        m.selfplay_eps[seat] = np.maximum(m.selfplay_eps[seat] * eps_lik,
                                                          1e-9 * (m.selfplay_eps[seat] > 0))
                    if model is not None:
                        from backend.opponent.exploit import wide_range_evidence

                        fold_only = (self.exploit and self.reference_model is not None
                                     and not wide_range_evidence(model, self.reference_model))
                        probs, note = modelled_strategy(sigma, m.ranges[seat], list(cfr.root.actions), facing,
                                                        group, model, self.model_strength, fold_only=fold_only)
                        lik = probs[i]
                        m.notes.append(f"opponent model {note}")
                    m.keep(cfr, k)
                except Exception as e:  # noqa: BLE001 - fall back, but say so
                    m.notes.append(f"opponent update fell back to the network: {e}")
            if lik is None:
                lik = self._network_likelihood(view, st, m, seat, act, oop, ip)
                if seat in m.selfplay:
                    m.selfplay[seat] = np.maximum(m.selfplay[seat] * lik, 1e-9 * (m.selfplay[seat] > 0))
                    m.selfplay_eps[seat] = np.maximum(m.selfplay_eps[seat] * lik,
                                                      1e-9 * (m.selfplay_eps[seat] > 0))
            m.ranges[seat] = np.maximum(m.ranges[seat] * lik, 1e-9 * (m.ranges[seat] > 0))
        m.processed = len(actions)
        return time.perf_counter() - t0

    def _network_likelihood(self, view, st, m, seat, act, oop, ip) -> np.ndarray:
        from backend.solver.subgame_gpu import BatchedBuilder

        if self._network() is None or len(st.board) == 3:  # the network has no flop training
            m.notes.append("heuristic range update")
            return heuristic_likelihood(act.action, strength_percentile(st.board))

        aggressive = act.action in (ActionType.BET, ActionType.RAISE)
        opts = BatchedBuilder(self._cfg(st, oop, ip), "cpu", None).root_options(
            st.board, self._root(st, seat, oop, ip, allow_raise=aggressive))
        labels = [a for a, _ in opts]
        if aggressive and not any(a[0] in "bra" for a in labels):
            m.notes.append("observed bet not representable: heuristic range update")
            return heuristic_likelihood(act.action, strength_percentile(st.board))
        combos = np.nonzero(m.ranges[seat] > 0)[0]
        probs = self._network().predict(
            board=cards_str(st.board), street="river" if len(st.board) == 5 else "turn",
            pot=st.pot_start, stack=min(st.behind_start[oop], st.behind_start[ip]),
            inv=(st.bets[oop], st.bets[ip]), player=0 if seat == oop else 1, actions=labels,
            reach_oop=m.ranges[oop], reach_ip=m.ranges[ip], combos=combos)
        if act.action in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            i = labels.index({ActionType.FOLD: "f", ActionType.CHECK: "x", ActionType.CALL: "c"}[act.action])
        else:
            adds = {i: add for i, (a, add) in enumerate(opts) if a[0] in "bra"}
            i = min(adds, key=lambda j: abs(adds[j] - act.added))
        out = np.zeros(N)
        out[combos] = probs[:, i]
        return out

    # ------------------------------------------------------------------ deciding

    def _to_engine(self, label: str, add: float, view: PlayerView, legal: LegalActions) -> Action:
        if label == "f":
            return Action.check() if legal.can_check else Action.fold()
        if label == "x":
            return Action.check() if legal.can_check else Action.call()
        if label == "c":
            return Action.call() if legal.can_call else Action.check()
        if legal.raise_type is None:
            return Action.call() if legal.can_call else Action.check()
        me = next(s for s in view.seats if s.seat == view.seat)
        to = me.street_bet + add
        to = legal.max_raise_to if label == "a" else min(max(round(to), legal.min_raise_to), legal.max_raise_to)
        return Action(legal.raise_type, int(to))

    def act(self, view: PlayerView, legal: LegalActions) -> Decision:
        if view.street is Street.PREFLOP:
            return self._preflop.act(view, legal)
        if len(view.seats) != 2 or sum(1 for s in view.seats if not s.folded) != 2:
            d = self._rule.act(view, legal)
            return Decision(d.action, StrategySource.HEURISTIC, reason=f"not heads-up; {d.reason}")
        t0 = time.perf_counter()
        m = self._memory(view)
        actions, st, oop = replay(view)
        ip = next(s.seat for s in view.seats if s.seat != oop)
        spent = self._update_ranges(view, m, actions, oop, ip, budget=0.3 * self.seconds)
        me = view.seat

        from backend.solver.subgame import COMBO_INDEX

        a, b = sorted(view.hole_cards)
        hero = COMBO_INDEX[(a, b)]
        ranges = dict(m.ranges)
        ranges[me] = np.maximum(ranges[me], 0)
        if ranges[me][hero] <= 0:
            ranges[me][hero] = RANGE_FLOOR  # our own combo must be in our range
        street = _street_name(view.board)
        fallback_reason = "mode=network"
        if self.mode == "solve":
            try:
                gadget, resolving = None, "unsafe re-solving"
                if self.resolving == "safe":
                    from backend.solver.subgame_gpu import Gadget

                    cfv, why = self._safe_targets(m, actions, me, oop, view.board)
                    if cfv is None:
                        resolving = f"unsafe re-solving ({why})"
                    else:
                        gadget = Gadget(1 if me == oop else 0, cfv)
                        resolving = f"safe re-solving ({why})"
                opp_seat = ip if me == oop else oop
                xspec = self._exploit_spec(opp_seat, 0 if opp_seat == oop else 1)
                cfr, info = self._solve(st, ranges, me, oop, ip, deadline=t0 + self.seconds, gadget=gadget,
                                        exploit=xspec)
                if info["iterations"] < self.min_iterations:
                    cfr.release()
                    raise TimeoutError(f"only {info['iterations']} iterations in the time budget")
                avg = cfr._avg(cfr.root)[:, 0, :].double().cpu().numpy()
                labels = list(cfr.root.actions)
                adds = [self._root_child_add(cfr, i) for i in range(len(labels))]
                if xspec is not None and cfr.exploited_nodes:
                    gap = "unrestricted exploitability"  # includes the deliberately biased opponent policy
                    xnote = (f"exploiting the opponent model at {cfr.exploited_nodes} nodes "
                             f"(confidence up to {max(xspec.confidence.values()):.2f})")
                else:
                    gap = "exploitability"
                    if self.exploit and self.reference_model is None:
                        xnote = "no exploitation (no reference model)"
                    elif self.exploit and self._model(opp_seat) is not None:
                        xnote = "no exploitation (insufficient evidence at this node)"
                    else:
                        xnote = "no exploitation (GTO baseline)"
                m.keep(cfr, len(actions))
                k = self._sample(avg[:, hero])
                m.own_likelihood.append(avg[k])
                kind = "flop, depth-limited: river dealt without betting" if street == "flop" else street
                return Decision(self._to_engine(labels[k], adds[k], view, legal), StrategySource.SOLVER_APPROX,
                                action_probs={l: round(float(p), 4) for l, p in zip(labels, avg[:, hero])},
                                reason=(f"real-time solve ({kind}): {info['iterations']} iterations, "
                                        f"{gap} {info['exploitability_pct_pot']:.2f}% pot, "
                                        f"{info['seconds']:.1f}s (tree {info['build_seconds']:.1f}s); "
                                        f"{resolving}; {self._model_note(ip if me == oop else oop)}; "
                                        f"{xnote}; range updates {spent:.1f}s"))
            except (MemoryError, TimeoutError, RuntimeError) as e:
                fallback_reason = f"solver unavailable: {e}"
        return self._network_decision(view, legal, st, ranges, me, oop, ip, hero, street, fallback_reason)

    def _network_decision(self, view, legal, st, ranges, me, oop, ip, hero, street, why) -> Decision:
        from backend.solver.subgame_gpu import BatchedBuilder

        if self._network() is None or street == "flop":  # no network for the flop
            d = self._rule.act(view, legal)
            kind = {ActionType.FOLD: ActionType.FOLD, ActionType.CHECK: ActionType.CHECK,
                    ActionType.CALL: ActionType.CALL}.get(d.action.type, ActionType.BET)
            self._mem.own_likelihood.append(heuristic_likelihood(kind, strength_percentile(tuple(view.board))))
            missing = "no flop policy network" if street == "flop" else f"no policy network model at {self.model_path}"
            return Decision(d.action, StrategySource.HEURISTIC, reason=f"{why}; {missing}; {d.reason}")

        opts = BatchedBuilder(self._cfg(st, oop, ip), "cpu", None).root_options(st.board, self._root(st, me, oop, ip))
        labels = [a for a, _ in opts]
        combos = np.nonzero(ranges[me] > 0)[0]
        probs = self._network().predict(
            board=cards_str(st.board), street=street, pot=st.pot_start,
            stack=min(st.behind_start[oop], st.behind_start[ip]), inv=(st.bets[oop], st.bets[ip]),
            player=0 if me == oop else 1, actions=labels, reach_oop=ranges[oop], reach_ip=ranges[ip],
            combos=combos)
        full = np.zeros((len(labels), N))
        full[:, combos] = probs.T
        k = self._sample(full[:, hero])
        self._mem.own_likelihood.append(full[k])
        return Decision(self._to_engine(labels[k], opts[k][1], view, legal), StrategySource.NEURAL_APPROX,
                        action_probs={l: round(float(p), 4) for l, p in zip(labels, full[:, hero])},
                        reason=f"policy network {self.model_path.parent.name} ({street}); {why}")

    def _sample(self, probs: np.ndarray) -> int:
        p = np.maximum(probs, 0)
        p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
        return int(self._rng.choices(range(len(p)), weights=p)[0])
