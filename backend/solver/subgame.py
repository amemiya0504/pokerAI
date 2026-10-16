"""Heads-up postflop subgame solver (flop, turn or river) with fixed bet sizes.

Vector-form CFR+: every public node holds one strategy row per action over
all 1,326 hole-card combos, and a traversal passes whole reach vectors.
Card removal is exact: terminal values only count opponent combos that share
no card with the hero's combo or the board, and chance nodes deal every
remaining card. Exploitability comes from an exact vector best response.

Values are chips relative to the subgame start with the starting pot split
evenly, so the game is zero-sum: winning a showdown is worth pot/2 plus the
opponent's investment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from backend.agents.base import StrategySource
from backend.engine.cards import card_str, parse_cards
from backend.engine.evaluator import evaluate

from .base import SolveRequest, SolveResult
from .hands import CLASS_NAMES, COMBOS

N = 1326
COMBO_LIST = list(combinations(range(52), 2))
COMBO_INDEX = {c: i for i, c in enumerate(COMBO_LIST)}
CARD_A = np.array([a for a, _ in COMBO_LIST])
CARD_B = np.array([b for _, b in COMBO_LIST])
# CARD_COMBOS[c] = the 51 combos holding card c
CARD_COMBOS = np.array([[i for i, (a, b) in enumerate(COMBO_LIST) if c in (a, b)] for c in range(52)])


def combo_str(i: int) -> str:
    a, b = COMBO_LIST[i]
    return card_str(b) + card_str(a)


def board_mask(board) -> np.ndarray:
    m = np.ones(N)
    for c in board:
        m[CARD_COMBOS[c]] = 0.0
    return m


# ---------------------------------------------------------------- ranges


def parse_range(text: str, board=()) -> np.ndarray:
    """Weights over the 1,326 combos. Tokens: AA, AKs, AKo, AK, 77+, ATs+,
    AsKs, with optional :weight; "all" or "*" for every combo."""
    w = np.zeros(N)
    text = text.strip()
    if text in ("all", "*", "random"):
        w[:] = 1.0
    else:
        names = {n: i for i, n in enumerate(CLASS_NAMES)}
        ranks = "AKQJT98765432"
        for tok in filter(None, (t.strip() for t in text.split(","))):
            tok, _, weight = tok.partition(":")
            weight = float(weight) if weight else 1.0
            classes: list[int] = []
            if len(tok) == 4 and tok[1] in "cdhs":
                a, b = parse_cards(tok)
                w[COMBO_INDEX[(min(a, b), max(a, b))]] = weight
                continue
            plus = tok.endswith("+")
            base = tok.rstrip("+")
            if len(base) == 2 and base[0] != base[1]:
                kinds = [base + "s", base + "o"]
            else:
                kinds = [base]
            for k in kinds:
                if k not in names:
                    raise ValueError(f"bad range token {tok!r}")
                if not plus:
                    classes.append(names[k])
                elif len(k) == 2:  # pairs: 77+ = 77..AA
                    for r in ranks[: ranks.index(k[0]) + 1]:
                        classes.append(names[r + r])
                else:  # ATs+ = ATs..AKs
                    hi = k[0]
                    for r in ranks[ranks.index(hi) + 1: ranks.index(k[1]) + 1]:
                        classes.append(names[hi + r + k[2]])
            for ci in classes:
                for a, b in COMBOS[ci]:
                    w[COMBO_INDEX[(min(a, b), max(a, b))]] = weight
    return w * board_mask(board)


# ---------------------------------------------------------------- showdown


class Showdown:
    """Win-minus-lose opponent mass for every combo on a complete board."""

    def __init__(self, board) -> None:
        valid = board_mask(board) > 0
        idx = np.nonzero(valid)[0]
        strength = np.array([evaluate((CARD_A[i], CARD_B[i], *board)) for i in idx])
        order = np.argsort(strength, kind="stable")
        self.order = idx[order]
        s_sorted = strength[order]
        # Group bounds in the global order: [lo, hi) holds equal strength.
        lo_sorted = np.searchsorted(s_sorted, s_sorted, side="left")
        hi_sorted = np.searchsorted(s_sorted, s_sorted, side="right")
        self.valid = valid
        self.lo = np.zeros(N, dtype=np.int64)
        self.hi = np.zeros(N, dtype=np.int64)
        self.lo[self.order] = lo_sorted
        self.hi[self.order] = hi_sorted
        # Per card: valid combos holding it, sorted by strength (padded with N).
        rank_of = np.full(N, -1)
        rank_of[self.order] = np.arange(len(self.order))
        per_card = []
        for c in range(52):
            cs = [i for i in CARD_COMBOS[c] if valid[i]]
            cs.sort(key=lambda i: rank_of[i])
            per_card.append(cs)
        L = max(len(x) for x in per_card)
        self.card_lists = np.full((52, L), N, dtype=np.int64)
        strength_full = np.full(N + 1, np.iinfo(np.int64).max)
        strength_full[idx] = strength
        for c, cs in enumerate(per_card):
            self.card_lists[c, : len(cs)] = cs
        lists_strength = strength_full[self.card_lists]
        self._rp = np.zeros(N + 1)  # reach padded with a zero slot at index N
        self._cum = np.zeros(len(self.order) + 1)
        self._cumc = np.zeros((52, L + 1))
        self.lo_a = np.zeros(N, dtype=np.int64)
        self.hi_a = np.zeros(N, dtype=np.int64)
        self.lo_b = np.zeros(N, dtype=np.int64)
        self.hi_b = np.zeros(N, dtype=np.int64)
        for i in idx:
            s = strength_full[i]
            for card, lo, hi in ((CARD_A[i], self.lo_a, self.hi_a), (CARD_B[i], self.lo_b, self.hi_b)):
                row = lists_strength[card]
                lo[i] = np.searchsorted(row, s, side="left")
                hi[i] = np.searchsorted(row, s, side="right")

    def masses(self, r_opp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(weaker, stronger): compatible opponent reach that loses to / beats each combo."""
        rp = self._rp
        rp[:N] = r_opp
        cum = self._cum
        np.cumsum(rp[self.order], out=cum[1:])
        cumc = self._cumc
        np.cumsum(rp[self.card_lists], axis=1, out=cumc[:, 1:])
        total, ctot = cum[-1], cumc[:, -1]
        weaker = cum[self.lo] - cumc[CARD_A, self.lo_a] - cumc[CARD_B, self.lo_b]
        stronger = ((total - cum[self.hi]) - (ctot[CARD_A] - cumc[CARD_A, self.hi_a])
                    - (ctot[CARD_B] - cumc[CARD_B, self.hi_b]))
        return np.where(self.valid, weaker, 0.0), np.where(self.valid, stronger, 0.0)

    def net_mass(self, r_opp: np.ndarray) -> np.ndarray:
        weaker, stronger = self.masses(r_opp)
        return weaker - stronger

    def equity(self, r_opp: np.ndarray) -> np.ndarray:
        """Showdown equity of each combo against the opponent's range (ties count half);
        0 where no opponent combo is compatible."""
        weaker, stronger = self.masses(r_opp)
        compat = compatible_mass(r_opp)
        ties = np.maximum(compat - weaker - stronger, 0.0)
        return np.where((compat > 0) & self.valid, (weaker + 0.5 * ties) / np.maximum(compat, 1e-30), 0.0)


def compatible_mass(r_opp: np.ndarray) -> np.ndarray:
    """Opponent reach over combos sharing no card with each hero combo."""
    cs = r_opp[CARD_COMBOS].sum(axis=1)
    return r_opp.sum() - cs[CARD_A] - cs[CARD_B] + r_opp


# ---------------------------------------------------------------- tree


@dataclass(frozen=True)
class TreeConfig:
    pot: float
    stack: float  # effective stack behind at the start of the subgame
    bet_sizes: dict[str, list[float]]  # street -> pot fractions for bets
    raise_sizes: dict[str, list[float]]  # street -> pot fractions for raises
    max_raises: int = 2
    allin: bool = True
    allin_threshold: float = 0.67  # a bet using >= this share of the stack becomes all-in


STREETS = {3: "flop", 4: "turn", 5: "river"}


@dataclass
class PNode:
    kind: str  # decision, chance, fold, showdown
    board: tuple[int, ...]
    inv: tuple[float, float]
    player: int = -1
    actions: list[str] = field(default_factory=list)
    children: list[PNode] = field(default_factory=list)
    cards: list[int] = field(default_factory=list)  # chance: card dealt for each child
    folder: int = -1
    path: str = ""
    regret: np.ndarray | None = None
    strat_sum: np.ndarray | None = None


class TreeBuilder:
    def __init__(self, cfg: TreeConfig) -> None:
        self.cfg = cfg
        self.decision_nodes = 0

    def build(self, board: tuple[int, ...]) -> PNode:
        return self._street(board, (0.0, 0.0), "")

    def _street(self, board, inv, path) -> PNode:
        return self._decide(board, inv, 0, (0.0, 0.0), 0, path + ("/" if path else ""), first=True)

    def _next(self, board, inv, path, allin: bool) -> PNode:
        """After a betting round closes: showdown on the river, else deal a card."""
        if len(board) == 5:
            return PNode("showdown", board, inv, path=path)
        node = PNode("chance", board, inv, path=path)
        for c in range(52):
            if c in board:
                continue
            nb = board + (c,)
            child = (self._next(nb, inv, path + card_str(c), True) if allin
                     else self._street(nb, inv, path + card_str(c)))
            node.children.append(child)
            node.cards.append(c)
        return node

    def _decide(self, board, inv, p, bet, raises, path, first=False, checked=False) -> PNode:
        cfg = self.cfg
        o = 1 - p
        node = PNode("decision", board, inv, player=p, path=path)
        self.decision_nodes += 1
        street = STREETS[len(board)]
        pot_now = cfg.pot + inv[0] + inv[1]
        remaining = cfg.stack - inv[p]
        to_call = bet[o] - bet[p]
        opts: list[tuple[str, PNode]] = []
        if to_call > 0:
            opts.append(("f", PNode("fold", board, inv, folder=p, path=path + "f")))
            call_inv = (inv[0] + (to_call if p == 0 else 0), inv[1] + (to_call if p == 1 else 0))
            opp_allin = cfg.stack - call_inv[o] <= 1e-9
            opts.append(("c", self._next(board, call_inv, path + "c", opp_allin or
                                         cfg.stack - call_inv[p] <= 1e-9)))
            sizes = cfg.raise_sizes.get(street, []) if raises < cfg.max_raises else []
            base = pot_now + to_call
        else:
            if checked:
                opts.append(("x", self._next(board, inv, path + "x", False)))
            else:
                opts.append(("x", self._decide(board, inv, o, bet, raises, path + "x", checked=True)))
            sizes = cfg.bet_sizes.get(street, [])
            base = pot_now
        can_raise = remaining > to_call + 1e-9 and (to_call == 0 or raises < cfg.max_raises)
        if can_raise and cfg.stack - inv[o] > 1e-9:
            amounts = []
            for f in sizes:
                add = to_call + f * base  # chips this player puts in now
                if add >= remaining * cfg.allin_threshold or add >= remaining - 1e-9:
                    continue
                amounts.append((f"{'r' if to_call else 'b'}{round(f * 100)}", add))
            if cfg.allin:
                amounts.append(("a", remaining))
            seen = set()
            for label, add in amounts:
                key = round(add, 6)
                if key in seen:
                    continue
                seen.add(key)
                new_inv = (inv[0] + (add if p == 0 else 0), inv[1] + (add if p == 1 else 0))
                new_bet = (bet[0] + (add if p == 0 else 0), bet[1] + (add if p == 1 else 0))
                opts.append((label, self._decide(board, new_inv, o, new_bet,
                                                 raises + (1 if to_call else 0), path + label)))
        node.actions = [a for a, _ in opts]
        node.children = [c for _, c in opts]
        node.regret = np.zeros((len(opts), N))
        node.strat_sum = np.zeros((len(opts), N))
        return node


# ---------------------------------------------------------------- solver


def _rm(regret: np.ndarray) -> np.ndarray:
    pos = np.maximum(regret, 0.0)
    tot = pos.sum(axis=0)
    return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / len(regret))


class SubgameCFR:
    """``variant``: "dcfr" (discounted CFR, alpha=1.5, beta=0, gamma=2; the
    default, usually fastest on poker subgames) or "cfr+"."""

    def __init__(self, board, ranges: tuple[np.ndarray, np.ndarray], cfg: TreeConfig,
                 variant: str = "dcfr") -> None:
        if variant not in ("dcfr", "cfr+"):
            raise ValueError(f"unknown CFR variant {variant!r}")
        self.variant = variant
        self.board = tuple(board)
        self.cfg = cfg
        mask = board_mask(self.board)
        self.ranges = (ranges[0] * mask, ranges[1] * mask)
        builder = TreeBuilder(cfg)
        self.root = builder.build(self.board)
        self.decision_nodes = builder.decision_nodes
        self._showdowns: dict[tuple[int, ...], Showdown] = {}
        self.t = 0

    def showdown(self, board) -> Showdown:
        key = tuple(sorted(board))
        if key not in self._showdowns:
            self._showdowns[key] = Showdown(board)
        return self._showdowns[key]

    def _terminal(self, node: PNode, i: int, r_opp: np.ndarray) -> np.ndarray:
        half = self.cfg.pot / 2
        if node.kind == "fold":
            mass = compatible_mass(r_opp)
            if node.folder == i:
                return -(half + node.inv[i]) * mass
            return (half + node.inv[1 - i]) * mass
        # Showdown: investments are equal by construction.
        return (half + node.inv[i]) * self.showdown(node.board).net_mass(r_opp)

    def _chance_factor(self, node: PNode) -> float:
        return 1.0 / (52 - len(node.board) - 4)

    def _walk(self, node, i, r_i, r_o, mode, avg=False) -> np.ndarray:
        """mode: 'cfr' updates regrets for i; 'br' returns i's best-response values;
        'ev' returns values with both players on ``avg`` strategies."""
        if node.kind in ("fold", "showdown"):
            return self._terminal(node, i, r_o)
        if node.kind == "chance":
            total = np.zeros(N)
            for c, child in zip(node.cards, node.children):
                m = np.ones(N)
                m[CARD_COMBOS[c]] = 0.0
                total += m * self._walk(child, i, r_i * m, r_o * m, mode, avg)
            return total * self._chance_factor(node)
        sigma = self._avg(node) if (avg or (mode == "br" and node.player != i)) else _rm(node.regret)
        if node.player != i:
            out = np.zeros(N)
            for k, child in enumerate(node.children):
                out += self._walk(child, i, r_i, r_o * sigma[k], mode, avg)
            return out
        vals = np.stack([self._walk(child, i, r_i * sigma[k], r_o, mode, avg)
                         for k, child in enumerate(node.children)])
        if mode == "br":
            return vals.max(axis=0)
        v = (sigma * vals).sum(axis=0)
        if mode == "cfr":
            node._delta = vals - v  # applied after the traversal (one sigma per iteration)
            if self.variant == "dcfr":
                node.strat_sum *= (self.t / (self.t + 1)) ** 2  # gamma = 2
                node.strat_sum += r_i * sigma
            else:
                node.strat_sum += self.t * r_i * sigma
        return v

    @staticmethod
    def _avg(node: PNode) -> np.ndarray:
        tot = node.strat_sum.sum(axis=0)
        return np.where(tot > 0, node.strat_sum / np.where(tot > 0, tot, 1.0), 1.0 / len(node.strat_sum))

    def _apply(self, node: PNode) -> None:
        if node.kind == "decision":
            if hasattr(node, "_delta"):
                if self.variant == "dcfr":
                    t = self.t
                    pos = t**1.5 / (t**1.5 + 1)  # alpha = 1.5
                    node.regret = np.where(node.regret > 0, node.regret * pos, node.regret * 0.5)
                    node.regret += node._delta  # beta = 0 halves negative regrets
                else:
                    node.regret = np.maximum(node.regret + node._delta, 0.0)
                del node._delta
        for c in node.children:
            self._apply(c)

    def iterate(self, n: int) -> None:
        for _ in range(n):
            self.t += 1
            for i in (0, 1):
                self._walk(self.root, i, self.ranges[i], self.ranges[1 - i], "cfr")
                self._apply(self.root)

    def _normaliser(self) -> float:
        r0, r1 = self.ranges
        return float((r0 * compatible_mass(r1)).sum())

    def exploitability(self) -> dict[str, float]:
        """Best-response values per player (chips per deal) against the average strategy."""
        norm = self._normaliser()
        br = [float((self.ranges[i] * self._walk(self.root, i, self.ranges[i], self.ranges[1 - i], "br")).sum()) / norm
              for i in (0, 1)]
        ev0 = float((self.ranges[0] * self._walk(self.root, 0, self.ranges[0], self.ranges[1], "ev", avg=True)).sum()) / norm
        nash_conv = br[0] + br[1]
        return {"nash_conv_chips": nash_conv,
                "exploitability_pct_pot": 100 * nash_conv / 2 / self.cfg.pot,
                "best_response_chips": br, "ev_oop_chips": ev0}

    def strategy(self, max_depth: int | None = None) -> dict[str, Any]:
        """Average strategy at every decision node of the current street's tree
        (and later streets' nodes when ``max_depth`` allows), for combos in range."""
        out: dict[str, Any] = {}

        def rec(node: PNode, dealt: int) -> None:
            if node.kind == "decision":
                avg = self._avg(node)
                combos = np.nonzero(self.ranges[node.player] > 0)[0]
                out[node.path or "root"] = {
                    "player": ["oop", "ip"][node.player],
                    "actions": node.actions,
                    "combos": {combo_str(i): [round(float(avg[k, i]), 4) for k in range(len(node.actions))]
                               for i in combos},
                }
            if node.kind == "chance":
                if max_depth is not None and dealt >= max_depth:
                    return
                dealt += 1
            for c in node.children:
                rec(c, dealt)

        rec(self.root, 0)
        return out


class SubgameSolver:
    """``SolveRequest("subgame", params, solver)`` with params: board,
    oop_range, ip_range, pot, stack, bet_sizes, raise_sizes, max_raises, allin."""

    def solve(self, request: SolveRequest) -> SolveResult:
        p = request.params
        board = parse_cards(p["board"])
        if len(board) not in (3, 4, 5):
            raise ValueError("board must have 3 to 5 cards")
        cfg = TreeConfig(
            pot=float(p["pot"]), stack=float(p["stack"]),
            bet_sizes={k: list(v) for k, v in p.get("bet_sizes", {}).items()},
            raise_sizes={k: list(v) for k, v in p.get("raise_sizes", {}).items()},
            max_raises=int(p.get("max_raises", 2)), allin=bool(p.get("allin", True)),
        )
        ranges = (parse_range(p["oop_range"], board), parse_range(p["ip_range"], board))
        if ranges[0].sum() == 0 or ranges[1].sum() == 0:
            raise ValueError("a range is empty on this board")
        iterations = int(request.solver.get("iterations", 300))
        target = request.solver.get("target_pct_pot")
        check_every = int(request.solver.get("check_every", 50))
        t0 = time.perf_counter()
        cfr = SubgameCFR(board, ranges, cfg, request.solver.get("variant", "dcfr"))
        history = []
        done = 0
        while done < iterations:
            step = min(check_every, iterations - done)
            cfr.iterate(step)
            done += step
            ex = cfr.exploitability()
            history.append((done, ex["exploitability_pct_pot"]))
            if target is not None and ex["exploitability_pct_pot"] <= float(target):
                break
        return SolveResult(
            request=request,
            strategy=cfr.strategy(max_depth=0),
            iterations=done,
            convergence={"metric": "exploitability_pct_pot", **ex,
                         "history": [{"iterations": i, "exploitability_pct_pot": e} for i, e in history]},
            seconds=time.perf_counter() - t0,
            strategy_source=StrategySource.SOLVER_APPROX,
            notes=[
                f"Heads-up subgame on {p['board']} with an abstracted bet tree; exact for that tree.",
                f"{cfr.decision_nodes} decision nodes; exploitability from an exact best response.",
            ],
        )
