"""Tabular CFR+ with exact best responses, for small games (Kuhn, Leduc).

The whole game tree is enumerated, so exploitability is exact: NashConv is
the sum of both players' best-response gains against the average strategy.
A result is labelled VERIFIED_EQUILIBRIUM only when NashConv is below the
request's threshold; otherwise SOLVER_APPROX.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.agents.base import StrategySource

from .base import SolveRequest, SolveResult
from .games import Kuhn, Leduc, SmallGame

GAMES: dict[str, type] = {"kuhn": Kuhn, "leduc": Leduc}
DEFAULT_THRESHOLD = 1e-3  # NashConv (chips per game) required for VERIFIED_EQUILIBRIUM


@dataclass
class _Node:
    kind: int  # 0 chance, 1 decision, 2 terminal
    player: int = -1
    infoset: int = -1
    children: list[int] = field(default_factory=list)
    probs: list[float] = field(default_factory=list)  # chance probabilities
    returns: tuple[float, float] = (0.0, 0.0)


class Tree:
    def __init__(self, game: SmallGame) -> None:
        self.game = game
        self.nodes: list[_Node] = []
        self.infosets: list[str] = []
        self.infoset_player: list[int] = []
        self.infoset_actions: list[list[int]] = []
        self.histories: list[tuple[int, ...]] = []
        self._ids: dict[str, int] = {}
        self._build(())

    def _build(self, h: tuple[int, ...]) -> int:
        g = self.game
        idx = len(self.nodes)
        self.histories.append(h)
        if g.is_terminal(h):
            self.nodes.append(_Node(2, returns=g.returns(h)))
            return idx
        if g.is_chance(h):
            node = _Node(0)
            self.nodes.append(node)
            for a, p in g.chance_outcomes(h):
                node.children.append(self._build(h + (a,)))
                node.probs.append(p)
            return idx
        key = g.infoset(h)
        acts = g.actions(h)
        if key not in self._ids:
            self._ids[key] = len(self.infosets)
            self.infosets.append(key)
            self.infoset_player.append(g.player(h))
            self.infoset_actions.append(acts)
        node = _Node(1, player=g.player(h), infoset=self._ids[key])
        self.nodes.append(node)
        for a in acts:
            node.children.append(self._build(h + (a,)))
        return idx


class TabularCFRPlus:
    def __init__(self, game: SmallGame) -> None:
        self.tree = Tree(game)
        sizes = [len(a) for a in self.tree.infoset_actions]
        self.regret = [np.zeros(n) for n in sizes]
        self.strat_sum = [np.zeros(n) for n in sizes]
        self.t = 0

    def current(self, i: int) -> np.ndarray:
        r = np.maximum(self.regret[i], 0)
        s = r.sum()
        return r / s if s > 0 else np.full(len(r), 1 / len(r))

    def average(self) -> list[np.ndarray]:
        out = []
        for s in self.strat_sum:
            tot = s.sum()
            out.append(s / tot if tot > 0 else np.full(len(s), 1 / len(s)))
        return out

    def _cfr(self, n: int, i: int, reach_i: float, reach_o: float, sigma, delta) -> float:
        node = self.tree.nodes[n]
        if node.kind == 2:
            return node.returns[i]
        if node.kind == 0:
            return sum(p * self._cfr(c, i, reach_i, reach_o * p, sigma, delta)
                       for c, p in zip(node.children, node.probs))
        s = sigma[node.infoset]
        if node.player != i:
            return sum(s[k] * self._cfr(c, i, reach_i, reach_o * s[k], sigma, delta)
                       for k, c in enumerate(node.children) if s[k] > 0)
        vals = np.array([self._cfr(c, i, reach_i * s[k], reach_o, sigma, delta)
                         for k, c in enumerate(node.children)])
        v = float(s @ vals)
        delta[node.infoset] += reach_o * (vals - v)
        self.strat_sum[node.infoset] += self.t * reach_i * s  # linear averaging
        return v

    def iterate(self, iterations: int) -> None:
        for _ in range(iterations):
            self.t += 1
            for i in (0, 1):  # alternating updates
                # One fixed strategy per traversal: an infoset is visited once
                # per history in it, and every visit must see the same sigma.
                sigma = [self.current(k) for k in range(len(self.regret))]
                delta = [np.zeros_like(r) for r in self.regret]
                self._cfr(0, i, 1.0, 1.0, sigma, delta)
                for r, d in zip(self.regret, delta):
                    r += d
                    np.maximum(r, 0, out=r)  # CFR+: floor regrets at zero


def expected_value(tree: Tree, policy: list[np.ndarray]) -> float:
    """Player 0's expected value when both players follow ``policy``."""

    def ev(n: int) -> float:
        node = tree.nodes[n]
        if node.kind == 2:
            return node.returns[0]
        probs = node.probs if node.kind == 0 else policy[node.infoset]
        return sum(p * ev(c) for c, p in zip(node.children, probs))

    return ev(0)


def best_response_value(tree: Tree, policy: list[np.ndarray], i: int) -> float:
    """Exact value player i gets by best-responding to ``policy``."""
    # Collect each of i's infosets' histories with their opponent+chance reach.
    members: dict[int, list[tuple[int, float]]] = {}

    def collect(n: int, reach: float) -> None:
        node = self_nodes[n]
        if node.kind == 2:
            return
        if node.kind == 0:
            for c, p in zip(node.children, node.probs):
                collect(c, reach * p)
        elif node.player == i:
            members.setdefault(node.infoset, []).append((n, reach))
            for c in node.children:
                collect(c, reach)
        else:
            for c, p in zip(node.children, policy[node.infoset]):
                if p > 0:
                    collect(c, reach * p)

    self_nodes = tree.nodes
    collect(0, 1.0)
    choice: dict[int, int] = {}
    memo: dict[int, float] = {}

    def value(n: int) -> float:
        if n in memo:
            return memo[n]
        node = self_nodes[n]
        if node.kind == 2:
            v = node.returns[i]
        elif node.kind == 0:
            v = sum(p * value(c) for c, p in zip(node.children, node.probs))
        elif node.player != i:
            v = sum(p * value(c) for c, p in zip(node.children, policy[node.infoset]) if p > 0)
        else:
            I = node.infoset
            if I not in choice:
                k = len(node.children)
                totals = [sum(w * value(self_nodes[h].children[a]) for h, w in members[I])
                          for a in range(k)]
                choice[I] = int(np.argmax(totals))
            v = value(node.children[choice[I]])
        memo[n] = v
        return v

    return value(0)


def nash_conv(tree: Tree, policy: list[np.ndarray]) -> tuple[float, float, float]:
    """(NashConv, player-0 value, player-1 value) for a two-player zero-sum game."""
    br0 = best_response_value(tree, policy, 0)
    br1 = best_response_value(tree, policy, 1)
    return br0 + br1, br0, br1


class SmallGameSolver:
    def solve(self, request: SolveRequest) -> SolveResult:
        if request.game not in GAMES:
            raise ValueError(f"unknown small game {request.game!r}")
        iterations = int(request.solver.get("iterations", 1000))
        threshold = float(request.solver.get("threshold", DEFAULT_THRESHOLD))
        t0 = time.perf_counter()
        cfr = TabularCFRPlus(GAMES[request.game]())
        cfr.iterate(iterations)
        avg = cfr.average()
        nc, br0, br1 = nash_conv(cfr.tree, avg)
        value = expected_value(cfr.tree, avg)
        verified = nc < threshold
        tree = cfr.tree
        strategy = {
            tree.infosets[k]: {"actions": tree.infoset_actions[k],
                               "probs": [round(float(p), 6) for p in avg[k]]}
            for k in range(len(tree.infosets))
        }
        return SolveResult(
            request=request,
            strategy=strategy,
            iterations=iterations,
            convergence={"metric": "nash_conv_exact", "nash_conv": nc, "threshold": threshold,
                         "best_response_values": [br0, br1], "game_value_p0": value},
            seconds=time.perf_counter() - t0,
            strategy_source=(StrategySource.VERIFIED_EQUILIBRIUM if verified
                             else StrategySource.SOLVER_APPROX),
            notes=[f"Exact NashConv {nc:.2e} {'<' if verified else '>='} threshold {threshold:g}."],
        )
