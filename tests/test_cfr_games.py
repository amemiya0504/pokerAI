"""Kuhn and Leduc: tabular CFR+ against analytic results and OpenSpiel."""

from __future__ import annotations

import pytest

from backend.agents.base import StrategySource
from backend.solver.base import SolveRequest
from backend.solver.cfr import SmallGameSolver, TabularCFRPlus, Tree, expected_value, nash_conv
from backend.solver.games import Kuhn, Leduc


@pytest.fixture(scope="module")
def kuhn():
    return SmallGameSolver().solve(SolveRequest("kuhn", {}, {"iterations": 2000}))


def test_kuhn_matches_analytic_solution(kuhn):
    c = kuhn.convergence
    assert c["game_value_p0"] == pytest.approx(-1 / 18, abs=1e-3)
    assert c["nash_conv"] < 1e-3
    assert kuhn.strategy_source is StrategySource.VERIFIED_EQUILIBRIUM
    s = {k: v["probs"] for k, v in kuhn.strategy.items()}
    alpha = s["J"][1]  # P1's bluff frequency with J; the equilibrium family has alpha in [0, 1/3]
    assert 0 <= alpha <= 1 / 3 + 1e-3
    assert s["K"][1] == pytest.approx(3 * alpha, abs=0.02)  # value bets K three times as often
    assert s["Q"][1] == pytest.approx(0, abs=0.01)  # never bets Q
    assert s["Qpb"][1] == pytest.approx(alpha + 1 / 3, abs=0.02)  # calls with Q at alpha + 1/3
    assert s["Jp"][1] == pytest.approx(1 / 3, abs=0.02)  # P2 bluffs J 1/3 after a check
    assert s["Qb"][1] == pytest.approx(1 / 3, abs=0.02)  # P2 calls with Q 1/3


def test_unconverged_result_is_not_verified():
    r = SmallGameSolver().solve(SolveRequest("kuhn", {}, {"iterations": 3, "threshold": 1e-6}))
    assert r.strategy_source is StrategySource.SOLVER_APPROX


def test_leduc_value_and_convergence():
    cfr = TabularCFRPlus(Leduc())
    cfr.iterate(300)
    nc, _, _ = nash_conv(cfr.tree, cfr.average())
    assert nc < 0.01
    assert expected_value(cfr.tree, cfr.average()) == pytest.approx(-0.0856, abs=0.003)
    assert len(cfr.tree.infosets) == 288  # suits merged: 3 ranks instead of 6 cards


def test_tree_sizes():
    assert len(Tree(Kuhn()).infosets) == 12
