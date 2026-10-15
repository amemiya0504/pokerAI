"""Heads-up postflop subgame solver: terminal maths, toy game theory, convergence."""

from __future__ import annotations

import random

import numpy as np
import pytest

from backend.agents.base import StrategySource
from backend.engine.cards import parse_cards
from backend.engine.evaluator import evaluate
from backend.solver.base import SolveRequest
from backend.solver.subgame import (
    COMBO_LIST,
    N,
    Showdown,
    SubgameCFR,
    SubgameSolver,
    TreeConfig,
    board_mask,
    compatible_mass,
    parse_range,
)


def _random_reach(rng, board):
    return np.array([rng.random() if rng.random() < 0.3 else 0.0 for _ in range(N)]) * board_mask(board)


def test_showdown_and_fold_masses_match_brute_force():
    rng = random.Random(4)
    for _ in range(3):
        board = tuple(rng.sample(range(52), 5))
        r = _random_reach(rng, board)
        sd = Showdown(board)
        fast_net, fast_compat = sd.net_mass(r), compatible_mass(r)
        strength = {i: evaluate((*COMBO_LIST[i], *board)) for i in range(N) if board_mask(board)[i]}
        for h in rng.sample(sorted(strength), 60):
            hs = set(COMBO_LIST[h])
            net = compat = 0.0
            for o, w in enumerate(r):
                if w == 0 or hs & set(COMBO_LIST[o]):
                    continue
                compat += w
                net += w * np.sign(strength[h] - strength[o])
            assert fast_net[h] == pytest.approx(net)
            assert fast_compat[h] == pytest.approx(compat)


def test_parse_range():
    w = parse_range("QQ+,AKs,A5s:0.5")
    assert w.sum() == pytest.approx(3 * 6 + 4 + 0.5 * 4)
    assert parse_range("ATs+").sum() == 4 * 4
    assert parse_range("AK").sum() == 16
    assert parse_range("AsKs").sum() == 1
    assert parse_range("all", parse_cards("AsKdQh")).sum() == 1176  # C(49, 2)
    with pytest.raises(ValueError):
        parse_range("XYz")


TOY = {
    "board": "Ks7d4c2h9s", "pot": 10, "stack": 1000,
    "oop_range": "QQ,JJ", "ip_range": "77,44,J8s,T8s,T6s,86s,65s,J5s",
    "bet_sizes": {"river": [1.0]}, "raise_sizes": {}, "max_raises": 0, "allin": False,
}


def test_polarized_river_matches_theory():
    """IP: nuts or air. OOP: bluff-catchers. Pot-sized bet: IP bluffs 1/3 of
    its betting range, OOP calls 1/2, OOP never leads."""
    r = SubgameSolver().solve(SolveRequest("subgame", TOY, {"iterations": 1000, "check_every": 500}))
    assert r.convergence["exploitability_pct_pot"] < 0.05
    assert r.strategy_source is StrategySource.SOLVER_APPROX
    after_check = r.strategy["x"]
    b = after_check["actions"].index("b100")
    value = sum(v[b] for k, v in after_check["combos"].items() if k[0] == k[2])
    air = sum(v[b] for k, v in after_check["combos"].items() if k[0] != k[2])
    assert value == pytest.approx(6, abs=0.05)
    assert air / (air + value) == pytest.approx(1 / 3, abs=0.02)
    facing = r.strategy["xb100"]
    c = facing["actions"].index("c")
    assert np.mean([v[c] for v in facing["combos"].values()]) == pytest.approx(0.5, abs=0.03)
    assert np.mean([v[0] for v in r.strategy["root"]["combos"].values()]) > 0.99


def test_cfr_variants_converge_and_dcfr_is_default():
    board = parse_cards("Ah9c6d3s2h")
    ranges = (parse_range("99+,AJs+,AQo+,T9s,87s", board), parse_range("77+,ATs+,KQs,JTs,98s,AJo+", board))
    cfg = TreeConfig(10, 30, {"river": [0.5, 1.0]}, {"river": [1.0]}, 1, True)
    for variant in ("dcfr", "cfr+"):
        cfr = SubgameCFR(board, ranges, cfg, variant)
        cfr.iterate(10)
        early = cfr.exploitability()["exploitability_pct_pot"]
        cfr.iterate(140)
        late = cfr.exploitability()
        assert late["exploitability_pct_pot"] < min(early, 1.0)
        assert late["best_response_chips"][0] + late["best_response_chips"][1] >= -1e-9
    assert SubgameCFR(board, ranges, cfg).variant == "dcfr"


def test_turn_subgame_reports_convergence_history():
    params = {"board": "Ks7d4c2h", "pot": 10, "stack": 15, "oop_range": "TT+,AKs",
              "ip_range": "99+,AQs+,KQs", "bet_sizes": {"turn": [1.0], "river": [1.0]},
              "raise_sizes": {}, "max_raises": 0, "allin": True}
    r = SubgameSolver().solve(SolveRequest("subgame", params, {"iterations": 60, "check_every": 30}))
    hist = r.convergence["history"]
    assert [h["iterations"] for h in hist] == [30, 60]
    assert hist[-1]["exploitability_pct_pot"] <= hist[0]["exploitability_pct_pot"]
    assert r.request.key() == SolveRequest("subgame", params, {"iterations": 60, "check_every": 30}).key()


def test_zero_sum_with_identical_ranges_and_no_betting():
    # With no bets allowed both players just show down: EV is zero for identical ranges.
    params = {**TOY, "oop_range": "all", "ip_range": "all", "bet_sizes": {}, "raise_sizes": {}}
    r = SubgameSolver().solve(SolveRequest("subgame", params, {"iterations": 1, "check_every": 1}))
    assert r.convergence["ev_oop_chips"] == pytest.approx(0, abs=1e-9)
    assert r.convergence["nash_conv_chips"] == pytest.approx(0, abs=1e-9)
