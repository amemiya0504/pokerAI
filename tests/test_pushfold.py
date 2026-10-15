"""Milestone 4: push/fold solver, datasets with provenance, and the agent's labelling."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from backend.agents.base import StrategySource
from backend.agents.pushfold import PushFoldAgent, preflop_order
from backend.engine.actions import Action
from backend.engine.cards import Deck, derive_seed
from backend.engine.hand import HandState
from backend.solver.hands import CLASS_NAMES, COMBOS, N_CLASSES, class_index
from backend.solver.pushfold import (
    PushFoldGame,
    PushFoldSolver,
    _Traverser,
    hu_cfr_plus,
    hu_equity_matrix,
    hu_exploitability,
    request,
)
from backend.solver.store import PUSHFOLD_DIR, compact, load_all, write

REQUIRED = {"dataset", "version", "strategy_source", "solver", "license", "game",
            "solver_config", "request_hash", "iterations", "convergence", "notes", "strategy"}


def test_hand_classes():
    assert len(COMBOS) == N_CLASSES and sum(len(c) for c in COMBOS) == 1326
    assert sorted(len(c) for c in COMBOS)[::-1][:1] == [12]
    assert {len(COMBOS[i]) for i in range(N_CLASSES)} == {4, 6, 12}
    assert CLASS_NAMES[0] == "AA" and CLASS_NAMES[1] == "AKs" and CLASS_NAMES[13] == "AKo"
    assert all(class_index(a, b) == i for i, cs in enumerate(COMBOS) for a, b in cs)


@pytest.mark.parametrize("n", range(2, 10))
def test_tree_size_and_zero_sum_payoffs(n):
    game = PushFoldGame(n, 10)
    # n-1 unopened nodes plus, for each pusher f, 2^(n-1-f) - 1 facing-push nodes.
    assert len(game.infosets()) == (n - 1) + sum(2 ** (n - 1 - f) - 1 for f in range(n - 1))
    tr = _Traverser(game)
    rng = random.Random(n)
    for _ in range(200):
        scores = [rng.randint(0, 5) for _ in range(n)]
        f = rng.randint(-1, n - 2)
        mask = 0 if f < 0 else sum(1 << i for i in range(f + 1, n) if rng.random() < 0.5)
        assert sum(tr.payoff(t, f, mask, scores) for t in range(n)) == pytest.approx(0)


@pytest.fixture(scope="module")
def hu_matrix():
    return hu_equity_matrix(samples=150, seed=3)


def test_heads_up_mccfr_matches_exact_solver(hu_matrix):
    E, W = hu_matrix
    push, call = hu_cfr_plus(E, W, 10, 2000)
    exact = hu_exploitability(E, W, 10, push, call)
    assert exact["nash_conv_bb"] < 1e-4
    combos = np.array([len(c) for c in COMBOS])
    assert 0.55 < (push * combos).sum() / 1326 < 0.61  # published HU 10bb: ~58% / ~37%
    assert 0.34 < (call * combos).sum() / 1326 < 0.41

    res = PushFoldSolver().solve(request(2, 10, batches=60, deals_per_batch=40_000,
                                         eval_deals=200_000, seed=7))
    mp = np.array(res.strategy["SB|open"]["aggress_prob"])
    mc = np.array(res.strategy["BB|vs SB push"]["aggress_prob"])
    # The sampled solver's strategy, scored exactly against the equity matrix.
    assert hu_exploitability(E, W, 10, mp, mc)["nash_conv_bb"] < 0.01
    c = res.convergence
    assert c["nash_conv_lower_bb"] <= c["nash_conv_upper_bb"]
    assert res.strategy_source is StrategySource.SOLVER_APPROX


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    d = tmp_path_factory.mktemp("pushfold")
    for n in (2, 3):
        res = PushFoldSolver().solve(request(n, 10, batches=20, deals_per_batch=20_000,
                                             eval_deals=50_000, seed=1))
        write(res, d)
    return d


def test_compact_nulls_rarely_visited_cells(small_data):
    res = PushFoldSolver(1).solve(request(3, 10, batches=2, deals_per_batch=2000, eval_deals=600))
    data = compact(res, min_visits=20)  # ~24 visits per class at the unopened nodes
    cells = [p for c in data["strategy"].values() for p in c["aggress_prob"]]
    assert any(p is None for p in cells) and any(p is not None for p in cells)
    assert REQUIRED <= set(data)


def check_dataset(data):
    assert REQUIRED <= set(data)
    assert data["strategy_source"] == "solver_approx"
    assert "no third-party data" in data["license"]
    c = data["convergence"]
    assert {"nash_conv_upper_bb", "nash_conv_lower_bb", "method", "per_player"} <= set(c)
    game = PushFoldGame(data["game"]["players"], data["game"]["stack_bb"])
    assert set(data["strategy"]) == {game.infoset_name(k) for k in game.infosets()}
    for cell in data["strategy"].values():
        assert len(cell["aggress_prob"]) == N_CLASSES
        assert all(p is None or 0 <= p <= 1 for p in cell["aggress_prob"])


def test_generated_datasets_carry_provenance(small_data):
    data = load_all(small_data)
    assert set(data) == {(2, 10.0), (3, 10.0)}
    for d in data.values():
        check_dataset(d)


def test_committed_datasets_carry_provenance():
    data = load_all(PUSHFOLD_DIR)
    if not data:
        pytest.skip("no committed push/fold datasets")
    for d in data.values():
        check_dataset(d)
    assert {p for p, _ in data} == set(range(2, 10))


def _hand(stacks, button=0, seed=1):
    return HandState(stacks, button, 50, 100, Deck(seed))


def test_preflop_order():
    h = _hand({0: 1000, 1: 1000, 2: 1000, 3: 1000}, button=1)
    v = h.observation(0)
    # button 1 -> SB 2, BB 3, then UTG 0, BTN 1.
    assert preflop_order(v) == [0, 1, 2, 3]
    hu = _hand({4: 1000, 7: 1000}, button=7)
    assert preflop_order(hu.observation(4)) == [7, 4]


def test_agent_labels_solved_and_unsolved_decisions(small_data):
    agent = PushFoldAgent(1, data_dir=small_data)
    # 3-handed at 10bb, unopened: inside the solved game.
    h = _hand({0: 1000, 1: 1000, 2: 1000})
    d = agent.act(h.observation(h.to_act), h.legal_actions())
    assert d.strategy_source is StrategySource.SOLVER_APPROX
    assert set(d.action_probs) == {"fold", "push"} and "pushfold 3max 10bb" in d.reason
    # 100bb deep: no data, falls back and says so.
    deep = _hand({0: 10_000, 1: 10_000, 2: 10_000})
    d = agent.act(deep.observation(deep.to_act), deep.legal_actions())
    assert d.strategy_source is StrategySource.HEURISTIC and d.reason.startswith("no solved data")
    # A limp is off the push/fold tree.
    h = _hand({0: 1000, 1: 1000, 2: 1000})
    h.apply(Action.call())
    d = agent.act(h.observation(h.to_act), h.legal_actions())
    assert d.strategy_source is StrategySource.HEURISTIC and "off-tree" in d.reason
    # 6-handed: no dataset in this directory.
    h = _hand({s: 1000 for s in range(6)})
    d = agent.act(h.observation(h.to_act), h.legal_actions())
    assert d.strategy_source is StrategySource.HEURISTIC and "6-handed" in d.reason


def test_agent_facing_push_and_legal_everywhere(small_data):
    rng = random.Random(5)
    agents = {}
    for hand_no in range(300):
        n = rng.choice([2, 3])
        stacks = {s: rng.randint(700, 1300) for s in range(n)}
        h = _hand(stacks, button=rng.randrange(n), seed=derive_seed(5, hand_no))
        sources = []
        while not h.is_over:
            seat = h.to_act
            agent = agents.setdefault(seat, PushFoldAgent(seat, data_dir=small_data))
            d = agent.act(h.observation(seat), h.legal_actions())
            h.apply(d.action)  # raises if illegal
            sources.append((d.strategy_source, d.reason))
        for src, reason in sources:
            assert src in (StrategySource.SOLVER_APPROX, StrategySource.HEURISTIC)
            if src is StrategySource.HEURISTIC:
                assert reason.startswith("no solved data")


def test_dataset_file_is_plain_json(small_data):
    raw = (small_data / "2max_10bb.json").read_text()
    data = json.loads(raw)
    assert data["game"] == {"players": 2, "stack_bb": 10, "small_blind_bb": 0.5, "ante": 0}
