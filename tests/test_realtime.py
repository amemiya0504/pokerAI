"""Real-time solving agent (Phase A): public replay, ranges, legality, labelling."""

from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.agents import CallingStationAgent, RandomAgent  # noqa: E402
from backend.agents.base import StrategySource  # noqa: E402
from backend.agents.realtime import (  # noqa: E402
    DEFAULT_MODEL,
    RealtimeAgent,
    heuristic_likelihood,
    preflop_range,
    replay,
    top_range,
)
from backend.engine.actions import Action  # noqa: E402
from backend.engine.cards import Deck, derive_seed  # noqa: E402
from backend.engine.hand import HandState, Street  # noqa: E402


def _hu(seed, stacks=(10000, 10000), button=0):
    return HandState({0: stacks[0], 1: stacks[1]}, button, 50, 100, Deck(derive_seed(11, seed)))


def test_replay_reconstructs_the_public_state():
    rng = random.Random(3)
    for hn in range(150):
        h = _hu(hn, (rng.randint(300, 20000), rng.randint(300, 20000)), button=hn % 2)
        agents = {s: RandomAgent(derive_seed(hn, s), all_in_weight=0.1) for s in (0, 1)}
        while not h.is_over:
            s = h.to_act
            v = h.observation(s)
            if v.street is not Street.PREFLOP:
                _, st, oop = replay(v)
                assert oop != v.button
                assert st.street is v.street and st.board == tuple(v.board)
                for seat in (0, 1):
                    assert st.bets[seat] == pytest.approx(h.street_bet(seat))
                    assert st.behind_start[seat] == pytest.approx(h.stack(seat) + h.street_bet(seat))
                assert st.pot_start == pytest.approx(h.pot - h.street_bet(0) - h.street_bet(1))
            h.apply(agents[s].act(v, h.legal_actions()).action)


def test_preflop_range_lines_and_heuristics():
    h = _hu(1)
    h.apply(Action.raise_to(300))  # button raises
    h.apply(Action.call())  # big blind calls
    v = h.observation(0)
    r_raise, line_raise = preflop_range(v, 0)
    r_call, line_call = preflop_range(v, 1)
    assert (line_raise, line_call) == ("raise", "call a raise")
    assert (r_raise >= 0.02).all() and r_raise.max() == 1
    assert (r_raise == 1).sum() < (r_call == 1).sum()  # raising range is tighter
    assert top_range(1.0).sum() == 1326
    s = np.linspace(0, 1, 1326)
    from backend.engine.actions import ActionType

    bet = heuristic_likelihood(ActionType.BET, s)
    assert bet[-1] > bet[0] and (heuristic_likelihood(ActionType.FOLD, s)[0] > heuristic_likelihood(ActionType.FOLD, s)[-1])


def _play(agent_factory, opponent_factory, hands, seconds_check=None):
    decisions = []
    for hn in range(hands):
        h = _hu(hn, button=hn % 2)
        agents = {0: agent_factory(hn), 1: opponent_factory(hn)}
        while not h.is_over:
            s = h.to_act
            v = h.observation(s)
            d = agents[s].act(v, h.legal_actions())
            h.apply(d.action)  # raises if illegal
            if s == 0:
                decisions.append((v.street, d))
    return decisions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real-time solving test uses the GPU")
def test_agent_solves_turn_and_river_legally_and_labels_everything():
    decisions = _play(lambda hn: RealtimeAgent(hn, seconds=3.0), lambda hn: CallingStationAgent(), 8)
    by_street = {}
    for street, d in decisions:
        by_street.setdefault(street, set()).add(d.strategy_source)
        assert d.reason
        if street in (Street.TURN, Street.RIVER) and d.strategy_source is StrategySource.SOLVER_APPROX:
            assert "real-time solve" in d.reason and "% pot" in d.reason
            assert abs(sum(d.action_probs.values()) - 1) < 1e-3
        if street is Street.FLOP and d.strategy_source is StrategySource.SOLVER_APPROX:
            assert "depth-limited" in d.reason  # turn and river dealt without betting
    assert StrategySource.SOLVER_APPROX in by_street.get(Street.RIVER, set()) | by_street.get(Street.TURN, set())
    assert StrategySource.SOLVER_APPROX in by_street.get(Street.FLOP, set())
    assert all(StrategySource.VERIFIED_EQUILIBRIUM not in v for v in by_street.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real-time solving test uses the GPU")
def test_agent_handles_off_tree_bet_sizes():
    # The random opponent bets arbitrary amounts; the agent re-solves from the actual state.
    decisions = _play(lambda hn: RealtimeAgent(hn, seconds=2.0), lambda hn: RandomAgent(hn, all_in_weight=0.0), 6)
    assert decisions


def test_network_mode_and_missing_model_fall_back_with_labels(tmp_path):
    mode = "network" if DEFAULT_MODEL.exists() else None
    if mode:
        decisions = _play(lambda hn: RealtimeAgent(hn, mode="network"), lambda hn: CallingStationAgent(), 4)
        late = [d for st, d in decisions if st in (Street.TURN, Street.RIVER)]
        assert late and all(d.strategy_source is StrategySource.NEURAL_APPROX for d in late)
    decisions = _play(lambda hn: RealtimeAgent(hn, mode="network", model_path=tmp_path / "missing.pt"),
                      lambda hn: CallingStationAgent(), 4)
    late = [d for st, d in decisions if st in (Street.TURN, Street.RIVER)]
    assert late and all(d.strategy_source is StrategySource.HEURISTIC and "no policy network" in d.reason
                        for d in late)


def test_not_heads_up_is_heuristic():
    h = HandState({0: 10000, 1: 10000, 2: 10000}, 0, 50, 100, Deck(1))
    agent = RealtimeAgent(0, mode="network")
    station = CallingStationAgent()
    while not h.is_over:
        s = h.to_act
        v = h.observation(s)
        d = (agent if s == 0 else station).act(v, h.legal_actions())
        if s == 0 and v.street is not Street.PREFLOP:
            assert d.strategy_source is StrategySource.HEURISTIC and "not heads-up" in d.reason
        h.apply(d.action)


def test_opponent_raising_past_the_abstraction_cap():
    # Regression: an observed raise beyond the tree's raise cap had no matching action.
    h = _hu(1)
    agent = RealtimeAgent(1, mode="network")
    while h.street is not Street.RIVER:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    for to in (100, 400, 1200, 3000, 6000):  # bet, then four raises
        h.apply(Action.bet(to) if to == 100 else Action.raise_to(to))
    v = h.observation(h.to_act)
    h.apply(agent.act(v, h.legal_actions()).action)
