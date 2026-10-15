"""Kuhn and Leduc: rules, payoffs and NashConv checked against OpenSpiel.

OpenSpiel has limited Windows support, so this module is skipped there.
"""

from __future__ import annotations

import pytest

from backend.solver.cfr import GAMES, TabularCFRPlus, nash_conv

pyspiel = pytest.importorskip("pyspiel")


def _openspiel_nash_conv(name: str, ours: TabularCFRPlus) -> float:
    """Load our average strategy into an OpenSpiel tabular policy and let
    OpenSpiel compute NashConv on its own implementation of the game."""
    from open_spiel.python import policy as os_policy
    from open_spiel.python.algorithms import exploitability

    game = pyspiel.load_game(name + "_poker")
    tab = os_policy.TabularPolicy(game)
    avg = ours.average()
    ids = {k: i for i, k in enumerate(ours.tree.infosets)}
    g = ours.tree.game

    def walk(state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for a, _ in state.chance_outcomes():
                walk(state.child(a))
            return
        h = tuple(state.history())
        probs = avg[ids[g.infoset(h)]]
        row = tab.policy_for_key(state.information_state_string())
        row[:] = 0
        for a, p in zip(g.actions(h), probs):
            row[a] = p
        assert sorted(g.actions(h)) == sorted(state.legal_actions())
        for a in state.legal_actions():
            walk(state.child(a))

    walk(game.new_initial_state())
    return exploitability.nash_conv(game, tab)


@pytest.mark.parametrize("name, iterations", [("kuhn", 200), ("leduc", 60)])
def test_openspiel_agrees_on_rules_and_nash_conv(name, iterations):
    cfr = TabularCFRPlus(GAMES[name]())
    cfr.iterate(iterations)
    ours, _, _ = nash_conv(cfr.tree, cfr.average())
    assert _openspiel_nash_conv(name, cfr) == pytest.approx(ours, abs=1e-9)


@pytest.mark.parametrize("name", ["kuhn", "leduc"])
def test_returns_match_openspiel(name):
    game = pyspiel.load_game(name + "_poker")
    ours = GAMES[name]()

    def walk(state):
        h = tuple(state.history())
        assert ours.is_terminal(h) == state.is_terminal()
        if state.is_terminal():
            assert tuple(ours.returns(h)) == pytest.approx(tuple(state.returns()))
            return
        assert ours.is_chance(h) == state.is_chance_node()
        acts = [a for a, _ in state.chance_outcomes()] if state.is_chance_node() else state.legal_actions()
        if not state.is_chance_node():
            assert ours.player(h) == state.current_player()
        for a in acts:
            walk(state.child(a))

    walk(game.new_initial_state())
