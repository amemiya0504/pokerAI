"""The 0.5 BB ante candidate tables: only the agreed nodes get wider, and nothing else moves."""

import copy
import gzip
import json

import pytest

from backend.agents.web_preflop import TABLE_PATH
from scripts.experiments.web_preflop_ante_tables import (CONTINUE_CAP, VS_OPEN_RAISE_CAP, generate_ante, is_defense,
                                                         is_widened, static_problems, widen_calls_for, widen_for,
                                                         width_factor)
from scripts.experiments.web_preflop_v8_tables import generate, load_equity


@pytest.fixture(scope="module")
def tables():
    E, W = load_equity()
    base = generate(E, W, "v8.3")
    return base, generate_ante(E, W, .25, base), generate_ante(E, W, .25, base, defense=True)


def test_width_factor_matches_the_agreed_range():
    assert width_factor(6, .25) == pytest.approx(1.316, abs=.001)
    assert width_factor(8, .25) == pytest.approx(1.384, abs=.001)


@pytest.mark.parametrize("key,widened", [
    ("6/100/BTN/open:normal", True), ("6/100/BTN/limped:cheap", True), ("6/100/SB/option:normal", True),
    ("6/100/CO/vs_open_late_small:normal", True), ("6/100/SB/sb_vs_open_multi_late_small:cheap", True),
    ("6/100/BB/bb_vs_open_single_CO_small:normal", True), ("6/100/BB/bb_limped_multi:normal", True),
    ("6/100/BTN/vs_3bet_late_small:normal", False), ("6/100/BTN/vs_4bet_cold:normal", False),
    ("6/100/BTN/vs_open_allin:allin", False), ("6/100/BTN/limp_vs_raise:mdf", False)])
def test_only_the_agreed_nodes_are_widened(key, widened):
    assert is_widened(key) is widened


def test_caps_never_narrow_and_vs_open_raises_stay_capped():
    widen = widen_for(8, .35)
    assert widen("unopened", 1, .2)[0] == 1  # the option node already continues everything
    assert widen("unopened", .80, .2)[0] == CONTINUE_CAP
    assert widen("vs_open", .3, .12)[1] == VS_OPEN_RAISE_CAP
    assert widen("vs_open", .3, .14)[1] == .14
    assert widen("vs_3bet", .3, .1) == (.3, .1)


def test_candidate_passes_the_static_rules(tables):
    base, candidate, defended = tables
    assert static_problems(base, candidate) == []
    assert static_problems(base, defended, defense=True) == []


def test_static_rules_catch_a_narrower_hand_and_a_moved_node(tables):
    base, candidate, _ = tables
    broken = copy.deepcopy(candidate)
    key = "6/100/BTN/open:normal"
    broken["tables"][key]["probs"][0] = [1.0, 0.0, 0.0]
    assert any("less than v8.3" in p for p in static_problems(base, broken))
    broken = copy.deepcopy(candidate)
    broken["tables"]["6/100/BTN/vs_4bet_cold:normal"]["raise_multiple"] = 9
    assert any("unmodified" in p for p in static_problems(base, broken))


def test_base_generation_still_reproduces_the_runtime_table(tables):
    base, *_ = tables
    runtime = json.loads(gzip.decompress(TABLE_PATH.read_bytes()))
    assert base["tables"] == runtime["tables"]


@pytest.mark.parametrize("key,defends", [
    ("6/100/BTN/vs_3bet_late_small:normal", True), ("6/100/BTN/vs_3bet_early_large:cheap", True),
    ("6/100/BTN/vs_4bet_late:cheap", True), ("6/100/BTN/vs_4bet_late:normal", False),
    ("6/100/BTN/vs_4bet_cold:cheap", False), ("6/100/BTN/cold_vs_3bet:normal", False),
    ("6/100/BTN/vs_5bet:cheap", False), ("6/100/BTN/open:normal", False)])
def test_only_the_agreed_defense_nodes_change(key, defends):
    assert is_defense(key) is defends


def test_defense_widens_calls_but_not_raises():
    widen = widen_calls_for(8, .35)
    assert widen("vs_3bet", .1, .04) == (pytest.approx(.1 * width_factor(8, .35)), .04)
    assert widen("vs_3bet", .8, .04) == (CONTINUE_CAP, .04)
    assert widen("vs_3bet", .9, .04) == (.9, .04)  # never narrower, even above the cap


def test_defense_changes_calls_only_and_keeps_the_stored_equity(tables):
    base, _, defended = tables
    key = "6/100/BTN/vs_3bet_late_small:normal"
    old, new = base["tables"][key], defended["tables"][key]
    assert new["probs"] != old["probs"]
    assert new["equity"] == old["equity"] and new["raiser_width"] == old["raiser_width"]
    assert [row[2] for row in new["probs"]] == pytest.approx([row[2] for row in old["probs"]], abs=1e-3)
    assert sum(row[0] for row in new["probs"]) < sum(row[0] for row in old["probs"])
    untouched = "6/100/BTN/vs_4bet_late:normal"
    assert defended["tables"][untouched] == base["tables"][untouched]


@pytest.fixture
def ante_table_on(monkeypatch):
    from backend.agents import web_preflop
    from scripts.experiments.web_preflop_ante_tables import table_path

    monkeypatch.setattr(web_preflop, "ANTE_TABLE_PATH", table_path(.15, defense=True))
    web_preflop.ante_tables.cache_clear()
    yield
    web_preflop.ante_tables.cache_clear()


def _spot(ante=50, folds=0):
    from backend.engine.actions import Action
    from backend.engine.cards import Deck
    from backend.engine.hand import HandState

    state = HandState({s: 10000 for s in range(6)}, 0, 50, 100, Deck(7), ante=ante)
    for _ in range(folds):
        state.apply(Action.fold())
    return state.observation(state.to_act)


def _agent(three_bets=None, adaptive=True):
    """An agent whose model of seat 4 has ``three_bets = (re-raises, passes)``."""
    from backend.agents.web_realtime import WebRealtimeAgent
    from backend.opponent.model import OpponentModel
    from backend.opponent.stats import Observation

    models = {}
    if three_bets:
        models[4] = OpponentModel()
        for outcome, count in zip(("yes", "no"), three_bets):
            for _ in range(count):
                models[4].add(Observation(4, "three_bet", outcome, "BTN"))
    return WebRealtimeAgent(0, opponent_models=models, adaptive=adaptive)


def test_the_accepted_table_is_the_k35_switch():
    from backend.agents import web_preflop
    from scripts.experiments.web_preflop_ante_tables import table_path

    assert web_preflop.ANTE_TABLE_PATH.name == table_path(.35, defense=True).name
    assert web_preflop.ANTE_TABLE_PATH.exists()


def test_without_an_ante_table_nothing_switches(monkeypatch):
    from backend.agents import web_preflop

    monkeypatch.setattr(web_preflop, "ANTE_TABLE_PATH", None)
    web_preflop.ante_tables.cache_clear()
    try:
        assert _agent().ante_table(_spot()) is None
    finally:
        web_preflop.ante_tables.cache_clear()


@pytest.mark.parametrize("ante,used", [(50, True), (40, True), (60, True), (39, False), (61, False), (0, False), (100, False)])
def test_ante_table_needs_a_half_blind_ante(ante_table_on, ante, used):
    assert (_agent().ante_table(_spot(ante)) is not None) is used


@pytest.mark.parametrize("three_bets,used", [((8, 0), False), ((5, 3), False), ((7, 0), True), ((1, 9), True), (None, True)])
def test_ante_table_is_dropped_against_a_seen_re_raiser(ante_table_on, three_bets, used):
    assert (_agent(three_bets).ante_table(_spot()) is not None) is used


def test_a_folded_re_raiser_no_longer_counts(ante_table_on):
    assert _agent((8, 0)).ante_table(_spot()) is None
    view = _spot(folds=2)  # seats 3 and 4 fold first
    assert next(s for s in view.seats if s.seat == 4).folded
    assert _agent((8, 0)).ante_table(view) is not None


def test_a_session_that_is_not_adaptive_keeps_the_runtime_table(ante_table_on):
    assert _agent(adaptive=False).ante_table(_spot()) is None


def test_lookup_reads_the_table_it_is_given(tables):
    from backend.agents.web_preflop import lookup, tables as runtime
    from backend.engine.actions import Action
    from backend.engine.cards import Deck
    from backend.engine.hand import HandState

    _, wide, _ = tables
    state = HandState({s: 10000 for s in range(6)}, 0, 50, 100, Deck(7), ante=50)
    view, legal = state.observation(state.to_act), state.legal_actions()
    narrow = lookup(view, legal)[0]
    assert (lookup(view, legal, data=wide)[0][:, 0] <= narrow[:, 0] + 1e-9).all()
    assert (lookup(view, legal, data=wide)[0] != narrow).any()
    assert (lookup(view, legal)[0] == narrow).all() and runtime()["version"] != wide["version"]
