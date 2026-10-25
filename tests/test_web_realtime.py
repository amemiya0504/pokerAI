"""Web policy isolation, frozen lookup, public ranges, CPU legality and labelling."""

from dataclasses import replace

import numpy as np
import pytest

from backend.agents import web_preflop
from backend.agents.base import StrategySource
from backend.agents.pushfold import PushFoldAgent, preflop_order
from backend.agents.web_preflop import (POSITIONS, TABLE_PATH, context, contexts_for, lookup, position,
                                       probabilities, ranges, tables)
from backend.agents.web_realtime import WebRealtimeAgent, _WebHeadsUp, equity, heuristic
from backend.engine.actions import Action
from backend.engine.cards import Deck
from backend.engine.hand import HandState, Street
from backend.solver.hands import CLASS_NAMES, COMBOS, class_index


V7_PATH = TABLE_PATH.with_name("web_preflop_v7.json")


@pytest.fixture(autouse=True)
def v7_tables(monkeypatch):
    """These tests pin the schema-1 v7 table; v8 has its own tests."""
    monkeypatch.setattr(web_preflop, "TABLE_PATH", V7_PATH)
    tables.cache_clear()
    yield
    tables.cache_clear()


def hand(n=2, seed=1, depth=100):
    seats = (0, 2, 4, 6, 8, 1, 3, 5, 7)[:n]
    return HandState({s: depth * 100 for s in seats}, 0, 50, 100, Deck(seed))


def test_frozen_schema_all_44_positions_and_generator_reproducibility():
    import json

    from scripts.experiments.web_preflop_tables import generate

    from pathlib import Path

    runtime = tables()
    data = generate()
    assert sum(map(len, POSITIONS.values())) == 44
    assert len(data["tables"]) == sum(len(contexts_for(n, pos)) for n, ps in POSITIONS.items() for pos in ps)
    assert runtime == generate(a3_runtime=True) == json.loads(V7_PATH.read_text())
    assert data == json.loads(Path("backend/agents/data/web_preflop_v6.json").read_text())
    assert runtime["version"] == "web-v7-a3-no-floor"
    assert "not GTO" in runtime["source"]
    for n, expected in ((5, .47), (7, .38), (9, .38)):
        bb = np.array(data["tables"][f"{n}/BB/vs_open_late_medium:normal"]["probs"])
        assert np.mean(1 - bb[:, 0]) == pytest.approx(expected, abs=.02)
    sb_open = data["tables"]["8/SB/open:normal"]["probs"]
    sb_rfi = sum(len(COMBOS[i]) * row[2] for i, row in enumerate(sb_open)) / 1326
    assert sb_rfi == pytest.approx(.41, abs=.01)
    assert data["tables"]["6/SB/vs_open_large:cheap"]["probs"][CLASS_NAMES.index("72o")][0] > .65
    for n in (2, 6, 9):
        opening = [data["tables"][f"{n}/{pos}/open:normal"]["probs"] for pos in POSITIONS[n]]
        assert len({json.dumps(rows, separators=(",", ":")) for rows in opening}) == len(POSITIONS[n])
        for pos in POSITIONS[n]:
            for ctx in contexts_for(n, pos):
                probs = np.array(data["tables"][f"{n}/{pos}/{ctx}"]["probs"])
                assert probs.shape == (169, 3)
                assert np.allclose(probs.sum(axis=1), 1)
                assert 0 < probs[:, 0].mean() < 1


def test_weak_hand_has_no_forced_preflop_continue_or_raise_floor():
    row = tables()["tables"]["6/SB/vs_open_large:cheap"]["probs"][CLASS_NAMES.index("72o")]

    assert row == [1.0, 0.0, 0.0]


@pytest.mark.parametrize("n", range(2, 10))
def test_position_mapping_with_gaps_and_rotated_buttons(n):
    for button in range(n):
        seats = (0, 2, 4, 6, 8, 1, 3, 5, 7)[:n]
        h = HandState({s: 10000 for s in seats}, seats[button], 50, 100, Deck(2))
        view = h.observation(h.to_act)
        order = preflop_order(view)
        assert [position(view, seat) for seat in order] == list(POSITIONS[n])
        assert position(view, view.button) == ("BTN/SB" if n == 2 else "BTN")


def test_lookup_context_and_legal_projection():
    h = hand()
    v, legal = h.observation(h.to_act), h.legal_actions()
    assert context(v, legal) == "open:cheap"  # SB has 3:1 price
    probs, _, key = lookup(v, legal)
    assert key == "2/BTN/SB/open:cheap"
    assert probs[CLASS_NAMES.index("AA"), 2] > probs[CLASS_NAMES.index("72o"), 2]
    h.apply(Action.call())
    v, legal = h.observation(h.to_act), h.legal_actions()
    assert context(v, legal) == "option:normal"
    probs, _, _ = lookup(v, replace(legal, raise_type=None, min_raise_to=None, max_raise_to=None))
    assert np.allclose(probs[:, 1], 1) and not probs[:, (0, 2)].any()
    h = hand()
    for action, expected in [(Action.raise_to(300), "vs_open_late"),
                             (Action.raise_to(1000), "vs_3bet"),
                             (Action.raise_to(3000), "vs_4bet")]:
        h.apply(action)
        assert context(h.observation(h.to_act), h.legal_actions()).startswith(expected)


def test_sb_open_context_tracks_callers_and_bb_behind():
    def sb_after_open(caller=False):
        h = hand(6)
        h.apply(Action.raise_to(300))
        while position(h.observation(h.to_act)) != "SB":
            view, legal = h.observation(h.to_act), h.legal_actions()
            action = Action.call() if caller and not any(
                    e.seat != view.seat and e.action is not None and e.action.value == "call"
                    for e in view.history
            ) else Action.fold()
            assert legal.is_legal(action)
            h.apply(action)
        return h, context(h.observation(h.to_act), h.legal_actions())

    _, single = sb_after_open()
    h, multi = sb_after_open(caller=True)
    assert single.startswith("sb_vs_open_early_medium:")
    assert multi.startswith("sb_vs_open_multi_early_medium:")
    table = tables()["tables"][f"6/SB/{single}"]
    assert "BB remains unacted" in table["decision_note"]
    assert np.mean(np.array(tables()["tables"][f"6/SB/{multi}"]["probs"])[:, 0]) > np.mean(
        np.array(table["probs"])[:, 0])

    h = hand(6)
    h.apply(Action.raise_to(300))
    h.apply(Action.raise_to(1000))
    while position(h.observation(h.to_act)) != "SB":
        h.apply(Action.fold())
    assert context(h.observation(h.to_act), h.legal_actions()).startswith("sb_vs_3bet:")


def test_bb_opener_factors_follow_relative_table_positions():
    from scripts.experiments.web_preflop_tables import generate

    data = generate()["tables"]
    for n, openers in ((6, ("LJ", "HJ", "CO", "BTN")),
                       (9, ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"))):
        continues = [
            1 - np.mean(np.array(data[f"{n}/BB/bb_vs_open_single_{opener}_medium:normal"]["probs"])[:, 0])
            for opener in openers
        ]
        assert continues == sorted(continues)


def test_bb_context_splits_single_opener_from_multiway():
    from scripts.experiments.web_preflop_tables import generate

    candidate_tables = generate()["tables"]
    def bb_after_open(caller=False, opener="LJ", target=300):
        h = hand(6)
        while position(h.observation(h.to_act)) != opener:
            h.apply(Action.fold())
        h.apply(Action.raise_to(target))
        while position(h.observation(h.to_act)) != "BB":
            view, legal = h.observation(h.to_act), h.legal_actions()
            already_called = any(e.action is not None and e.action.value == "call" for e in view.history)
            action = Action.call() if caller and not already_called else Action.fold()
            assert legal.is_legal(action)
            h.apply(action)
        view, legal = h.observation(h.to_act), h.legal_actions()
        return view, legal, context(view, legal)

    single_view, single_legal, single = bb_after_open()
    _, _, multi = bb_after_open(caller=True)
    assert single.startswith("bb_vs_open_single_LJ_medium:")
    assert multi.startswith("bb_vs_open_multi_LJ_medium:")
    assert lookup(single_view, single_legal)[2] == "6/BB/bb_vs_open_single_early_medium:normal"
    single_table = candidate_tables[f"6/BB/{single}"]
    multi_table = candidate_tables[f"6/BB/{multi}"]
    assert "calling closes preflop action" in single_table["decision_note"]
    assert np.mean(np.array(single_table["probs"])[:, 0]) < np.mean(
        np.array(multi_table["probs"])[:, 0])
    btn, small = bb_after_open(opener="BTN")[2], bb_after_open(opener="LJ", target=200)[2]
    assert btn.startswith("bb_vs_open_single_BTN_medium:")
    assert small.startswith("bb_vs_open_single_LJ_small:")
    assert np.mean(np.array(candidate_tables[f"6/BB/{btn}"]["probs"])[:, 0]) < np.mean(
        np.array(single_table["probs"])[:, 0])

    h = hand(6)
    h.apply(Action.raise_to(300))
    h.apply(Action.raise_to(1000))
    while position(h.observation(h.to_act)) != "BB":
        h.apply(Action.fold())
    assert context(h.observation(h.to_act), h.legal_actions()).startswith("bb_vs_3bet_single:")


@pytest.mark.parametrize("depth", (5, 10, 20, 100))
def test_hu_range_tracker_uses_actual_preflop_probabilities(depth):
    from backend.engine.fast_eval import CARD_A, CARD_B

    h = hand(depth=depth)
    pf = PushFoldAgent(4)
    ci = np.array([class_index(int(a), int(b)) for a, b in zip(CARD_A, CARD_B)])
    expected = {0: np.ones(1326), 2: np.ones(1326)}
    # Public actions may be off the push/fold tree; both paths must agree there too.
    for action in (Action.call(), Action.check()):
        v, legal = h.observation(h.to_act), h.legal_actions()
        expected[v.seat] *= probabilities(v, legal, pf)[ci, 1]
        h.apply(action)
    actual = ranges(h.observation(h.to_act), pf)
    for seat in expected:
        np.testing.assert_allclose(actual[seat], expected[seat])


def test_hu_solver_receives_frozen_ranges_without_old_top_x(monkeypatch):
    from backend.agents.realtime import _HandMemory, replay
    from backend.solver.subgame import board_mask

    h = hand()
    h.apply(Action.raise_to(300))
    h.apply(Action.call())
    v = h.observation(h.to_act)
    agent = _WebHeadsUp(1, mode="network", use_opponent_model=False, exploit=False)
    memory = _HandMemory(v.hole_cards)
    actions, _, oop = replay(v)
    agent._update_ranges(v, memory, actions, oop, v.button, 0)
    expected = ranges(v, agent._preflop)
    for seat in expected:
        np.testing.assert_allclose(memory.ranges[seat], expected[seat] * board_mask(v.board))
    assert "frozen web tables" in memory.notes[0]


@pytest.mark.parametrize("n", range(2, 10))
def test_web_preflop_and_multiway_are_legal_cpu_only(n):
    for seed in range(4):
        h = hand(n, seed)
        agents = {s: WebRealtimeAgent(seed + s, mode="network") for s in h.seats}
        while not h.is_over:
            v, legal = h.observation(h.to_act), h.legal_actions()
            if n == 2 and v.street is not Street.PREFLOP:
                # HU solver integration is checked separately; no GPU during M9.
                break
            decision = agents[v.seat].act(v, legal)
            assert legal.is_legal(decision.action)
            assert decision.strategy_source is StrategySource.HEURISTIC
            assert sum(decision.action_probs.values()) == pytest.approx(1)
            if v.street is Street.PREFLOP:
                assert decision.reason.startswith("frozen preflop")
            else:
                assert decision.reason.startswith("HEURISTIC multiway")
            h.apply(decision.action)


@pytest.mark.parametrize("n", range(2, 10))
def test_existing_short_stack_pushfold_is_preserved(n):
    for depth in (5, 10, 20):
        h = hand(n, depth=depth)
        v, legal = h.observation(h.to_act), h.legal_actions()
        pf = PushFoldAgent(9)
        if isinstance(pf._lookup(v), str):
            continue
        assert WebRealtimeAgent(9).act(v, legal) == pf.act(v, legal)


def test_heuristic_cheap_price_cap_and_equity_blockers():
    import random

    from backend.engine.cards import parse_cards

    h = hand(3)
    while h.street is Street.PREFLOP:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    h.apply(Action.bet(100))
    view, legal = h.observation(h.to_act), h.legal_actions()
    assert legal.call_amount / (view.pot + legal.call_amount) <= .25
    d = heuristic(view, legal)
    assert 0 <= d.action_probs["fold"] <= 1
    # A royal flush on the board is an exact split, regardless of the runout sampler.
    view = replace(view, board=tuple(parse_cards("As Ks Qs Js Ts")), hole_cards=tuple(parse_cards("2c 3d")))
    assert equity(view) == pytest.approx(1 / 3)


def test_multiway_equity_uses_public_action_conditioned_ranges():
    from backend.agents.web_realtime import _public_ranges

    h = hand(3)
    while h.street is Street.PREFLOP:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    bettor = h.to_act
    h.apply(Action.bet(100))
    view = h.observation(h.to_act)
    base = ranges(view, PushFoldAgent(0))
    conditioned = _public_ranges(view)

    assert not np.allclose(base[bettor], conditioned[bettor])
    assert equity(view, samples=12) == pytest.approx(equity(view, samples=12))



@pytest.mark.parametrize(("equity_delta", "expected_fold", "expected_call"), [
    (-.01, 1.0, 0.0),
    (.01, 0.0, 1.0),
])
def test_call_decision_follows_range_equity_pot_odds(monkeypatch, equity_delta, expected_fold, expected_call):
    h = hand(3)
    while h.street is Street.PREFLOP:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    h.apply(Action.bet(100))
    view, legal = h.observation(h.to_act), h.legal_actions()
    price = legal.call_amount / (view.pot + legal.call_amount)
    monkeypatch.setattr("backend.agents.web_realtime.equity", lambda _: price + equity_delta)

    decision = heuristic(view, legal)

    assert decision.action_probs["fold"] == expected_fold
    assert decision.action_probs["call"] == expected_call
    assert sum(decision.action_probs.values()) == pytest.approx(1.0)


def test_multiway_value_bet_threshold_avoids_thin_raises(monkeypatch):
    h = hand(4)
    while h.street is Street.PREFLOP:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    view, legal = h.observation(h.to_act), h.legal_actions()
    assert legal.can_check
    monkeypatch.setattr("backend.agents.web_realtime.equity", lambda _: .40)
    decision = heuristic(view, legal)
    assert decision.action_probs["raise"] == 0
    assert decision.action_probs["check"] == 1


def test_web_factory_is_separate_from_arena_and_has_no_gpu_initialisation():
    from backend.api.sessions import _realtime
    from backend.arena.agents import REGISTRY
    from backend.agents.realtime import RealtimeAgent

    assert isinstance(_realtime(1), WebRealtimeAgent)
    assert isinstance(REGISTRY["realtime"](1, mode="network"), RealtimeAgent)
    assert _realtime(1)._hu is None


def test_hu_fallback_source_is_labelled_as_heuristic():
    from backend.api.sessions import _strategy_label
    from backend.agents.base import Decision

    d = Decision(Action.check(), StrategySource.HEURISTIC, reason="HU HEURISTIC fallback (mode=network)")
    assert _strategy_label(d) == "HU 翻後 · HEURISTIC"


def test_rule_output_unchanged_on_160_seeded_hands():
    import hashlib
    import json

    from backend.agents.rule_based import RuleBasedAgent
    from backend.engine.cards import derive_seed

    rows = []
    for n in range(2, 10):
        for deal in range(20):
            h = HandState({s: 10000 for s in range(n)}, deal % n, 50, 100, Deck(derive_seed(872, n, deal)))
            actions = []
            while not h.is_over:
                a = RuleBasedAgent().act(h.observation(h.to_act), h.legal_actions()).action
                actions.append((h.to_act, str(a.type), a.amount))
                h.apply(a)
            rows.append((n, deal, actions))
    assert hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest() == "320996ac21789756c2e003fb428b88b004433f325a1752bfb667c06135953ad5"


def test_web_keeps_public_models_even_when_hand_ends_preflop():
    models = {}
    agent = WebRealtimeAgent(1, opponent_models=models)
    h = hand()
    h.apply(Action.fold())
    agent.hand_over(h.observation(0))
    assert agent._hu is None and models[2].hands == 1
    hu = _WebHeadsUp(1, mode="network", opponent_models=models, exploit=False, reference_model_path=None)
    assert hu._model(2) is models[2] and hu.reference_model is None


def test_web_acceptance_requires_position_coverage_and_cheap_samples():
    from scripts.experiments.web_realtime_accept import gate, rates

    counts = {"hands": 100, "vpip_hands": 45, "pfr_hands": 30,
              "pf_folds": 40, "pf_fold_opportunities": 100, "cheap_folds": 15, "cheap_opportunities": 30,
              "three_bets": 4, "three_bet_opportunities": 40,
              "bb_defends": 20, "bb_defense_opportunities": 40}
    from collections import Counter

    cell = rates(Counter(counts))
    assert cell["vpip"] == .45 and cell["three_bet"] == .1 and cell["bb_defense"] == .5
    row = {"positions": {"BB": cell}, "profit": {"bb_per_100": 10, "ci": [-1, 21]}}
    rows = {"2": {"rule": row, "station": row}}
    assert gate(rows, "arena")["passed"]
    assert not gate(rows, "cpu")["passed"]
    cell["cheap_opportunities"] = 2
    assert not gate(rows, "arena")["passed"]
    empty = rates(Counter({"hands": 10}))
    assert empty["three_bets"] == empty["three_bet_opportunities"] == 0
    assert empty["three_bet"] is None


def test_cpu_policy_simulation_captures_same_deal_samples():
    from scripts.experiments.web_realtime_accept import simulate

    result = simulate(3, "rule", 2, 73, "network", 2, log=lambda *_args, **_kwargs: None,
                     capture_samples=True)

    assert len(result["_samples"]["profit"]) == 2
    assert set(result["_samples"]["profit_by_position"]) == set(POSITIONS[3])
    assert all(len(values) == 2 for values in result["_samples"]["profit_by_position"].values())
    assert result["preflop_contexts"]["SB"]
    assert all(abs(sum(result["preflop_contexts"]["SB"][node][name]
                       for name in ("fold", "check", "call", "raise")) - 1) < 1e-9
               for node in result["preflop_contexts"]["SB"])
    assert result["profit"]["bb_per_100"] == pytest.approx(
        100 * np.mean(result["_samples"]["profit"]))


def test_simulation_deal_shards_replay_the_same_seeded_deals():
    from scripts.experiments.web_realtime_accept import simulate

    no_log = lambda *_args, **_kwargs: None
    full = simulate(3, "rule", 4, 73, "network", .01, log=no_log, capture_samples=True)
    first = simulate(3, "rule", 2, 73, "network", .01, log=no_log,
                     capture_samples=True, deal_start=0)
    second = simulate(3, "rule", 2, 73, "network", .01, log=no_log,
                      capture_samples=True, deal_start=2)

    assert full["_samples"]["profit"] == first["_samples"]["profit"] + second["_samples"]["profit"]
    for pos in POSITIONS[3]:
        assert full["_samples"]["profit_by_position"][pos] == (
            first["_samples"]["profit_by_position"][pos] +
            second["_samples"]["profit_by_position"][pos])


def test_paired_cpu_runner_uniform_and_public_range_sources():
    from scripts.experiments.web_policy_ab import PUBLIC_RANGE_SOURCE, range_source

    h = hand(3)
    view = h.observation(h.to_act)
    uniform = range_source("uniform")(view)

    assert range_source("public") is PUBLIC_RANGE_SOURCE
    assert set(uniform) == {seat.seat for seat in view.seats if not seat.folded}
    assert all(np.all(weights == 1) for weights in uniform.values())


def test_paired_cpu_delta_is_candidate_minus_baseline():
    from scripts.experiments.web_policy_ab import paired_delta

    result = paired_delta([.01, -.02, .03], [.02, .00, .04])

    assert result["bb_per_100"] == pytest.approx(100 * (.01 + .02 + .01) / 3)



def test_cheap_call_floor_control_only_moves_fold_mass_to_legal_call():
    from backend.agents.base import Decision, StrategySource
    from backend.engine.actions import ActionType
    from scripts.experiments.web_policy_ab import cheap_call_floor_control

    h = hand(3)
    while h.street is Street.PREFLOP:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    h.apply(Action.bet(100))
    view, legal = h.observation(h.to_act), h.legal_actions()
    assert legal.call_amount / (view.pot + legal.call_amount) <= .25
    original = Decision(Action.fold(), StrategySource.HEURISTIC,
                        {"fold": .7, "call": .2, "raise": .1},
                        {"fold": 0.0, "call": -2.0, "raise": -1.0},
                        "test EV policy")

    controlled = cheap_call_floor_control(lambda *_: original)(view, legal)

    assert controlled.action.type is ActionType.CALL
    assert controlled.action_probs["call"] == pytest.approx(.9)
    assert controlled.action_probs["raise"] == pytest.approx(.1)
    assert controlled.action_probs["fold"] == 0
    assert sum(controlled.action_probs.values()) == pytest.approx(1.0)
    assert controlled.action_ev == original.action_ev


def _multiway_flop():
    h = hand(3)
    while h.street is Street.PREFLOP:
        legal = h.legal_actions()
        h.apply(Action.check() if legal.can_check else Action.call())
    view = h.observation(h.to_act)
    legal = h.legal_actions()
    assert legal.can_check and legal.can_raise
    return view, legal


def test_multiway_bluff_uses_range_blockers_and_pot_odds(monkeypatch):
    from backend.engine.fast_eval import CARD_A, CARD_B
    from backend.agents import web_realtime

    view, legal = _multiway_flop()
    blocked_cards = set(view.hole_cards)
    strength = np.where(np.isin(CARD_A, tuple(blocked_cards)) |
                        np.isin(CARD_B, tuple(blocked_cards)), .99, 0.0)
    monkeypatch.setattr(web_realtime, "_board_strength", lambda _: strength)
    monkeypatch.setattr(
        web_realtime, "_public_ranges",
        lambda v: {s.seat: np.ones(1326) for s in v.seats},
    )

    candidate = web_realtime._multiway_bluff(view, legal, .05)

    assert candidate is not None
    assert candidate["blocker_delta"] > 0
    assert candidate["fold_equity"] > 0
    assert candidate["ev"] > 0
    assert 0 < candidate["mix"] <= 1

    monkeypatch.setattr(web_realtime, "_board_strength", lambda _: np.ones(1326))
    unprofitable = web_realtime._multiway_bluff(view, legal, .05)
    assert unprofitable is not None and unprofitable["mix"] == 0


def test_multiway_bluff_is_sampled_legally_with_reproducible_mix(monkeypatch):
    import random

    from backend.agents import web_realtime

    view, legal = _multiway_flop()
    stats = {"ev": 12.0, "check_ev": 5.0, "mix": .4, "fold_equity": .7, "blocker_delta": .1, "bet": 70.0}
    monkeypatch.setattr(web_realtime, "equity", lambda _: .10)
    monkeypatch.setattr(web_realtime, "_multiway_bluff", lambda *args: stats)

    first = heuristic(view, legal, random.Random(98), allow_bluff=True)
    second = heuristic(view, legal, random.Random(98), allow_bluff=True)
    assert first == second
    assert legal.is_legal(first.action)
    assert first.action_probs["raise"] == pytest.approx(.4)
    assert first.action_probs["check"] == pytest.approx(.6)
    assert sum(first.action_probs.values()) == pytest.approx(1)
    assert first.action_ev == {"check": 5.0, "raise": 12.0}
    assert "HEURISTIC multiway" in first.reason and "estimated bluff mix" in first.reason

    disabled = heuristic(view, legal, random.Random(98))
    assert disabled.action_probs["raise"] == 0
    assert disabled.action == Action.check()

def test_bluff_ev_report_converts_chips_to_big_blinds(monkeypatch):
    from scripts.experiments import web_realtime_accept as experiment

    original = experiment.WebRealtimeAgent.act

    def with_estimate(self, view, legal):
        decision = original(self, view, legal)
        if view.street is not Street.PREFLOP and legal.can_check:
            return replace(decision, action_ev={"check": 50.0, "raise": 250.0}, reason="estimated bluff mix")
        return decision

    monkeypatch.setattr(experiment.WebRealtimeAgent, "act", with_estimate)
    result = experiment.simulate(3, "rule", 2, 20261101, "network", .01, log=lambda *_a, **_k: None)

    assert result["bluff"]["opportunities"] > 0
    assert result["bluff"]["estimated_raise_ev_bb_per_opportunity"] == pytest.approx(2.5)
    assert result["bluff"]["estimated_check_ev_bb_per_opportunity"] == pytest.approx(.5)
    assert result["bluff"]["estimated_raise_advantage_bb_per_opportunity"] == pytest.approx(2.0)
