"""Opponent modeling (Milestone 8): statistics, posteriors, style agents, storage."""

from __future__ import annotations

import collections
import json
import random

import pytest

from backend.agents import RandomAgent, RuleBasedAgent
from backend.agents.style import STYLES, StyleAgent
from backend.engine.actions import Action
from backend.engine.cards import Deck
from backend.engine.hand import HandState
from backend.engine.records import hand_record
from backend.opponent.model import OpponentModel, Prior, beta_ppf, fit_prior
from backend.opponent.stats import CONTEXTS, GROUPS, Observation, PublicHand, observations, positions


def _hu(actions, stacks=(10000, 10000)):
    h = HandState({0: stacks[0], 1: stacks[1]}, 0, 50, 100, Deck(3))
    for a in actions:
        h.apply(a)
    return h


def _obs(h):
    return {(o.seat, o.group, o.outcome, o.position) for o in observations(PublicHand.from_state(h))}


def test_positions():
    assert positions([3, 5], 5) == {5: "BTN", 3: "BB"}
    assert positions([0, 1, 2], 1) == {1: "BTN", 2: "SB", 0: "BB"}
    six = positions(range(6), 0)
    assert [six[s] for s in range(6)] == ["BTN", "SB", "BB", "LJ", "HJ", "CO"]
    nine = positions(range(9), 8)
    assert nine[8] == "BTN" and nine[2] == "UTG" and nine[7] == "CO" and len(set(nine.values())) == 9


def test_preflop_three_bet_hand():
    h = _hu([Action.raise_to(250), Action.raise_to(750), Action.fold()])
    got = _obs(h)
    assert (0, "pf_open", "raise", "BTN") in got
    assert (1, "pf_vs_raise", "raise", "BB") in got and (1, "three_bet", "yes", "BB") in got
    assert (0, "pf_vs_3bet", "fold", "BTN") in got and (0, "fold_to_three_bet", "yes", "BTN") in got
    for seat, pos in ((0, "BTN"), (1, "BB")):
        assert (seat, "vpip", "yes", pos) in got and (seat, "pfr", "yes", pos) in got
    assert not any(g in ("wtsd", "wsd", "cbet_flop") for _, g, _, _ in got)


def test_cbet_and_fold_to_cbet():
    h = _hu([Action.raise_to(250), Action.call(), Action.check(), Action.bet(300), Action.fold()])
    got = _obs(h)
    assert (1, "three_bet", "no", "BB") in got and (1, "pf_vs_raise", "call", "BB") in got
    assert (1, "flop_unbet", "check", "BB") in got
    assert (0, "flop_unbet", "bet", "BTN") in got and (0, "cbet_flop", "yes", "BTN") in got
    assert (1, "flop_vs_bet", "fold", "BB") in got and (1, "fold_to_cbet_flop", "yes", "BB") in got
    assert (0, "wtsd", "no", "BTN") in got and (1, "wtsd", "no", "BB") in got


def test_context_only_counted_when_every_outcome_is_legal():
    h = _hu([Action.raise_to(10000), Action.call()])  # BB cannot re-raise an all-in
    got = _obs(h)
    assert not any(g == "pf_vs_raise" for s, g, _, _ in got if s == 1)
    assert (1, "three_bet", "no", "BB") in got
    assert {(s, o) for s, g, o, _ in got if g == "wtsd"} == {(0, "yes"), (1, "yes")}
    assert sum(1 for s, g, o, _ in got if g == "wsd" and o == "yes") >= 1


def test_public_hand_from_view_record_and_state_agree():
    rng = random.Random(4)
    for hn in range(60):
        n = 2 + hn % 5
        stacks = {s: rng.randint(500, 20000) for s in range(n)}
        h = HandState(stacks, hn % n, 50, 100, Deck(hn))
        agents = {s: RandomAgent(hn * 10 + s, all_in_weight=0.1) for s in range(n)}
        while not h.is_over:
            s = h.to_act
            h.apply(agents[s].act(h.observation(s), h.legal_actions()).action)
        a = PublicHand.from_state(h)
        assert dict(a.start_stacks) == stacks
        assert PublicHand.from_view(h.observation(hn % n)) == a
        assert PublicHand.from_record(json.loads(json.dumps(hand_record(h)))) == a


def test_style_agent_contexts_match_extracted_statistics():
    """The agent decides by the same contexts the statistics extract."""
    seen, extracted = collections.Counter(), collections.Counter()
    for hn in range(300):
        h = HandState({0: 10000, 1: 10000}, hn % 2, 50, 100, Deck(hn))
        agents = {0: StyleAgent(hn, "lag"), 1: RuleBasedAgent(hn)}
        while not h.is_over:
            s = h.to_act
            d = agents[s].act(h.observation(s), h.legal_actions())
            if s == 0:
                group = d.reason.split("context ")[1].split(" ")[0]
                if set(d.action_probs) == set(CONTEXTS[group]):
                    seen[group] += 1
            h.apply(d.action)
        for o in observations(PublicHand.from_state(h)):
            if o.seat == 0 and o.group in CONTEXTS:
                extracted[o.group] += 1
    assert seen == extracted and sum(seen.values()) > 200


def test_posterior_matches_the_dirichlet_formula():
    m = OpponentModel()
    for outcome, n in (("fold", 30), ("call", 15), ("raise", 5)):
        for _ in range(n):
            m.add(Observation(0, "flop_vs_bet", outcome, "BTN"))
    e = m.estimate("flop_vs_bet")
    assert e.opportunities == 50
    assert e.mean["fold"] == pytest.approx(31 / 53) and e.alpha == (31.0, 16.0, 6.0)
    scipy = pytest.importorskip("scipy.stats")
    assert e.low["raise"] == pytest.approx(scipy.beta.ppf(0.05, 6, 47), abs=1e-9)
    assert e.high["raise"] == pytest.approx(scipy.beta.ppf(0.95, 6, 47), abs=1e-9)
    assert beta_ppf(0.5, 2, 2) == pytest.approx(0.5)
    # Position level: shrunk toward the other positions (no data there: the prior, weight 3).
    p = m.estimate("flop_vs_bet", "BTN")
    assert p.alpha == pytest.approx((31.0, 16.0, 6.0))
    bb = m.estimate("flop_vs_bet", "BB")
    assert bb.opportunities == 0 and bb.mean["fold"] == pytest.approx(31 / 53)
    assert sum(bb.alpha) == pytest.approx(20.0)  # position_strength caps the borrowed weight


def test_model_round_trips():
    m = OpponentModel()
    for hn in range(200):
        h = HandState({0: 10000, 1: 10000}, hn % 2, 50, 100, Deck(hn))
        agents = {0: StyleAgent(hn, "tag"), 1: RandomAgent(hn)}
        while not h.is_over:
            s = h.to_act
            h.apply(agents[s].act(h.observation(s), h.legal_actions()).action)
        m.observe(PublicHand.from_state(h), 0)
    again = OpponentModel.from_json(json.loads(json.dumps(m.to_json())))
    assert again.counts == m.counts and again.hands == 200
    rows = m.statistic_rows()
    rebuilt = OpponentModel.from_statistic_rows(rows, hands=200)
    assert rebuilt.counts == m.counts
    for r in rows:
        assert r["n_hits"] <= r["n_opportunities"]


def test_posterior_converges_to_a_style_agents_frequencies():
    m = OpponentModel()
    for hn in range(3000):
        h = HandState({0: 10000, 1: 10000}, hn % 2, 50, 100, Deck(hn))
        agents = {0: StyleAgent(hn, "tag"), 1: RuleBasedAgent(hn)}
        while not h.is_over:
            s = h.to_act
            h.apply(agents[s].act(h.observation(s), h.legal_actions()).action)
        m.observe(PublicHand.from_state(h), 0)
    for g in ("pf_open", "pf_option", "pf_vs_raise", "flop_unbet", "turn_unbet"):
        e = m.estimate(g)
        assert e.opportunities > 100
        for o, p in zip(CONTEXTS[g], STYLES["tag"][g]):
            assert abs(e.mean[o] - p) < 4 * e.sd(o), (g, o, e.mean[o], p)


def test_fitted_population_prior_recovers_the_spread_of_players():
    rng = random.Random(1)
    models = []
    for _ in range(200):
        p = rng.betavariate(6, 14)  # population: mean 0.3, strength 20
        m = OpponentModel()
        m.counts["vpip"] = {"BTN": [0, 0]}
        for _ in range(60):
            m.counts["vpip"]["BTN"][0 if rng.random() < p else 1] += 1
        models.append(m)
    prior = fit_prior(models, "test")
    a, b = prior.alpha["vpip"]
    assert a / (a + b) == pytest.approx(0.3, abs=0.03)
    assert 10 < a + b < 40
    assert prior.alpha["flop_vs_bet"] == Prior.weak().alpha["flop_vs_bet"]  # no data: weak prior kept


def test_statistics_are_stored_and_rebuilt(tmp_path):
    pytest.importorskip("sqlalchemy")
    from backend.db import OpponentModelRow, Player, PlayerStatistic, make_engine, make_sessionmaker, upgrade
    from backend.db.repository import opponent_model, record_statistics

    url = f"sqlite:///{tmp_path / 'm8.db'}"
    upgrade(url)
    with make_sessionmaker(make_engine(url))() as db:
        players = [Player(name=f"p{i}", kind="ai") for i in range(2)]
        db.add_all(players)
        db.commit()
        ids = {0: players[0].id, 1: players[1].id}
        mem = {0: OpponentModel(), 1: OpponentModel()}
        for hn in range(40):
            h = HandState({0: 10000, 1: 10000}, hn % 2, 50, 100, Deck(hn))
            agents = {0: StyleAgent(hn, "lag"), 1: RuleBasedAgent(hn)}
            while not h.is_over:
                s = h.to_act
                h.apply(agents[s].act(h.observation(s), h.legal_actions()).action)
            record_statistics(db, hand_record(h, hand_no=hn, player_ids=ids))
            for s in (0, 1):
                mem[s].observe(PublicHand.from_state(h), s)
        for s in (0, 1):
            m = opponent_model(db, ids[s])
            assert m.counts == mem[s].counts and m.hands == 40
            row = db.query(OpponentModelRow).filter_by(player_id=ids[s]).one()
            assert row.sample_size == 40 and "vpip" in row.posterior["estimates"]
        vpip = db.query(PlayerStatistic).filter_by(player_id=ids[0], stat="vpip:yes").all()
        assert sum(r.n_opportunities for r in vpip) == sum(mem[0].total("vpip")) > 15
    assert set(GROUPS) >= set(CONTEXTS)


def test_style_agents_are_in_the_arena_registry():
    from backend.arena.agents import REGISTRY

    for name in STYLES:
        agent = REGISTRY[f"style-{name}"](3)
        assert isinstance(agent, StyleAgent) and agent.style == name
    with pytest.raises(ValueError):
        StyleAgent(0, "shark")


# ----------------------------------------------------------------------------- model -> range tracking


def test_tilt_matches_target_frequencies_and_keeps_the_equilibrium_shape():
    np = pytest.importorskip("numpy")
    from backend.opponent.ranges import tilt

    rng = np.random.default_rng(0)
    sigma = rng.dirichlet([1, 1, 1, 1], size=500).T  # actions f, c, r50, r100
    reach = rng.random(500)
    classes = ["fold", "call", "raise", "raise"]
    target = {"fold": 0.1, "call": 0.7, "raise": 0.2}
    p = tilt(sigma, reach, classes, target)
    mass = reach / reach.sum()
    assert p.sum(axis=0) == pytest.approx(np.ones(500))
    assert p[0] @ mass == pytest.approx(0.1, abs=1e-6) and p[1] @ mass == pytest.approx(0.7, abs=1e-6)
    assert (p[2] + p[3]) @ mass == pytest.approx(0.2, abs=1e-6)
    # Within a class the sizes keep their ratio, and every combo is scaled by the same class weights.
    assert np.allclose(p[2] / p[3], sigma[2] / sigma[3])
    w = p / sigma / (p / sigma)[1]
    assert np.allclose(w[0], w[0][0]) and np.allclose(w[2], w[2][0])


def test_target_frequencies_shrink_toward_the_baseline():
    from backend.opponent.ranges import target_frequencies

    base = {"fold": 0.5, "call": 0.3, "raise": 0.2}
    assert target_frequencies([0, 0, 0], ("fold", "call", "raise"), base, 20) == pytest.approx(base)
    t = target_frequencies([10, 80, 10], ("fold", "call", "raise"), base, 20)
    assert t["call"] == pytest.approx((80 + 6) / 120)
    # Only the outcomes available here (no raise when facing an all-in).
    t = target_frequencies([10, 80, 10], ("fold", "call", "raise"), {"fold": 0.6, "call": 0.4}, 20)
    assert set(t) == {"fold", "call"} and sum(t.values()) == pytest.approx(1)


def test_modelled_preflop_range_without_observations_is_the_heuristic():
    np = pytest.importorskip("numpy")
    from backend.agents.realtime import modelled_preflop_range, preflop_context, preflop_range

    h = _hu([Action.raise_to(250), Action.call()])
    v = h.observation(0)
    assert preflop_context(v, 0) == ("pf_open", "raise") and preflop_context(v, 1) == ("pf_vs_raise", "call")
    for seat in (0, 1):
        r, _ = modelled_preflop_range(v, seat, OpponentModel(), 20)
        assert np.array_equal(r, preflop_range(v, seat)[0])
    # A station that calls raises 90% of the time gets a much wider calling range.
    m = OpponentModel()
    rng = random.Random(0)
    for _ in range(200):
        m.add(Observation(1, "pf_vs_raise", "call" if rng.random() < 0.9 else "fold", "BB"))
    wide, note = modelled_preflop_range(v, 1, m, 20)
    assert wide.sum() > 2 * preflop_range(v, 1)[0].sum() and "opponent model" in note


def test_agent_hand_over_builds_opponent_models():
    from backend.agents.realtime import RealtimeAgent

    shared = {}
    for hn in range(5):
        h = HandState({0: 10000, 1: 10000}, hn % 2, 50, 100, Deck(hn))
        agents = {0: StyleAgent(hn, "station"), 1: StyleAgent(hn, "maniac")}
        while not h.is_over:
            s = h.to_act
            h.apply(agents[s].act(h.observation(s), h.legal_actions()).action)
        RealtimeAgent(hn, mode="network", opponent_models=shared).hand_over(h.observation(1))
    assert set(shared) == {0} and shared[0].hands == 5
