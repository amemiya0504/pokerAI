"""Web preflop v8.1 (runtime) and v8: frozen schema 2, depth buckets, vs 3-bet/4-bet/5-bet/all-in contexts, all-in pricing.

v8.2 (failed its gate, not runtime) adds the capped-range defence nodes; its tests are at the end of this module.
"""

import numpy as np
import pytest

from backend.agents import web_preflop, web_realtime
from backend.agents.base import StrategySource
from backend.agents.web_preflop import POSITIONS, lookup, position, tables
from backend.agents.web_preflop_v8 import (BASE_GROUPS, BEHIND_MARGIN, CALL_SMOOTHING, DEPTHS, context,
                                          contexts_for, depth_bucket, expected_keys)
from backend.agents.web_realtime import WebRealtimeAgent
from backend.engine.actions import Action
from backend.engine.cards import Deck
from backend.engine.hand import HandState, Street
from backend.solver.hands import CLASS_NAMES, COMBOS


DEFAULT_PATH = web_preflop.TABLE_PATH
V81_PATH = web_preflop.TABLE_PATH.with_name("web_preflop_v8_1.json.gz")
V82_PATH = web_preflop.TABLE_PATH.with_name("web_preflop_v8_2.json.gz")
VERSION = "web-v8.1-depth-allin"


@pytest.fixture(autouse=True)
def v81_tables(monkeypatch):
    monkeypatch.setattr(web_preflop, "TABLE_PATH", V81_PATH)
    tables.cache_clear()
    yield
    tables.cache_clear()


def table(n=6, depth=100, seed=1):
    seats = (0, 2, 4, 6, 8, 1, 3, 5, 7)[:n]
    return HandState({s: depth * 100 for s in seats}, 0, 50, 100, Deck(seed))


def fold_to(h, pos):
    while position(h.observation(h.to_act)) != pos:
        h.apply(Action.fold())


def ctx(h):
    return context(h.observation(h.to_act), h.legal_actions())


def test_frozen_v8_and_v81_match_generator_and_cover_every_key():
    import gzip
    import json

    from scripts.experiments.web_preflop_v8_tables import VARIANTS, generate, load_equity

    data = tables()
    assert web_preflop.TABLE_PATH == V81_PATH
    assert data["schema"] == 2 and data["version"] == VERSION
    assert "not GTO" in data["source"]
    assert set(data["tables"]) == expected_keys()
    assert len(data["tables"]) == sum(len(contexts_for(n, p)) for n, ps in POSITIONS.items() for p in ps) * len(DEPTHS)
    E, W = load_equity()
    assert json.loads(json.dumps(generate(E, W))) == data
    v8 = json.loads(gzip.decompress(VARIANTS["v8"]["path"].read_bytes()))
    assert json.loads(json.dumps(generate(E, W, "v8"))) == v8 and v8["version"] == "web-v8-depth-allin"
    v82 = json.loads(gzip.decompress(VARIANTS["v8.2"]["path"].read_bytes()))
    assert json.loads(json.dumps(generate(E, W, "v8.2"))) == v82 and v82["version"] == "web-v8.2-capped-defence"


def test_v81_unopened_and_vs_open_match_100bb_at_every_depth():
    """v8.1 keeps v7's open sizes and ranges at every depth; only 3-bet+ and all-in nodes vary by depth."""
    from backend.agents.web_preflop import BB_A3_GROUPS, SB_GROUPS

    t = tables()["tables"]
    same = (*BASE_GROUPS, *(g for g in (*SB_GROUPS, *BB_A3_GROUPS) if "3bet" not in g))
    for n, ps in POSITIONS.items():
        for pos in ps:
            for ctx in contexts_for(n, pos):
                group = ctx.split(":")[0]
                for depth in DEPTHS:
                    k, ref = f"{n}/{depth}/{pos}/{ctx}", t[f"{n}/100/{pos}/{ctx}"]
                    if group in same:
                        assert t[k] == ref, k
    assert t["6/40/LJ/vs_3bet_late_small:normal"] != t["6/100/LJ/vs_3bet_late_small:normal"]


def test_v8_rows_are_valid_and_rank_strong_hands_first():
    aa, trash = CLASS_NAMES.index("AA"), CLASS_NAMES.index("72o")
    for key, t in tables()["tables"].items():
        if key.endswith(":allin"):
            eq = np.array(t["equity"])
            assert eq[aa] > .8 and eq[trash] < .35, key
            continue
        p = np.array(t["probs"])
        assert np.allclose(p.sum(axis=1), 1), key
        assert 1 - p[aa, 0] >= 1 - p[trash, 0], key
        assert (p[:, 2] <= 1 - p[:, 0] + 1e-9).all(), key


@pytest.mark.parametrize("bb, bucket", [(20, 20), (29.9, 20), (30, 40), (49.9, 40), (50, 60), (79.9, 60),
                                         (80, 100), (124.9, 100), (125, 150), (400, 150)])
def test_depth_buckets(bb, bucket):
    assert depth_bucket(bb) == bucket


def test_lookup_key_uses_effective_stack_bucket():
    for depth, bucket in ((25, 20), (40, 40), (60, 60), (100, 100), (200, 150)):
        h = table(6, depth)
        assert lookup(h.observation(h.to_act), h.legal_actions())[2] == f"6/{bucket}/LJ/open:normal"


def test_opener_facing_3bet_splits_3bettor_side_and_size():
    h = table(6)
    h.apply(Action.raise_to(300))       # LJ opens
    fold_to(h, "CO")
    h.apply(Action.raise_to(900))       # CO 3-bets 3x: small
    fold_to(h, "LJ")
    assert ctx(h).startswith("vs_3bet_late_small:")

    h = table(6)
    h.apply(Action.raise_to(300))
    fold_to(h, "SB")
    h.apply(Action.raise_to(1200))      # SB 3-bets 4x: large
    fold_to(h, "LJ")
    assert ctx(h).startswith("vs_3bet_late_large:")

    h = table(9)
    h.apply(Action.raise_to(300))       # UTG opens
    h.apply(Action.raise_to(900))       # UTG+1 3-bets
    fold_to(h, "UTG")
    assert ctx(h).startswith("vs_3bet_early_small:")


def test_cold_3bet_4bet_and_5bet_nodes():
    h = table(6)
    h.apply(Action.raise_to(300))       # LJ
    h.apply(Action.raise_to(900))       # HJ
    assert ctx(h).startswith("cold_vs_3bet:")   # CO has not acted

    h = table(6)
    h.apply(Action.raise_to(300))       # LJ opens
    fold_to(h, "BTN")
    h.apply(Action.raise_to(900))       # BTN 3-bets
    fold_to(h, "LJ")
    h.apply(Action.raise_to(2000))      # LJ 4-bets
    assert ctx(h).startswith("vs_4bet_early:")  # back to BTN, the 3-bettor
    h.apply(Action.raise_to(4500))      # BTN 5-bets, not all-in
    assert ctx(h).startswith("vs_5bet:")

    h = table(6)
    h.apply(Action.raise_to(300))       # LJ opens
    h.apply(Action.raise_to(900))       # HJ 3-bets
    h.apply(Action.raise_to(2000))      # CO cold 4-bets
    fold_to(h, "LJ")
    assert ctx(h).startswith("vs_4bet_cold:")


def test_allin_states():
    h = table(6)
    h.apply(Action.raise_to(10000))     # LJ open-jams
    assert ctx(h) == "vs_allin_early_unopened:allin"

    h = table(6)
    h.apply(Action.raise_to(300))
    fold_to(h, "BTN")
    h.apply(Action.raise_to(10000))     # BTN jams over the open
    fold_to(h, "LJ")
    assert ctx(h) == "vs_allin_late_after_open:allin"

    h = table(6)
    h.apply(Action.raise_to(300))
    fold_to(h, "BTN")
    h.apply(Action.raise_to(900))
    fold_to(h, "LJ")
    h.apply(Action.raise_to(10000))     # LJ 4-bet jams
    assert ctx(h) == "vs_allin_early_after_3bet:allin"

    h = table(6)
    h.apply(Action.raise_to(300))
    h.apply(Action.raise_to(10000))     # HJ jams over LJ's open
    assert ctx(h) == "vs_allin_early_cold:allin"


def test_short_jammer_counts_as_allin_even_when_hero_covers():
    seats = (0, 2, 4, 6, 8, 1)
    lj = HandState({s: 10000 for s in seats}, 0, 50, 100, Deck(1)).to_act
    stacks = {s: (2500 if s == lj else 10000) for s in seats}
    h = HandState(stacks, 0, 50, 100, Deck(1))
    h.apply(Action.raise_to(2500))      # short LJ jams; HJ covers
    v, legal = h.observation(h.to_act), h.legal_actions()
    assert legal.call_amount < next(s.stack for s in v.seats if s.seat == v.seat)
    assert context(v, legal) == "vs_allin_early_unopened:allin"


def test_allin_call_uses_actual_price_and_players_behind():
    h = table(6)
    h.apply(Action.raise_to(10000))     # LJ jams 100 BB
    fold_to(h, "BB")
    v, legal = h.observation(h.to_act), h.legal_actions()
    probs, t, key = lookup(v, legal)
    assert key == "6/100/BB/vs_allin_early_unopened:allin"
    price = legal.call_amount / (v.pot + legal.call_amount)
    expected = 1 / (1 + np.exp(-np.clip((np.array(t["equity"]) - price) / CALL_SMOOTHING, -60, 60)))
    np.testing.assert_allclose(probs[:, 1] + probs[:, 2], expected)
    assert probs[CLASS_NAMES.index("AA"), 1] > .99 and probs[CLASS_NAMES.index("72o"), 0] > .99
    assert not probs[:, 2].any()

    h = table(6)
    h.apply(Action.raise_to(10000))
    v, legal = h.observation(h.to_act), h.legal_actions()   # HJ, four players behind
    probs, t, _ = lookup(v, legal)
    need = legal.call_amount / (v.pot + legal.call_amount) + 4 * BEHIND_MARGIN
    expected = 1 / (1 + np.exp(-np.clip((np.array(t["equity"]) - need) / CALL_SMOOTHING, -60, 60)))
    np.testing.assert_allclose(probs[:, 1], expected)


def test_v8_large_raise_becomes_a_jam(monkeypatch):
    def facing_3bet(depth):
        h = table(6, depth)
        h.apply(Action.raise_to(220))
        fold_to(h, "BTN")
        h.apply(Action.raise_to(700))
        fold_to(h, "LJ")
        return h.observation(h.to_act), h.legal_actions()

    raise_all = np.zeros((169, 3))
    raise_all[:, 2] = 1
    sizing = {"raise_bb": 2.2, "raise_multiple": 2.3, "jam_fraction": .40}
    monkeypatch.setattr(web_realtime, "lookup", lambda view, legal: (raise_all, sizing, "test"))
    v, legal = facing_3bet(40)          # 4-bet to 1610 >= 40% of 4000 -> jam
    assert WebRealtimeAgent(0).act(v, legal).action.amount == legal.max_raise_to
    v, legal = facing_3bet(100)         # 1610 < 40% of 10000 -> normal 4-bet
    assert WebRealtimeAgent(0).act(v, legal).action.amount == 1610


@pytest.mark.parametrize("n", range(2, 10))
def test_v8_preflop_is_legal_at_every_depth(n):
    for depth in (25, 40, 60, 100, 150):
        for seed in range(3):
            h = table(n, depth, seed)
            agents = {s: WebRealtimeAgent(seed + s, mode="network") for s in h.seats}
            while not h.is_over and h.street is Street.PREFLOP:
                v, legal = h.observation(h.to_act), h.legal_actions()
                decision = agents[v.seat].act(v, legal)
                assert legal.is_legal(decision.action)
                assert decision.strategy_source is StrategySource.HEURISTIC
                assert sum(decision.action_probs.values()) == pytest.approx(1)
                assert decision.reason.startswith(f"frozen preflop {VERSION}")
                h.apply(decision.action)


def test_v8_open_ranges_are_combo_weighted_like_v7_at_100bb():
    w = np.array([len(c) for c in COMBOS]) / 1326
    v7 = web_preflop.json.loads(web_preflop.TABLE_PATH.with_name("web_preflop_v7.json").read_text())["tables"]
    v8 = tables()["tables"]
    for n, ps in POSITIONS.items():
        for pos in ps:
            a = np.array(v7[f"{n}/{pos}/open:normal"]["probs"])
            b = np.array(v8[f"{n}/100/{pos}/open:normal"]["probs"])
            assert w @ b[:, 2] == pytest.approx(w @ a[:, 2], abs=.002)


# ---------------------------------------------------------------- v8.2 capped-range defence (failed gate)

@pytest.fixture
def v82(monkeypatch):
    monkeypatch.setattr(web_preflop, "TABLE_PATH", V82_PATH)
    tables.cache_clear()
    return tables()


def hu(depth=100):
    return HandState({0: depth * 100, 1: depth * 100}, 0, 50, 100, Deck(1))


def flat_then_squeeze(depth=100):
    h = table(6, depth)
    fold_to(h, "CO")
    h.apply(Action.raise_to(250))       # CO opens
    h.apply(Action.call())              # BTN flats
    h.apply(Action.raise_to(1000))      # SB squeezes
    h.apply(Action.fold())              # BB
    h.apply(Action.fold())              # CO
    return h


def test_v82_only_adds_defence_nodes_and_runtime_stays_v81(v82):
    from backend.agents.web_preflop_v8 import DEFEND_GROUPS

    assert DEFAULT_PATH == V81_PATH  # v8.2 failed its 2026-10-26 gate
    assert v82["version"] == "web-v8.2-capped-defence" and "not GTO" in v82["source"]
    assert v82["defend_groups"] == list(DEFEND_GROUPS)
    assert set(v82["tables"]) == expected_keys(defend=True)
    v81 = web_preflop.json.loads(__import__("gzip").decompress(V81_PATH.read_bytes()))["tables"]
    assert {k: t for k, t in v82["tables"].items() if not k.endswith(":mdf")} == v81


def test_limper_facing_raise_and_flatter_facing_squeeze_contexts(v82):
    h = hu()
    h.apply(Action.call())              # BTN/SB limps
    h.apply(Action.raise_to(400))       # BB isolates
    v, legal = h.observation(h.to_act), h.legal_actions()
    assert context(v, legal, defend=True) == "limp_vs_raise:mdf"
    assert context(v, legal).startswith("vs_open_early_medium:")   # v8.1 tables keep the old node
    assert lookup(v, legal)[2] == "2/100/BTN/SB/limp_vs_raise:mdf"

    h = flat_then_squeeze()
    v, legal = h.observation(h.to_act), h.legal_actions()
    assert context(v, legal, defend=True) == "flat_vs_squeeze:mdf"
    assert context(v, legal) == "cold_vs_3bet:normal"

    h = table(6)
    fold_to(h, "CO")
    h.apply(Action.call())              # CO limps
    h.apply(Action.raise_to(400))       # BTN isolates
    h.apply(Action.raise_to(1200))      # SB 3-bets: the limper is cold, not a flatter
    h.apply(Action.fold())
    assert context(h.observation(h.to_act), h.legal_actions(), defend=True) == "cold_vs_3bet:normal"

    h = hu()
    h.apply(Action.call())
    h.apply(Action.raise_to(10000))     # BB jams over the limp: the all-in node wins
    assert context(h.observation(h.to_act), h.legal_actions(), defend=True) == "vs_allin_late_unopened:allin"


def test_defence_continues_mdf_share_of_the_capped_range(v82):
    from backend.agents.web_preflop_v8 import DEFEND_SMOOTHING

    h = hu()
    h.apply(Action.call())
    h.apply(Action.raise_to(400))
    v, legal = h.observation(h.to_act), h.legal_actions()
    probs, t, _ = lookup(v, legal)
    fraction = t["mdf_scale"] * (1 - legal.call_amount / v.pot)
    assert fraction == pytest.approx(.75 * .4)
    cont = 1 / (1 + np.exp(-np.clip((fraction - np.array(t["percentile"])) / DEFEND_SMOOTHING, -60, 60)))
    np.testing.assert_allclose(1 - probs[:, 0], cont, atol=1e-12)
    limp = np.array([row[1] for row in v82["tables"]["2/100/BTN/SB/open:normal"]["probs"]])
    w = np.array([len(c) for c in COMBOS]) * limp
    assert (w @ cont) / w.sum() == pytest.approx(fraction, abs=.03)   # v8.1 continued with none of it

    v, legal = (h := flat_then_squeeze()).observation(h.to_act), h.legal_actions()
    probs, _, _ = lookup(v, legal)
    aa, sevens, trash = (CLASS_NAMES.index(c) for c in ("AA", "77", "72o"))
    assert probs[aa, 2] > .9 and probs[sevens, 1] > .5 and probs[trash, 0] > .99


def test_preflop_probes_attack_capped_ranges():
    from scripts.experiments.preflop_exploiters import IsoProbe, SqueezeProbe

    h = hu()
    h.apply(Action.call())
    assert IsoProbe(0).act(h.observation(h.to_act), h.legal_actions()).action.amount == 400

    h = table(6)
    fold_to(h, "CO")
    h.apply(Action.raise_to(250))
    h.apply(Action.call())
    assert SqueezeProbe(0).act(h.observation(h.to_act), h.legal_actions()).action.amount == 1000

    h = table(6)                        # nothing to attack: plain rule-based play
    v, legal = h.observation(h.to_act), h.legal_actions()
    assert "probe" not in (SqueezeProbe(0).act(v, legal).reason or "")
    assert "probe" not in (IsoProbe(0).act(v, legal).reason or "")


@pytest.mark.parametrize("n", (2, 4, 6, 9))
def test_v82_preflop_is_legal_against_probes(v82, n):
    from scripts.experiments.preflop_exploiters import IsoProbe, SqueezeProbe

    hit = set()
    for depth in (25, 100):
        for seed in range(12):
            for probe in (IsoProbe, SqueezeProbe):
                h = table(n, depth, seed)
                hero = h.seats[seed % n]
                agents = {s: (WebRealtimeAgent(seed, mode="network") if s == hero else probe(seed + s)) for s in h.seats}
                while not h.is_over and h.street is Street.PREFLOP:
                    v, legal = h.observation(h.to_act), h.legal_actions()
                    decision = agents[v.seat].act(v, legal)
                    assert legal.is_legal(decision.action)
                    if v.seat == hero and decision.reason.startswith("frozen preflop"):
                        assert decision.reason.startswith("frozen preflop web-v8.2-capped-defence")
                        hit |= {g for g in ("limp_vs_raise", "flat_vs_squeeze") if g in decision.reason}
                    h.apply(decision.action)
    assert hit, "probes should reach at least one defence node"
