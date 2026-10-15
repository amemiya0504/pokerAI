"""Arena: determinism, duplicate correctness, replayable output, statistics, storage."""

from __future__ import annotations

import json
import random
from statistics import NormalDist

import pytest

from backend.arena import AgentSpec, ArenaConfig, run_arena
from backend.arena.stats import summarize, t_quantile
from backend.engine.actions import Action, ActionType
from backend.engine.cards import Deck
from backend.engine.hand import HandState


def specs(*kinds: str) -> tuple[AgentSpec, ...]:
    return tuple(AgentSpec(f"{k}{i}", k) for i, k in enumerate(kinds))


MIXED = specs("rule", "random", "station", "random")


def test_same_seed_same_results_regardless_of_workers():
    cfg = ArenaConfig(MIXED, deals=60, seed=11)
    one = run_arena(cfg, workers=1)
    many = run_arena(cfg, workers=3, chunk=7)
    assert one.hand_rows == many.hand_rows
    assert one.player_rows == many.player_rows
    assert one.stats == many.stats
    other = run_arena(ArenaConfig(MIXED, deals=60, seed=12), workers=1)
    assert other.hand_rows != one.hand_rows


def test_every_hand_is_zero_sum_and_every_agent_plays_every_seat():
    cfg = ArenaConfig(MIXED, deals=40, seed=3)
    r = run_arena(cfg, workers=1)
    assert len(r.hand_rows) == 40 * 4 and len(r.player_rows) == 40 * 4 * 4
    by_hand: dict[tuple[int, int], int] = {}
    seats: dict[str, set[int]] = {}
    for deal, rot, seat, agent, _pos, net, _sd in r.player_rows:
        by_hand[deal, rot] = by_hand.get((deal, rot), 0) + net
        seats.setdefault(agent, set()).add(seat)
    assert set(by_hand.values()) == {0}
    assert all(s == {0, 1, 2, 3} for s in seats.values())


@pytest.mark.parametrize("kind, n", [("station", 2), ("rule", 2), ("rule", 3), ("station", 6)])
def test_identical_deterministic_agents_break_exactly_even(kind, n):
    # Duplicate play gives every agent every seat with the same cards, so
    # identical deterministic agents must net exactly zero on every deal.
    cfg = ArenaConfig(tuple(AgentSpec(f"{kind}{i}", kind) for i in range(n)), deals=50, seed=9)
    r = run_arena(cfg, workers=1)
    for s in r.stats:
        assert s.bb_per_100 == 0 and s.stddev == 0


def test_hands_replay_from_recorded_actions():
    cfg = ArenaConfig(MIXED, deals=30, seed=21)
    r = run_arena(cfg, workers=1)
    nets = {(d, rot, seat): net for d, rot, seat, _a, _p, net, _s in r.player_rows}
    kinds = {"f": ActionType.FOLD, "x": ActionType.CHECK, "c": ActionType.CALL,
             "b": ActionType.BET, "r": ActionType.RAISE}
    stack = cfg.stack_bb * cfg.big_blind
    for deal, rot, deck_seed, seating, actions, board in r.hand_rows:
        n = len(seating.split(","))
        h = HandState({s: stack for s in range(n)}, 0, cfg.small_blind, cfg.big_blind, Deck(deck_seed))
        for tok in actions.split(","):
            h.apply(Action(kinds[tok[0]], int(tok[1:] or 0)))
        assert h.is_over
        assert all(h.result().net[s] == nets[deal, rot, s] for s in range(n))


def test_process_boundary_matches_inline():
    inline = run_arena(ArenaConfig(MIXED, deals=8, seed=4), workers=1)
    isolated = run_arena(ArenaConfig(MIXED, deals=8, seed=4, boundary="process"), workers=1)
    assert inline.player_rows == isolated.player_rows


def test_config_validation():
    with pytest.raises(ValueError):
        ArenaConfig(specs("rule"), deals=1, seed=0)
    with pytest.raises(ValueError):
        ArenaConfig((AgentSpec("a", "rule"), AgentSpec("a", "random")), deals=1, seed=0)
    with pytest.raises(ValueError):
        ArenaConfig(MIXED, deals=0, seed=0)
    with pytest.raises(ValueError):
        AgentSpec("x", "nope").build(0)


@pytest.mark.parametrize("df, expected", [(3, 3.182), (5, 2.571), (10, 2.228), (30, 2.042), (1000, 1.962)])
def test_t_quantile(df, expected):
    assert t_quantile(0.975, df) == pytest.approx(expected, abs=0.01 if df > 3 else 0.05)


def test_interval_coverage_is_about_95_percent():
    rng = random.Random(1)
    true_mean, trials, hits = 0.3, 2000, 0
    for _ in range(trials):
        blocks = [rng.gauss(true_mean, 2.0) for _ in range(12)]
        s = summarize("x", blocks, hands=12)
        hits += s.ci_low <= true_mean * 100 <= s.ci_high
    # Binomial sd at p=.95, n=2000 is about 0.005.
    assert 0.935 <= hits / trials <= 0.965


def test_summary_values():
    s = summarize("x", [1.0, 2.0, 3.0], hands=3)
    assert s.bb_per_100 == pytest.approx(200)
    assert s.stddev == pytest.approx(100)
    half = t_quantile(0.975, 2) * 1.0 / 3**0.5 * 100
    assert (s.ci_low, s.ci_high) == pytest.approx((200 - half, 200 + half))
    assert NormalDist().inv_cdf(0.975) < t_quantile(0.975, 50)


def test_parquet_and_database(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    from backend.arena.__main__ import main
    from backend.db import BenchmarkResult, SimulationRun, make_engine, make_sessionmaker

    out = tmp_path / "run"
    db_url = f"sqlite:///{tmp_path / 'sim.db'}"
    assert main(["rule", "a=random", "station", "--deals", "20", "--seed", "5", "--workers", "1",
                 "--out", str(out), "--db", db_url]) == 0
    hands = pq.read_table(out / "hands.parquet")
    players = pq.read_table(out / "players.parquet")
    assert hands.num_rows == 60 and players.num_rows == 180
    summary = json.loads((out / "summary.json").read_text())
    assert summary["config"]["seed"] == 5 and {s["agent"] for s in summary["stats"]} == {"rule", "a", "station"}
    with make_sessionmaker(make_engine(db_url))() as db:
        run = db.query(SimulationRun).one()
        assert run.n_hands == 60 and run.seed == 5 and run.data_path == str(out)
        assert {r.agent for r in db.query(BenchmarkResult)} == {"rule", "a", "station"}
