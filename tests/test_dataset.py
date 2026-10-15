"""Milestone 6: dataset pipeline checks, interruption/resume, and validator power."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

from backend.dataset.pipeline import DatasetConfig, register, run, shard_path, solve_spot  # noqa: E402
from backend.dataset.spots import make_spot, tree_config  # noqa: E402
from backend.dataset.validate import validate  # noqa: E402
from backend.engine.cards import parse_cards  # noqa: E402
from backend.solver.subgame import N, board_mask, compatible_mass  # noqa: E402
from backend.solver.subgame_gpu import BatchedSubgameCFR  # noqa: E402

CFG = DatasetConfig("test", "v0", seed=3, spots=6, shard_size=2, streets={"river": 1.0},
                    max_iterations=100, target_pct_pot=2.0)


def test_spots_are_deterministic_and_sane():
    a = make_spot(1, 7, {"river": 0.5, "turn": 0.5})
    assert a == make_spot(1, 7, {"river": 0.5, "turn": 0.5})
    assert a != make_spot(1, 8, {"river": 0.5, "turn": 0.5})
    for i in range(30):
        s = make_spot(1, i, {"river": 0.5, "turn": 0.5})
        board = parse_cards(s.board)
        assert len(set(board)) == len(board) == {"river": 5, "turn": 4}[s.street]
        for r in s.ranges():
            assert r.sum() > 0 and not (r * (1 - board_mask(board))).any()


def test_node_table_evs_reproduce_the_game_value():
    spot = make_spot(2, 0, {"river": 1.0})
    cfr = BatchedSubgameCFR(parse_cards(spot.board), spot.ranges(), tree_config(spot, CFG.trees),
                            device="cpu", dtype=torch.float64)
    cfr.iterate(60)
    rows = cfr.node_table()
    root = rows[0]
    assert root["path"] == "root" and root["player"] == 0
    r_oop, r_ip = root["reach"]
    mass = compatible_mass(r_ip)
    node_ev = np.nansum(root["strategy"] * root["ev"], axis=0)  # per combo, under the average strategy
    norm = (r_oop * mass).sum()
    assert (r_oop * mass * node_ev).sum() / norm == pytest.approx(cfr.exploitability()["ev_oop_chips"], rel=1e-9)
    # Every child row's reach is the parent's reach times the chosen action's probability.
    child = next(r for r in rows if r["path"] == "x")
    np.testing.assert_allclose(child["reach"][0], r_oop * root["strategy"][0], rtol=1e-12)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds") / "full"
    run(CFG, out, device="cpu")
    return out


def test_build_validates_and_registers(built, tmp_path):
    report = validate(built)
    assert report["ok"], report["errors"]
    assert report["spots"] == 6 and report["missing_shards"] == 0 and report["rows"] >= 6
    m = json.loads((built / "manifest.json").read_text())
    assert m["strategy_source"] == "solver_approx" and "no third-party data" in m["license"]
    db = f"sqlite:///{tmp_path / 'r.db'}"
    register(db, built, report)
    from backend.db import GtoDataset, make_engine, make_sessionmaker

    with make_sessionmaker(make_engine(db))() as s:
        row = s.query(GtoDataset).one()
        assert row.row_count == report["rows"] and row.source == "solver_approx"


def test_interrupted_build_resumes_to_the_same_dataset(built, tmp_path):
    out = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        run(CFG, out, device="cpu", stop_after=3)  # dies inside the second shard
    assert shard_path(out, 0).exists() and not shard_path(out, 1).exists()
    assert not list(out.glob("shard-00001*"))  # no half-written shard under the real name
    res = run(CFG, out, device="cpu")
    assert res["solved_now"] == 4  # shard 0 was kept; shards 1 and 2 solved now
    for k in range(CFG.n_shards):
        assert pq.read_table(shard_path(out, k)).equals(pq.read_table(shard_path(built, k)))


def test_resume_refuses_a_different_config(built):
    other = DatasetConfig(**{**CFG.__dict__, "seed": 99})
    with pytest.raises(ValueError, match="different config"):
        run(other, built, device="cpu")


def _corrupt(src, dst, fn):
    import shutil

    shutil.copytree(src, dst)
    t = pq.read_table(shard_path(dst, 0)).to_pylist()
    fn(t[0])
    pq.write_table(pa.Table.from_pylist(t, schema=pq.read_schema(shard_path(dst, 0))), shard_path(dst, 0))
    return validate(dst)


@pytest.mark.parametrize("name, mutate, message", [
    ("probability", lambda r: r["strategy"].__setitem__(
        int(np.argmax(np.asarray(r["reach_oop"]) > 0)), 0.2), "sum to"),
    ("card conflict", lambda r: r["reach_oop"].__setitem__(
        int(np.argmin(board_mask(parse_cards(r["board"])))), 0.5), "conflicts with the board"),
    ("illegal action", lambda r: r.__setitem__("actions", ["f"] + r["actions"][1:]), "actions"),
    ("unknown node", lambda r: r.__setitem__("path", "zz"), "not in the bet tree"),
    ("negative reach", lambda r: r["reach_ip"].__setitem__(
        int(np.argmax(np.asarray(r["reach_ip"]) > 0)), -0.5), "negative reach"),
])
def test_validator_catches_corruption(built, tmp_path, name, mutate, message):
    report = _corrupt(built, tmp_path / name.replace(" ", "_"), mutate)
    assert not report["ok"]
    assert any(message in e for e in report["errors"]), report["errors"]


def test_validator_reports_missing_shard(built, tmp_path):
    import shutil

    dst = tmp_path / "missing"
    shutil.copytree(built, dst)
    shard_path(dst, 1).unlink()
    report = validate(dst)
    assert not report["ok"] and report["missing_shards"] == 1


def test_solve_spot_respects_target():
    spot = make_spot(5, 1, {"river": 1.0})
    rows, conv = solve_spot(spot, CFG, device="cpu")
    assert conv["exploitability_pct_pot"] <= CFG.target_pct_pot or conv["iterations"] == CFG.max_iterations
    assert rows[0]["strategy"].shape[1] == N
