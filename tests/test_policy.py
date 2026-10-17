"""Milestone 7: policy network data, metrics, resume, inference parity and latency."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from backend.dataset.pipeline import DatasetConfig, run  # noqa: E402
from backend.policy.data import Tensors, prepare  # noqa: E402
from backend.policy.features import SLOT, VOCAB  # noqa: E402
from backend.policy.infer import PolicyRunner, benchmark  # noqa: E402
from backend.policy.train import TrainConfig, train, weighted_metrics  # noqa: E402

CFG = DatasetConfig("ptest", "v0", seed=7, spots=24, shard_size=12, streets={"river": 0.7, "turn": 0.3},
                    max_iterations=60, target_pct_pot=3.0)


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    root = tmp_path_factory.mktemp("policy")
    run(CFG, root / "ds", device="cpu")
    out = root / "f.npz"
    prepare([root / "ds"], out)
    return out


def tcfg(features, **kw):
    return TrainConfig(str(features), epochs=2, batch_size=4096, hidden=64, range_dim=16, **kw)


def test_splits_are_disjoint_by_spot(features):
    d = Tensors(features, "cpu")
    spots = {name: set(d.node_spot[d.s_node[idx].numpy()]) for name, idx in
             (("train", d.train_idx), ("val", d.val_idx), ("test", d.test_idx))}
    assert spots["train"] and spots["test"]
    assert not (spots["train"] & spots["val"]) and not (spots["train"] & spots["test"])
    assert not (spots["val"] & spots["test"])


def test_targets_and_weights(features):
    d = Tensors(features, "cpu")
    t = d.s_target.float()
    assert torch.allclose(t.sum(1), torch.ones(len(t)), atol=1e-2)
    assert not (t[~d.legal[d.s_node]] > 0).any()  # no target mass on illegal actions
    assert (d.s_weight > 0).all()


def test_metrics_are_zero_for_the_solver_policy():
    target = torch.tensor([[0.2, 0.8, 0.0], [1.0, 0.0, 0.0]])
    ev = torch.tensor([[1.0, 2.0, 0.0], [3.0, -1.0, 0.0]])
    w = torch.tensor([1.0, 3.0])
    ones = torch.ones(2, dtype=torch.bool)
    pot = torch.tensor([10.0, 10.0])
    logp = torch.where(target > 0, target.log(), torch.full_like(target, float("-inf")))
    m = weighted_metrics(logp, target, ev, ones, w, pot)
    assert m["kl"] == pytest.approx(0, abs=1e-5) and m["regret"] == pytest.approx(0, abs=1e-5)
    uniform = torch.log(torch.full((2, 3), 1 / 3))
    m = weighted_metrics(uniform, target, ev, ones, w, pot)
    # regret rows: (1.8 - 1.0) / 10 * 100 = 8 ; (3 - 2/3) / 10 * 100 = 23.33
    assert m["regret"] / m["w_ev"] == pytest.approx((1 * 8 + 3 * (3 - 2 / 3) * 10) / 4)


def test_training_learns_and_resume_is_exact(features, tmp_path):
    full = train(tcfg(features), tmp_path / "full", device="cpu", resume=False, log=lambda *a: None)
    assert full["history"][-1]["train_loss"] < full["history"][0]["train_loss"] + 1e-6
    assert full["test"]["kl_nats"] < full["baselines"]["uniform"]["kl_nats"]
    with pytest.raises(KeyboardInterrupt):
        train(tcfg(features), tmp_path / "resumed", device="cpu", stop_after_epoch=1, log=lambda *a: None)
    resumed = train(tcfg(features), tmp_path / "resumed", device="cpu", log=lambda *a: None)
    assert resumed["resumed_from"] == 1
    a = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError, match="different config"):
        train(tcfg(features, seed=5), tmp_path / "resumed", device="cpu", log=lambda *a: None)


def test_inference_matches_training_features(features, tmp_path):
    train(tcfg(features), tmp_path / "run", device="cpu", resume=False, log=lambda *a: None)
    runner = PolicyRunner(tmp_path / "run" / "model.pt")
    import pyarrow.parquet as pq

    row = next(r for r in pq.read_table(sorted((features.parent / "ds").glob("shard-*.parquet"))[0]).to_pylist()
               if r["street"] == "river")
    reach = (np.asarray(row["reach_oop"], dtype=np.float64), np.asarray(row["reach_ip"], dtype=np.float64))
    combos = np.nonzero(reach[row["player"]] > 0)[0][:20]
    probs = runner.predict(board=row["board"], street=row["street"], pot=row["pot"], stack=row["stack"],
                           inv=(row["inv_oop"], row["inv_ip"]), player=row["player"], actions=row["actions"],
                           reach_oop=reach[0], reach_ip=reach[1], combos=combos)
    assert probs.shape == (len(combos), len(row["actions"]))
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-5)
    # Same samples through the training pipeline.
    d = Tensors(features, "cpu")
    node = next(i for i in range(len(d.numeric)) if d.node_spot[i] == row["spot"])
    idx = torch.nonzero(d.s_node == node).flatten()
    want = {int(c): i for i, c in enumerate(d.s_combo[idx])}
    inputs = d.batch(idx[[want[int(c)] for c in combos]])[0]
    p_train = torch.softmax(runner.model(*inputs), 1)[:, [SLOT[a] for a in row["actions"]]].detach().numpy()
    np.testing.assert_allclose(probs, p_train, atol=2e-3)  # float16 storage of training features


def test_cpu_latency_under_100ms(features, tmp_path):
    from backend.policy.__main__ import _sample_spots

    train(tcfg(features), tmp_path / "lat", device="cpu", resume=False, log=lambda *a: None)
    runner = PolicyRunner(tmp_path / "lat" / "model.pt")
    spots = _sample_spots(10)
    runner.predict(**spots[0])
    assert benchmark(runner, spots)["p95_ms"] < 100


def test_vocabulary_covers_dataset_trees():
    from backend.dataset.spots import DEFAULT_TREES

    labels = {"f", "x", "c", "a"}
    for t in DEFAULT_TREES.values():
        labels |= {f"b{round(f * 100)}" for v in t.bet_sizes.values() for f in v}
        labels |= {f"r{round(f * 100)}" for v in t.raise_sizes.values() for f in v}
    assert labels <= set(VOCAB)


def test_exploitability_tool_reproduces_the_solver_and_measures_a_policy():
    """Writing the solver's own strategy back through ``install`` must give the
    solver's exploitability; a uniform policy must be far more exploitable."""
    from backend.dataset.spots import DEFAULT_TREES, make_spot, tree_config
    from backend.engine.cards import parse_cards
    from backend.policy.exploit import evaluate_spot, install
    from backend.solver.subgame_gpu import BatchedSubgameCFR

    spot = make_spot(3, 0, {"river": 1.0})
    ref = BatchedSubgameCFR(parse_cards(spot.board), spot.ranges(), tree_config(spot, DEFAULT_TREES),
                            device="cpu", dtype=torch.float64)
    ref.iterate(100)
    solved = ref.exploitability()["exploitability_pct_pot"]
    avg = {}

    def collect(node):
        if node.kind == "decision":
            avg[node.path or "root"] = ref._avg(node)[:, 0, :].numpy()
            for c in node.children:
                collect(c)

    collect(ref.root)
    install(ref, spot, lambda ctx: avg[ctx["path"]][:, ctx["combos"]])
    assert ref.exploitability()["exploitability_pct_pot"] == pytest.approx(solved, rel=1e-6)

    uniform = evaluate_spot(spot, lambda ctx: np.full((len(ctx["actions"]), len(ctx["combos"])),
                                                      1 / len(ctx["actions"])),
                            device="cpu", dtype=torch.float64, max_iterations=100)
    assert uniform.solver_pct_pot <= 0.5 + 1e-9 or uniform.solver_pct_pot < uniform.policy_pct_pot
    assert uniform.policy_pct_pot > 5 * uniform.solver_pct_pot


def test_head_to_head_is_zero_against_itself_and_positive_against_uniform():
    from backend.dataset.spots import DEFAULT_TREES, make_spot, tree_config
    from backend.engine.cards import parse_cards
    from backend.policy.exploit import average_table, head_to_head, table_policy
    from backend.solver.subgame_gpu import BatchedSubgameCFR

    spot = make_spot(3, 0, {"river": 1.0})
    cfr = BatchedSubgameCFR(parse_cards(spot.board), spot.ranges(), tree_config(spot, DEFAULT_TREES),
                            device="cpu", dtype=torch.float64)
    cfr.iterate(100)
    solved = table_policy(average_table(cfr))
    assert head_to_head(spot, solved, solved, cfr, 0.0).a_advantage_chips == pytest.approx(0, abs=1e-9)
    uniform = lambda ctx: np.full((len(ctx["actions"]), len(ctx["combos"])), 1 / len(ctx["actions"]))  # noqa: E731
    assert head_to_head(spot, solved, uniform, cfr, 0.0).a_advantage_chips > 0
