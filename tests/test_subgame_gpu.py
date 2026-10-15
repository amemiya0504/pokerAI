"""Batched (PyTorch) subgame solver: identical to the NumPy reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from backend.engine.cards import parse_cards  # noqa: E402
from backend.solver.base import SolveRequest  # noqa: E402
from backend.solver.subgame import SubgameCFR, TreeConfig, parse_range  # noqa: E402
from backend.solver.subgame_gpu import BatchedSubgameCFR, GpuSubgameSolver  # noqa: E402

SPOTS = [
    ("Ah9c6d3s2h", TreeConfig(10, 30, {"river": [0.5, 1.0]}, {"river": [1.0]}, 1, True),
     "99+,AJs+,AQo+,T9s,87s", "77+,ATs+,KQs,JTs,98s,AJo+"),
    ("Ks7d4c2h", TreeConfig(10, 15, {"turn": [1.0], "river": [1.0]}, {}, 0, True),
     "TT+,AKs", "99+,AQs+,KQs"),
]


@pytest.mark.parametrize("variant", ["dcfr", "cfr+"])
@pytest.mark.parametrize("board, cfg, oop, ip", SPOTS)
def test_batched_float64_equals_numpy_reference(board, cfg, oop, ip, variant):
    b = parse_cards(board)
    ranges = (parse_range(oop, b), parse_range(ip, b))
    ref = SubgameCFR(b, ranges, cfg, variant)
    ref.iterate(15)
    bat = BatchedSubgameCFR(b, ranges, cfg, variant, device="cpu", dtype=torch.float64)
    bat.iterate(15)
    e_ref, e_bat = ref.exploitability(), bat.exploitability()
    # Regrets agree to ~1e-13 per step. CFR+ floors regrets at zero, so such a
    # difference can flip a regret between 0 and a tiny positive value and
    # switch that hand between uniform and pure play: allow for it there.
    rel, prob_tol = (1e-9, 1e-4) if variant == "dcfr" else (1e-3, 0.05)
    for k in ("nash_conv_chips", "ev_oop_chips"):
        assert e_bat[k] == pytest.approx(e_ref[k], rel=rel, abs=1e-12)
    s_ref, s_bat = ref.strategy(max_depth=0), bat.strategy()
    assert s_ref.keys() == s_bat.keys()
    for node in s_ref:
        for combo, probs in s_ref[node]["combos"].items():
            assert s_bat[node]["combos"][combo] == pytest.approx(probs, abs=prob_tol)
    assert bat.template_nodes <= ref.decision_nodes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_graph_matches_eager_cuda():
    board, cfg, oop, ip = SPOTS[1]
    b = parse_cards(board)
    ranges = (parse_range(oop, b), parse_range(ip, b))
    graph = BatchedSubgameCFR(b, ranges, cfg, device="cuda")
    graph.iterate(25)
    eager = BatchedSubgameCFR(b, ranges, cfg, device="cuda")
    eager.use_graph = False
    eager.iterate(25)
    assert graph._graph is not None
    assert graph.exploitability()["exploitability_pct_pot"] == pytest.approx(
        eager.exploitability()["exploitability_pct_pot"], rel=1e-3)


def test_memory_limit_is_enforced():
    b = parse_cards("Ks7d4c")
    cfg = TreeConfig(10, 90, {"flop": [0.33, 0.75], "turn": [0.75], "river": [0.75]}, {}, 0, True)
    with pytest.raises(MemoryError, match="GB"):
        BatchedSubgameCFR(b, (parse_range("all", b), parse_range("all", b)), cfg,
                          device="cpu", memory_limit_gb=0.001)


def test_gpu_solver_result_shape():
    params = {"board": "Ks7d4c2h9s", "pot": 10, "stack": 1000, "oop_range": "QQ,JJ",
              "ip_range": "77,44,J8s,T8s,T6s,86s,65s,J5s", "bet_sizes": {"river": [1.0]},
              "raise_sizes": {}, "max_raises": 0, "allin": False}
    r = GpuSubgameSolver().solve(SolveRequest("subgame", params, {"iterations": 600, "check_every": 300}))
    assert r.convergence["exploitability_pct_pot"] < 0.05
    facing = r.strategy["xb100"]
    c = facing["actions"].index("c")
    calls = [v[c] for v in facing["combos"].values()]
    assert sum(calls) / len(calls) == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("device", ["cpu", "numpy"])
def test_cli_subgame(device, capsys):
    from backend.solver.__main__ import main

    assert main(["subgame", "--board", "Ks7d4c2h9s", "--oop", "QQ,JJ", "--ip", "77,44,J8s,T8s",
                 "--pot", "10", "--stack", "100", "--bets", "river=100", "--raises",
                 "--max-raises", "0", "--no-allin", "--iterations", "50", "--device", device]) == 0
    assert "exploitability" in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_back_to_back_cuda_solves_with_garbage_collection():
    # Regression: freeing an earlier solver's CUDA graph while the next one
    # was being captured invalidated the capture (seen in long dataset builds).
    import gc

    board, cfg, oop, ip = SPOTS[1]
    b = parse_cards(board)
    ranges = (parse_range(oop, b), parse_range(ip, b))
    leftovers = []
    for k in range(6):
        cfr = BatchedSubgameCFR(b, ranges, cfg, device="cuda")
        cfr.iterate(6)
        assert cfr.exploitability()["exploitability_pct_pot"] >= 0
        if k % 2:
            cfr.release()
        else:
            leftovers.append(cfr)  # left for the garbage collector
        gc.collect()
    leftovers.clear()
    gc.collect()
