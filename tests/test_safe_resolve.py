"""Safe re-solving: the re-solve gadget, counterfactual values, batched boards."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.engine.cards import parse_cards  # noqa: E402
from backend.solver.subgame import TreeConfig, parse_range  # noqa: E402
from backend.solver.subgame_gpu import BatchedSubgameCFR, Gadget  # noqa: E402

BOARD = parse_cards("Ah9c6d3s2h")
CFG = TreeConfig(10, 30, {"river": [0.5, 1.0]}, {"river": [1.0]}, 1, True)
RANGES = (parse_range("99+,AJs+,AQo+,T9s,87s", BOARD), parse_range("77+,ATs+,KQs,JTs,98s,AJo+", BOARD))


def _cfr(**kw):
    return BatchedSubgameCFR(BOARD, RANGES, CFG, device="cpu", dtype=torch.float64, **kw)


@pytest.mark.parametrize("player", [0, 1])
def test_gadget_with_extreme_targets(player):
    plain = _cfr()
    plain.iterate(200)
    follow = _cfr(gadget=Gadget(player, np.full(1326, -1e4)))  # terminating is terrible: always follow
    follow.iterate(200)
    e = follow.exploitability()
    assert e["follow_share"] == pytest.approx(1.0)
    assert e["exploitability_pct_pot"] == pytest.approx(plain.exploitability()["exploitability_pct_pot"], abs=0.01)
    leave = _cfr(gadget=Gadget(player, np.full(1326, 1e4)))  # terminating is great: never follow
    leave.iterate(200)
    e = leave.exploitability()
    assert e["follow_share"] < 0.01 and e["safety_excess_chips"] == 0


@pytest.mark.parametrize("player", [0, 1])
def test_resolving_against_best_response_values_is_safe(player):
    """Targets = the protected player's counterfactual best-response values
    against a poor blueprint: after re-solving, his best response may not beat
    any of them (Brown & Sandholm 2017)."""
    bp = _cfr()
    bp.iterate(15)
    cbv, _, _ = bp.counterfactual_values(player, best_response=True)
    cfr = _cfr(gadget=Gadget(player, cbv))
    cfr.iterate(400)
    assert cfr.exploitability()["safety_excess_chips"] < 1e-3 * CFG.pot  # per hand
    v = cfr._traverse(cfr.root, player, cfr.ranges[player], cfr.ranges[1 - player], "br")[0].numpy()
    live = RANGES[player] > 0
    opp_mass = float(cfr.ranges[1 - player].sum())
    assert (v[live] - cbv[live]).max() < 1e-3 * CFG.pot * opp_mass  # no combo gains more than 0.1% pot


def test_counterfactual_values_match_a_subgame_rooted_at_the_node():
    """Values at a river node of a turn tree equal the root values of a river
    subgame (pot and stack carried over) playing the same strategy from the
    same reach: the units ``Gadget.cfv`` needs."""
    turn = parse_cards("Ks7d4c2h")
    cfg = TreeConfig(10, 15, {"turn": [1.0], "river": [1.0]}, {}, 0, True)
    ranges = (parse_range("TT+,AKs", turn), parse_range("99+,AQs+,KQs", turn))
    full = BatchedSubgameCFR(turn, ranges, cfg, device="cpu", dtype=torch.float64)
    full.iterate(60)
    path = "b100c*/"
    node = full.find(path)
    assert node.inv == (10.0, 10.0)
    river = tuple(int(c) for c in node.boards[7])
    sub_cfg = TreeConfig(30, 5, {"river": [1.0]}, {}, 0, True)
    for player in (0, 1):
        for br in (False, True):
            v, r0, r1 = full.counterfactual_values(player, path, river, best_response=br)
            sub = BatchedSubgameCFR(river, (r0, r1), sub_cfg, device="cpu", dtype=torch.float64)

            def copy(n):
                if n.kind == "decision":
                    src = full.find(path + n.path)
                    b = [tuple(int(c) for c in x) for x in src.boards].index(river)
                    n.strat_sum = full._avg(src)[:, b:b + 1, :].clone()
                for c in n.children:
                    copy(c)

            copy(sub.root)
            w, _, _ = sub.counterfactual_values(player, best_response=br)
            np.testing.assert_allclose(w, v, atol=1e-9)


def test_batched_boards_equal_separate_solves():
    boards = [parse_cards(b) for b in ("Ah9c6d3s2h", "Ah9c6d3sKc", "Ah9c6d3s9s")]
    sep = []
    for b in boards:
        r = (parse_range("99+,AJs+,AQo+,T9s,87s", b), parse_range("77+,ATs+,KQs,JTs,98s,AJo+", b))
        c = BatchedSubgameCFR(b, r, CFG, device="cpu", dtype=torch.float64)
        c.iterate(50)
        sep.append(c)
    r = tuple(np.stack([parse_range(s, b) for b in boards]) for s in ("99+,AJs+,AQo+,T9s,87s", "77+,ATs+,KQs,JTs,98s,AJo+"))
    batch = BatchedSubgameCFR(np.array(boards), r, CFG, device="cpu", dtype=torch.float64)
    batch.iterate(50)
    for i, c in enumerate(sep):
        torch.testing.assert_close(batch._avg(batch.root)[:, i], c._avg(c.root)[:, 0])
