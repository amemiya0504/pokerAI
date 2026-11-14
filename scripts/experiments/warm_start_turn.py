"""Warm-starting the turn solver from the policy network (Phase B experiment).

Result: no variant beats the cold start at any iteration count, and running
the network at every node costs 3 to 21 s, so the agent does not use it
(README, "即時求解 Agent"). Run from the repo root:

    .venv/bin/python scripts/experiments/warm_start_turn.py 12

Variants (all DCFR, CUDA graphs off so every variant runs the same code):
  cold        : regrets and strategy sums start at zero (what the agent does)
  regret-W    : Brown & Sandholm (2016) style: treat the network's strategy as
                W iterations already played: regrets = W x instantaneous regrets
                of the network profile, strategy sums = W x reach x network, t = W
  avg-W       : only the strategy sums (average strategy) start from the network
Exploitability is measured after k further iterations.
"""
import json, sys, time
import numpy as np, torch
from backend.dataset.spots import DEFAULT_TREES, make_spot, tree_config
from backend.engine.cards import parse_cards
from backend.policy.data import SPLIT_EVERY
from backend.policy.exploit import install, runner_policy
from backend.policy.infer import PolicyRunner
from backend.solver.subgame_gpu import BatchedSubgameCFR

N_SPOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CHECKS = [10, 25, 50, 100, 200, 400]
VARIANTS = ["cold", "regret-10", "regret-50", "avg-10", "net"]

runner = PolicyRunner("out/policy/run-v1v2/model.pt", device="cuda", cache_boards=128)
policy = runner_policy(runner)


def warm(cfr, spot, variant):
    if variant == "cold":
        return 0.0
    t0 = time.perf_counter()
    install(cfr, spot, policy)  # strat_sum = network strategy at every node
    kind, w = variant.split("-") if "-" in variant else (variant, "0")
    w = int(w)
    with torch.no_grad():
        if kind == "regret":
            cfr._decay.fill_(0.0)  # strategy sum := own reach x network strategy
            for i in (0, 1):
                cfr._traverse(cfr.root, i, cfr.ranges[i], cfr.ranges[1 - i], "cfr", avg=True)
                cfr._apply(cfr.root)  # regrets were 0, so they become the instantaneous regrets
        elif kind == "avg":
            def own_reach(node, r):
                if node.kind == "chance":
                    idx, m = node.parent_idx, node.child_mask
                    own_reach(node.children[0], [r[0][idx] * m, r[1][idx] * m]); return
                if node.kind != "decision":
                    return
                s = cfr._avg(node)
                node.strat_sum = s * r[node.player]
                for k, c in enumerate(node.children):
                    nr = list(r); nr[node.player] = r[node.player] * s[k]; own_reach(c, nr)
            own_reach(cfr.root, [cfr.ranges[0], cfr.ranges[1]])

        def scale(node):
            if node.kind == "decision":
                node.regret.mul_(w); node.strat_sum.mul_(w)
            for c in node.children:
                scale(c)
        if kind != "net":
            scale(cfr.root)
            cfr.t = w
    torch.cuda.synchronize()
    return time.perf_counter() - t0


rows = []
spots = []
for idx in range(0, 3000, SPLIT_EVERY):  # test split: the network never trained on these
    s = make_spot(1, idx, {"river": 0.7, "turn": 0.3})
    if s.street == "turn":
        spots.append(s)
    if len(spots) >= N_SPOTS:
        break
for s in spots:
    row = {"spot": s.index, "board": s.board, "spr": s.stack / s.pot}
    for v in VARIANTS:
        cfr = BatchedSubgameCFR(parse_cards(s.board), s.ranges(), tree_config(s, DEFAULT_TREES), dtype=torch.float32)
        cfr.use_graph = False
        row[f"{v}_setup_s"] = warm(cfr, s, v)
        if v == "net":
            row[v] = {0: cfr.exploitability()["exploitability_pct_pot"]}
            cfr.release(); continue
        curve, done = {}, 0
        for k in CHECKS:
            cfr.iterate(k - done); done = k
            curve[k] = cfr.exploitability()["exploitability_pct_pot"]
        row[v] = curve
        cfr.release()
    rows.append(row)
    print(f"{s.board} SPR {row['spr']:.0f}: net {row['net'][0]:.1f}% | " + " | ".join(
        f"{v} ({row[f'{v}_setup_s']:.1f}s) " + " ".join(f"{row[v][k]:.2f}" for k in CHECKS)
        for v in VARIANTS if v != "net"), flush=True)

summary = {}
for v in VARIANTS[:-1]:
    summary[v] = {k: float(np.median([r[v][k] for r in rows])) for k in CHECKS}
    summary[v]["setup_s_median"] = float(np.median([r[f"{v}_setup_s"] for r in rows]))
    # iterations saved: ratio of cold to warm exploitability at each checkpoint
print(json.dumps(summary, indent=1))
json.dump({"summary": summary, "spots": rows}, open("out/policy/warm_start_turn.json", "w"), indent=1)
