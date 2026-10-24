"""Safe vs unsafe re-solving of river subgames inside a turn game (Phase C).

A blueprint is the turn game solved for only a few iterations (like the
agent's 10-second turn solves, which stop at a few % pot). One player (the
hero) then re-solves every river subgame (each turn line x river card) and
plays the re-solved river strategy, keeping the blueprint on the turn:

  blueprint : no re-solving
  unsafe    : river subgame with both ranges taken from the blueprint
  safe-cbv  : re-solve gadget (prior-weighted, Brown & Sandholm 2017) with the
              opponent's counterfactual best-response values under the blueprint
  safe-ev   : the same gadget with the opponent's blueprint values (both
              playing the blueprint; DeepStack-style targets, not guaranteed safe)

``--range-error`` makes the opponent range the re-solver uses wrong (the hero's
own range and the blueprint, hence the gadget's targets, stay right), as when
the agent's range tracking misreads the opponent:

  noise : each combo's weight x exp(N(0, 1)), fixed per spot
  stale : the opponent's range at the start of the turn (his turn actions ignored)

Measured on the full turn game: the hero's exploitability (opponent's best
response vs the hero's combined strategy, minus the game value) and the hero's
EV against a near-equilibrium opponent, both in % pot. Run from the repo root:

    .venv/bin/python scripts/experiments/safe_resolve_turn.py --spots 8
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from backend.dataset.spots import DEFAULT_TREES, make_spot, tree_config
from backend.engine.cards import parse_cards
from backend.policy.data import SPLIT_EVERY
from backend.policy.exploit import average_table, install, mixed, table_policy
from backend.solver.subgame import TreeConfig
from backend.solver.subgame_gpu import BatchedSubgameCFR, BNode, Gadget

VARIANTS = ("unsafe", "safe-cbv", "safe-ev")


def river_roots(node: BNode, out: list[BNode]) -> list[BNode]:
    """Decision nodes that start the river (path ends with "*/")."""
    if node.kind == "decision" and node.path.endswith("*/"):
        out.append(node)
        return out
    for c in node.children:
        river_roots(c, out)
    return out


def resolve_rivers(bp: BatchedSubgameCFR, cfg: TreeConfig, hero: int, iterations: int, variant: str,
                   table: dict[str, np.ndarray], range_error: str = "none", seed: int = 0) -> None:
    """Overwrite the hero's river nodes in ``table`` with re-solved strategies."""
    o = 1 - hero
    noise = np.exp(np.random.default_rng(seed).normal(0.0, 1.0, 1326))
    for r in river_roots(bp.root, []):
        assert abs(r.inv[0] - r.inv[1]) < 1e-9
        reach = bp.reach_to(r)
        if float(reach[0].sum()) == 0 or float(reach[1].sum()) == 0:
            continue
        sub_cfg = TreeConfig(cfg.pot + r.inv[0] + r.inv[1], cfg.stack - r.inv[0], {"river": cfg.bet_sizes["river"]},
                             {"river": cfg.raise_sizes.get("river", [])}, cfg.max_raises, cfg.allin,
                             cfg.allin_threshold)
        gadget = None
        if variant != "unsafe":
            with torch.no_grad():
                mode = "br" if variant == "safe-cbv" else "ev"
                v = bp._traverse(r, o, reach[o], reach[hero], mode, avg=mode == "ev")
            gadget = Gadget(o, v.double().cpu().numpy())
        ranges = [x.double().cpu().numpy() for x in reach]
        if range_error == "noise":
            ranges[o] = ranges[o] * noise
        elif range_error == "stale":
            ranges[o] = np.broadcast_to(bp.ranges[o].double().cpu().numpy(), ranges[o].shape) * (ranges[o] > 0)
        sub = BatchedSubgameCFR(r.boards, ranges, sub_cfg, dtype=torch.float32, gadget=gadget)
        sub.iterate(iterations)
        for path, avg in average_table(sub).items():
            full = r.path + ("" if path == "root" else path)
            node = sub.find(path)
            if node.player == hero:
                table[full] = avg
        sub.release()


def evaluate(spot, cfg, hero_table, opp_table, hero: int) -> dict:
    """Hero's best-response gap and EV vs ``opp_table`` on the full game (chips)."""
    cfr = BatchedSubgameCFR(parse_cards(spot.board), spot.ranges(), cfg, dtype=torch.float32)
    hp, op = table_policy(hero_table), table_policy(opp_table)
    install(cfr, spot, mixed(hp, op) if hero == 0 else mixed(op, hp))
    ex = cfr.exploitability()
    cfr.release()
    ev_hero = ex["ev_oop_chips"] if hero == 0 else -ex["ev_oop_chips"]
    return {"opp_best_response": ex["best_response_chips"][1 - hero], "ev": ev_hero}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spots", type=int, default=8)
    ap.add_argument("--blueprint-iterations", type=int, nargs="+", default=[20, 60])
    ap.add_argument("--resolve-iterations", type=int, default=400)
    ap.add_argument("--range-error", nargs="+", default=["none"], choices=["none", "noise", "stale"])
    ap.add_argument("--out", default="out/policy/safe_resolve_turn.json")
    args = ap.parse_args()

    spots = []
    for idx in range(0, 5000, SPLIT_EVERY):
        s = make_spot(1, idx, {"river": 0.7, "turn": 0.3})
        if s.street == "turn":
            spots.append(s)
        if len(spots) >= args.spots:
            break
    rows = []
    for spot in spots:
        t0 = time.perf_counter()
        cfg = tree_config(spot, DEFAULT_TREES)
        board = parse_cards(spot.board)
        eq = BatchedSubgameCFR(board, spot.ranges(), cfg, dtype=torch.float32)
        while True:
            eq.iterate(200)
            eqx = eq.exploitability()
            if eqx["exploitability_pct_pot"] < 0.05 or eq.t >= 2000:
                break
        eq_table = average_table(eq)
        value = [eqx["ev_oop_chips"], -eqx["ev_oop_chips"]]  # game value estimate per seat
        eq.release()
        for k in args.blueprint_iterations:
            bp = BatchedSubgameCFR(board, spot.ranges(), cfg, dtype=torch.float32)
            bp.iterate(k)
            bpx = bp.exploitability()["exploitability_pct_pot"]
            bp_table = average_table(bp)
            for err in args.range_error:
                row = {"spot": spot.index, "board": spot.board, "spr": spot.stack / spot.pot, "blueprint_iterations": k,
                       "range_error": err, "blueprint_pct_pot": bpx, "equilibrium_pct_pot": eqx["exploitability_pct_pot"]}
                for hero in (0, 1):
                    tables = {"blueprint": bp_table}
                    for v in VARIANTS:
                        t = dict(bp_table)
                        resolve_rivers(bp, cfg, hero, args.resolve_iterations, v, t, err, seed=spot.index)
                        tables[v] = t
                    for name, t in tables.items():
                        e = evaluate(spot, cfg, t, eq_table, hero)
                        row[f"h{hero}_{name}_exploit_pct_pot"] = 100 * (e["opp_best_response"] - value[1 - hero]) / spot.pot
                        row[f"h{hero}_{name}_vs_eq_pct_pot"] = 100 * (e["ev"] - value[hero]) / spot.pot
                row["seconds"] = time.perf_counter() - t0
                rows.append(row)
                names = ("blueprint",) + VARIANTS
                ex = " ".join(f"{n} {np.mean([row[f'h{h}_{n}_exploit_pct_pot'] for h in (0, 1)]):.2f}" for n in names)
                vs = " ".join(f"{n} {np.mean([row[f'h{h}_{n}_vs_eq_pct_pot'] for h in (0, 1)]):+.2f}" for n in names)
                print(f"{spot.board} SPR {row['spr']:.0f} bp {k} it ({bpx:.1f}%) range {err}: exploit [{ex}] | "
                      f"vs eq [{vs}] ({row['seconds']:.0f}s)", flush=True)
            bp.release()
    summary = {}
    for k in args.blueprint_iterations:
        for err in args.range_error:
            rs = [r for r in rows if r["blueprint_iterations"] == k and r["range_error"] == err]
            key = f"bp{k}-{err}"
            summary[key] = {n: {m: float(np.median([np.mean([r[f"h{h}_{n}_{m}"] for h in (0, 1)]) for r in rs]))
                                for m in ("exploit_pct_pot", "vs_eq_pct_pot")}
                            for n in ("blueprint",) + VARIANTS}
            summary[key]["blueprint_pct_pot_median"] = float(np.median([r["blueprint_pct_pot"] for r in rs]))
    print(json.dumps(summary, indent=1))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "spots": rows, "args": vars(args)}, f, indent=1)


if __name__ == "__main__":
    main()
