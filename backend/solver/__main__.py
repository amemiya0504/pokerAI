"""Solvers: preflop push/fold datasets, Kuhn/Leduc, heads-up postflop subgames.

    python -m backend.solver pushfold --players 2-9 --stacks 5,8,10,12,15,20 --db sqlite:///poker.db
    python -m backend.solver small leduc --iterations 3000
    python -m backend.solver subgame --board Ks7d4c2h --oop "22+,ATs+" --ip "55+,KQs" \
        --pot 10 --stack 40 --bets turn=50,100 river=50,100 --raises turn=100 river=100
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .pushfold import PushFoldSolver, request
from .store import PUSHFOLD_DIR, record, write


def _range(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def default_deals(players: int) -> int:
    """Total sampled deals: bigger tables have many more infosets."""
    return {2: 6_000_000, 3: 8_000_000, 4: 10_000_000, 5: 12_000_000}.get(players, 20_000_000)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.solver", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("pushfold")
    pf.add_argument("--players", default="2-9", help="e.g. 2-9 or 2,6,9")
    pf.add_argument("--stacks", default="5,8,10,12,15,20", help="effective stacks in big blinds")
    pf.add_argument("--deals", type=int, help="sampled deals per solve (default depends on players)")
    pf.add_argument("--batches", type=int, default=200)
    pf.add_argument("--eval-deals", type=int, default=2_000_000)
    pf.add_argument("--seed", type=int, default=1)
    pf.add_argument("--workers", type=int)
    pf.add_argument("--out", default=str(PUSHFOLD_DIR))
    pf.add_argument("--db", help="register datasets and cache results in this database")
    sm = sub.add_parser("small", help="Kuhn or Leduc with exact exploitability")
    sm.add_argument("game", choices=["kuhn", "leduc"])
    sm.add_argument("--iterations", type=int, default=2000)
    sm.add_argument("--threshold", type=float, default=1e-3,
                    help="NashConv below this is labelled verified_equilibrium")
    sg = sub.add_parser("subgame", help="heads-up postflop subgame")
    sg.add_argument("--board", required=True)
    sg.add_argument("--oop", required=True, help="OOP range, e.g. 22+,ATs+,KQo")
    sg.add_argument("--ip", required=True, help="IP range")
    sg.add_argument("--pot", type=float, required=True)
    sg.add_argument("--stack", type=float, required=True, help="effective stack behind")
    sg.add_argument("--bets", nargs="*", default=["flop=33,75", "turn=75", "river=75,150"],
                    help="street=percent,... of the pot")
    sg.add_argument("--raises", nargs="*", default=["flop=100", "turn=100", "river=100"])
    sg.add_argument("--max-raises", type=int, default=2)
    sg.add_argument("--no-allin", action="store_true")
    sg.add_argument("--iterations", type=int, default=300)
    sg.add_argument("--target", type=float, default=0.5, help="stop at this exploitability (%% pot)")
    sg.add_argument("--variant", choices=["dcfr", "cfr+"], default="dcfr")
    sg.add_argument("--device", default="auto",
                    help="auto (CUDA if available), cuda, cpu (batched PyTorch) or numpy (reference)")
    sg.add_argument("--memory-limit-gb", type=float, default=6.0)
    sg.add_argument("--out", help="write the result JSON here")
    for p in (sm, sg):
        p.add_argument("--db", help="cache results by request hash in this database")
    args = ap.parse_args(argv)

    if args.cmd in ("small", "subgame"):
        return _solve_one(args)

    from pathlib import Path

    solver = PushFoldSolver(args.workers)
    for players in _range(args.players):
        for stack in [float(x) for x in args.stacks.split(",")]:
            deals = args.deals or default_deals(players)
            req = request(players, stack, batches=args.batches,
                          deals_per_batch=-(-deals // args.batches),
                          eval_deals=args.eval_deals, seed=args.seed)
            t0 = time.perf_counter()
            res = solver.solve(req)
            path = write(res, Path(args.out))
            c = res.convergence
            print(f"{players}max {stack:g}bb: {res.iterations:,} deals, {time.perf_counter() - t0:.0f}s, "
                  f"NashConv in [{c['nash_conv_lower_bb']:.4f}, {c['nash_conv_upper_bb']:.4f}] bb/hand "
                  f"-> {path}", flush=True)
            if args.db:
                record(args.db, res, path)
    return 0



def _sizes(items: list[str]) -> dict[str, list[float]]:
    out = {}
    for item in items:
        street, _, pcts = item.partition("=")
        out[street] = [int(x) / 100 for x in pcts.split(",") if x]
    return out


def _solve_one(args) -> int:
    import json
    from pathlib import Path

    from .base import SolveRequest, SolveResult
    from .cfr import SmallGameSolver
    from .subgame import SubgameSolver

    if args.cmd == "small":
        req = SolveRequest(args.game, {}, {"iterations": args.iterations, "threshold": args.threshold})
        solver = SmallGameSolver()
    else:
        req = SolveRequest("subgame", {
            "board": args.board, "oop_range": args.oop, "ip_range": args.ip,
            "pot": args.pot, "stack": args.stack, "bet_sizes": _sizes(args.bets),
            "raise_sizes": _sizes(args.raises), "max_raises": args.max_raises,
            "allin": not args.no_allin,
        }, {"iterations": args.iterations, "target_pct_pot": args.target, "variant": args.variant})
        if args.device == "numpy":
            solver = SubgameSolver()
        else:
            from .subgame_gpu import GpuSubgameSolver, default_device

            device = default_device() if args.device == "auto" else args.device
            # The device does not change the answer, so it stays out of the request hash.
            solver = _WithOptions(GpuSubgameSolver(), device=device,
                                  memory_limit_gb=args.memory_limit_gb)

    result = None
    if args.db:
        from backend.db import SolverResultRow, make_engine, make_sessionmaker, upgrade

        upgrade(args.db)
        with make_sessionmaker(make_engine(args.db))() as db:
            row = db.query(SolverResultRow).filter_by(request_hash=req.key()).one_or_none()
            if row is not None:
                result = SolveResult.from_json(row.result)
                print(f"cache hit {req.key()[:12]}")
    if result is None:
        result = solver.solve(req)
        if args.db:
            _cache(args.db, result)
    c = result.convergence
    if args.cmd == "small":
        print(f"{args.game}: NashConv {c['nash_conv']:.2e}, value(P1) {c['game_value_p0']:.5f}, "
              f"{result.iterations} iterations, {result.seconds:.1f}s -> {result.strategy_source}")
    else:
        print(f"subgame {args.board}: exploitability {c['exploitability_pct_pot']:.3f}% pot after "
              f"{result.iterations} iterations ({result.seconds:.0f}s), OOP EV {c['ev_oop_chips']:+.3f} chips")
        root = result.strategy.get("root")
        if root:
            freq = np.mean(list(root["combos"].values()), axis=0)
            print("OOP root:", ", ".join(f"{a} {100 * f:.1f}%" for a, f in zip(root["actions"], freq)))
    if getattr(args, "out", None):
        Path(args.out).write_text(json.dumps(result.to_json(), ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def _cache(db_url: str, result) -> None:
    from backend.db import SolverResultRow, make_engine, make_sessionmaker

    with make_sessionmaker(make_engine(db_url))() as db:
        c = result.convergence
        db.add(SolverResultRow(
            request_hash=result.request.key(),
            config={"game": result.request.game, **result.request.params, "solver": result.request.solver},
            iterations=result.iterations,
            exploitability=c.get("nash_conv", c.get("exploitability_pct_pot")),
            result=result.to_json(), source=str(result.strategy_source),
        ))
        db.commit()



class _WithOptions:
    """Pass run-only options (device, memory limit) without changing the request hash."""

    def __init__(self, solver, **options) -> None:
        self.solver, self.options = solver, options

    def solve(self, req):
        from dataclasses import replace

        res = self.solver.solve(replace(req, solver={**req.solver, **self.options}))
        res.request = req
        return res


if __name__ == "__main__":
    sys.exit(main())
