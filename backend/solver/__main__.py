"""Solve push/fold games and write strategy datasets.

    python -m backend.solver pushfold --players 2-9 --stacks 5,8,10,12,15,20 --db sqlite:///poker.db
"""

from __future__ import annotations

import argparse
import sys
import time

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
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
