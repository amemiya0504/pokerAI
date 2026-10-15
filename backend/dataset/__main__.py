"""Build, resume and validate GTO datasets.

    python -m backend.dataset build out/datasets/v1 --spots 1000 --seed 1
    python -m backend.dataset validate out/datasets/v1 --db sqlite:///poker.db

``build`` resumes automatically: run the same command again after an
interruption and only the missing shards are solved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import DatasetConfig, register, run
from .validate import validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.dataset", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("out", type=Path)
    b.add_argument("--name", default="gto-postflop-hu")
    b.add_argument("--version", default="v1")
    b.add_argument("--spots", type=int, default=1000)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--shard-size", type=int, default=50)
    b.add_argument("--river-share", type=float, default=0.7, help="the rest are turn spots")
    b.add_argument("--target", type=float, default=0.5, help="exploitability target, %% of pot")
    b.add_argument("--max-iterations", type=int, default=600)
    b.add_argument("--device", default=None, help="cuda / cpu (default: CUDA if available)")
    v = sub.add_parser("validate")
    v.add_argument("out", type=Path)
    v.add_argument("--db", help="register the dataset here if it passes")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        streets = {"river": args.river_share, "turn": 1 - args.river_share}
        cfg = DatasetConfig(args.name, args.version, args.seed, args.spots, args.shard_size,
                            {k: v for k, v in streets.items() if v > 0},
                            target_pct_pot=args.target, max_iterations=args.max_iterations)

        def progress(done: int, total: int, secs: float) -> None:
            print(f"\r{done}/{total} spots, {secs:.0f}s", end="", file=sys.stderr, flush=True)

        res = run(cfg, args.out, args.device, progress)
        print(file=sys.stderr)
        print(f"solved {res['solved_now']} spots in {res['seconds']:.0f}s; "
              f"{len(res['shards'])}/{cfg.n_shards} shards complete")
        return 0

    report = validate(args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, ensure_ascii=False))
    for e in report["errors"]:
        print("  -", e)
    if report["ok"] and args.db:
        register(args.db, args.out, report)
        print(f"registered in {args.db}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
