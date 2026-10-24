"""Opponent modeling (Milestone 8).

    python -m backend.opponent accept --out out/opponent/acceptance.json
    python -m backend.opponent exploit-accept            # M9 (hours: real-time agents)
    python -m backend.opponent exploit-spots             # M9 exact gains / costs on subgames
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.opponent", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("accept", help="M8 acceptance: posteriors vs known true frequencies")
    a.add_argument("--seeds", type=int, default=20)
    a.add_argument("--reference-hands", type=int, default=200_000)
    a.add_argument("--workers", type=int)
    a.add_argument("--seed-tag", default="v2", help='"v1" reproduces the first run')
    a.add_argument("--gate-version", type=int, default=2, choices=[1, 2])
    a.add_argument("--out", type=Path, help="write the report (with every estimate) as JSON")
    x = sub.add_parser("exploit-accept", help="M9 acceptance: exploit vs GTO baseline in the arena")
    x.add_argument("--deals", type=int, default=200)
    x.add_argument("--seconds", type=float, default=2.0)
    x.add_argument("--workers", type=int, default=1)
    x.add_argument("--out", type=Path, default=Path("out/m9"))
    sp = sub.add_parser("exploit-spots", help="M9 exact gain and cost of exploiting on subgames")
    sp.add_argument("--spots", type=int, default=30)
    sp.add_argument("--out", type=Path, default=Path("out/m9/spots.json"))
    sp.add_argument("--absolute", action="store_true", help="the first method: absolute frequencies")
    args = ap.parse_args(argv)

    if args.cmd == "exploit-accept":
        from .m9 import run as run_m9

        report = run_m9(args.out, args.deals, args.seconds, args.workers)
        print(json.dumps({k: report[k] for k in ("E1_fixed_opponents", "E2_vs_gto", "passed",
                                                 "significant_positive", "significant_negative")}, indent=2))
        return 0 if report["passed"] else 1
    if args.cmd == "exploit-spots":
        from .m9_spots import run as run_spots

        print(json.dumps(run_spots(args.spots, args.out, relative=not args.absolute), indent=2))
        return 0

    from .acceptance import run

    t0 = time.perf_counter()
    report = run(args.seeds, args.reference_hands, args.workers, seed_tag=args.seed_tag,
                 gate_version=args.gate_version)
    report["seconds"] = time.perf_counter() - t0
    g = report["gates"]
    print(json.dumps({k: v for k, v in g.items()}, indent=2, default=str))
    for s, v in report["per_subject"].items():
        line = " ".join(f"{n}: cov {x['coverage']:.2f} mae {x['median_abs_error']:.3f}" for n, x in v["exact"].items())
        ref = " ".join(f"{n}: cov {x['coverage']:.2f} mae {x['median_abs_error']:.3f}" for n, x in v["reference"].items())
        print(f"{s:8s} exact [{line}] | reference [{ref}]", file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, default=str))
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
