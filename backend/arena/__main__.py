"""Run an AI-vs-AI duplicate arena.

    python -m backend.arena rule random station --deals 2000 --seed 1 --out out/arena/run1

Agents are ``kind`` or ``name=kind`` (kinds: random, station, rule); repeat a
kind with distinct names, e.g. ``a=rule b=rule``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents import REGISTRY, AgentSpec
from .runner import ArenaConfig, run_arena
from .storage import git_commit, save_run, write_parquet


def parse_agents(items: list[str]) -> tuple[AgentSpec, ...]:
    specs = []
    for item in items:
        name, _, kind = item.partition("=")
        kind = kind or name
        if kind not in REGISTRY:
            raise SystemExit(f"unknown agent kind {kind!r} (known: {', '.join(sorted(REGISTRY))})")
        specs.append(AgentSpec(name, kind))
    return tuple(specs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.arena", description=__doc__.splitlines()[0])
    ap.add_argument("agents", nargs="+", help="2-9 agents: kind or name=kind")
    ap.add_argument("--deals", type=int, default=1000, help="deck seeds; hands = deals x agents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stack-bb", type=int, default=100)
    ap.add_argument("--blinds", default="50/100")
    ap.add_argument("--workers", type=int, default=None, help="default: all CPUs")
    ap.add_argument("--boundary", choices=["inline", "process"], default="inline",
                    help="process: run each agent in its own process (slower, fully isolated)")
    ap.add_argument("--out", type=Path, help="write hands.parquet, players.parquet, summary.json")
    ap.add_argument("--db", help="also record the run in this database URL")
    args = ap.parse_args(argv)

    sb, bb = (int(x) for x in args.blinds.split("/"))
    cfg = ArenaConfig(parse_agents(args.agents), args.deals, args.seed, args.stack_bb, sb, bb,
                      args.boundary)

    def progress(done: int, total: int) -> None:
        print(f"\r{done}/{total} deals", end="", file=sys.stderr, flush=True)

    result = run_arena(cfg, args.workers, progress=progress)
    print(file=sys.stderr)
    commit = git_commit()

    print(f"{len(result.hand_rows):,} hands in {result.seconds:.1f}s "
          f"({result.hands_per_second:,.0f} hands/s, {result.workers} workers), seed {cfg.seed}")
    print(f"{'agent':<12}{'bb/100':>10}{'95% CI':>24}{'sd':>10}")
    for s in sorted(result.stats, key=lambda s: -s.bb_per_100):
        mark = " *" if s.significant else ""
        print(f"{s.agent:<12}{s.bb_per_100:>10.2f}   [{s.ci_low:>8.2f}, {s.ci_high:>8.2f}]{s.stddev:>10.1f}{mark}")
    print("* interval excludes 0")
    if args.out:
        write_parquet(result, args.out, commit)
        print(f"wrote {args.out}")
    if args.db:
        run_id = save_run(args.db, result, commit, args.out)
        print(f"recorded simulation run {run_id} in {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
