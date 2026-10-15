"""Command-line demo: random agents play hands and write JSON hand records.

    python -m backend.cli --players 6 --hands 5 --seed 42 --out out/hands.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.agents import CallingStationAgent, RandomAgent
from backend.engine.cards import derive_seed
from backend.engine.runner import run_session
from backend.engine.table import Table


def describe(record: dict) -> str:
    lines = [f"Hand #{record['hand_no']}  button=seat {record['button']}  "
             f"blinds {record['small_blind']}/{record['big_blind']}"]
    for p in record["players"]:
        lines.append(f"  seat {p['seat']} {p['player_id']:<8} {p['start_stack']:>6}  {p['hole_cards']}")
    street = None
    for a in record["actions"]:
        if a["street"] != street:
            street = a["street"]
            lines.append(f"  -- {street}")
        amount = f" {a['amount']}" if a["amount"] else ""
        lines.append(f"     seat {a['seat']}: {a['type']}{amount}")
    res = record["result"]
    lines.append(f"  board: {record['board'] or '-'}")
    for seat, sd in res["showdown"].items():
        lines.append(f"  seat {seat} shows {sd['cards']} ({sd['hand']})")
    for i, pot in enumerate(res["pots"]):
        lines.append(f"  pot {i}: {pot['amount']} -> {pot['shares']}")
    net = ", ".join(f"{s}:{v:+d}" for s, v in res["net"].items())
    lines.append(f"  net: {net}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--players", type=int, default=6, choices=range(2, 10), metavar="2-9")
    ap.add_argument("--hands", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stack", type=int, default=10_000)
    ap.add_argument("--blinds", default="50/100", help="small/big, e.g. 50/100")
    ap.add_argument("--out", type=Path, help="write JSON lines here")
    ap.add_argument("--quiet", action="store_true", help="do not print hands")
    args = ap.parse_args(argv)

    sb, bb = (int(x) for x in args.blinds.split("/"))
    table = Table(sb, bb, seed=args.seed)
    agents = {}
    for seat in range(args.players):
        if seat == 0:
            table.sit(seat, "station", args.stack)
            agents[seat] = CallingStationAgent()
        else:
            table.sit(seat, f"random{seat}", args.stack)
            agents[seat] = RandomAgent(derive_seed(args.seed, "agent", seat))

    records = run_session(table, agents, args.hands)
    if not args.quiet:
        for r in records:
            print(describe(r), end="\n\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} hands to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
