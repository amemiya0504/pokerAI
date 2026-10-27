"""Paired CPU comparison of web policies on identical deals."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics

import numpy as np
from pathlib import Path

from backend.agents import web_preflop, web_realtime
from backend.engine.actions import Action, ActionType
from backend.agents.style import StyleAgent
try:
    from . import web_realtime_accept as experiment
    from .preflop_exploiters import PROBES
except ImportError:  # Direct ``python scripts/experiments/web_policy_ab.py`` invocation.
    import web_realtime_accept as experiment
    from preflop_exploiters import PROBES


def interval(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / len(values) ** .5 if len(values) > 1 else float("inf")
    return {"bb_per_100": 100 * mean, "ci": [100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se)]}


def paired_delta(baseline: list[float], candidate: list[float]) -> dict:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired samples must be nonempty and have equal lengths")
    return interval([b - a for a, b in zip(baseline, candidate)])


PUBLIC_RANGE_SOURCE = web_realtime._public_ranges


def cheap_call_floor_control(policy):
    """Counterfactual: map every fold at 25% pot odds or better to a call."""
    from backend.agents.base import Decision

    def controlled(view, legal):
        decision = policy(view, legal)
        if not (legal.can_fold and legal.can_call and
                legal.call_amount / (view.pot + legal.call_amount) <= .25):
            return decision
        probs = dict(decision.action_probs or {})
        fold = probs.pop("fold", 0.0)
        probs["fold"] = 0.0
        probs["call"] = probs.get("call", 0.0) + fold
        action = Action.call() if decision.action.type is ActionType.FOLD else decision.action
        return Decision(action, decision.strategy_source, probs, decision.action_ev,
                        f"{decision.reason}; counterfactual 25% price floor control")

    return controlled


def range_source(mode: str):
    if mode == "public":
        return PUBLIC_RANGE_SOURCE
    if mode == "uniform":
        return lambda view: {
            seat.seat: np.ones(1326) for seat in view.seats if not seat.folded
        }
    raise ValueError(f"unknown equity range source: {mode}")


def opponents() -> dict:
    rows = dict(experiment.OPPONENTS)
    rows.update({name: (lambda seed, style=name: StyleAgent(seed, style))
                 for name in ("nit", "tag", "lag", "maniac")})
    rows.update(PROBES)  # v8.2 gate: iso-raise every limp / squeeze every flat
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="out/web-realtime/preflop-a0-before-floor-removal.json")
    parser.add_argument("--candidate", default=str(web_preflop.TABLE_PATH))
    parser.add_argument("--baseline-label", default="A0")
    parser.add_argument("--candidate-label", default="A1")
    parser.add_argument("--players", type=int, nargs="+", default=list(range(2, 10)))
    parser.add_argument("--opponents", nargs="+", choices=tuple(opponents()), default=list(opponents()))
    parser.add_argument("--deals", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=experiment.CPU_SEED)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--baseline-range-source", choices=("public", "uniform"), default="public")
    parser.add_argument("--candidate-range-source", choices=("public", "uniform"), default="public")
    parser.add_argument("--baseline-cheap-call-floor-control", action="store_true")
    parser.add_argument("--baseline-multiway-bluff", action="store_true")
    parser.add_argument("--candidate-multiway-bluff", action="store_true")
    parser.add_argument("--out", default="out/web-realtime/a1-paired-cpu.json")
    parser.add_argument("--stack-bb", type=float, default=100)
    parser.add_argument("--persistent-models", action="store_true",
                        help="hero keeps its opponent models across hands, as in a web session")
    parser.add_argument("--candidate-adaptive", action="store_true", help="A8.1 style adaptation for the candidate")
    parser.add_argument("--candidate-station-ranges", action="store_true",
                        help="with --candidate-adaptive: also A8's station range switch (reproduces A8)")
    args = parser.parse_args()
    if args.deals < 2 or args.seconds <= 0 or any(n not in range(2, 10) for n in args.players):
        parser.error("need at least 2 deals, 2–9 players and positive seconds")

    if not args.baseline_label or not args.candidate_label or args.baseline_label == args.candidate_label:
        parser.error("baseline and candidate labels must be distinct and nonempty")
    paths = {args.baseline_label: Path(args.baseline), args.candidate_label: Path(args.candidate)}
    for label, path in paths.items():
        if not path.is_file():
            parser.error(f"{label} table does not exist: {path}")
    config = {"deals": args.deals, "players": args.players, "opponents": args.opponents,
              "seed": args.seed, "seconds": args.seconds, "mode": "network",
              "baseline_label": args.baseline_label, "candidate_label": args.candidate_label,
              "baseline_range_source": args.baseline_range_source,
              "candidate_range_source": args.candidate_range_source,
              "baseline_cheap_call_floor_control": args.baseline_cheap_call_floor_control,
              "baseline_multiway_bluff": args.baseline_multiway_bluff,
              "candidate_multiway_bluff": args.candidate_multiway_bluff}
    if args.stack_bb != 100:  # default depth stays out of the config so earlier reports remain resumable
        config["stack_bb"] = args.stack_bb
    if args.persistent_models:
        config["persistent_models"] = True
    if args.candidate_adaptive:
        config["candidate_adaptive"] = True
    if args.candidate_station_ranges:
        config["candidate_station_ranges"] = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    report = {"config": config, "table_sha256": hashes, "pairs": {}}
    if out.exists():
        previous = json.loads(out.read_text())
        if previous.get("config") == config and previous.get("table_sha256") == hashes:
            report = previous
        else:
            parser.error(f"refusing to overwrite a different experiment in {out}; choose another --out")

    old_path = web_preflop.TABLE_PATH
    old_opponents = experiment.OPPONENTS
    old_range_source = web_realtime._public_ranges
    old_heuristic = web_realtime.heuristic
    experiment.OPPONENTS = opponents()
    try:
        for opponent in args.opponents:
            by_opponent = report["pairs"].setdefault(opponent, {})
            for n in args.players:
                if str(n) in by_opponent:
                    continue
                variants = {}
                for label, path in paths.items():
                    web_preflop.TABLE_PATH = path
                    web_preflop.tables.cache_clear()
                    mode = (args.baseline_range_source if label == args.baseline_label
                            else args.candidate_range_source)
                    web_realtime._public_ranges = range_source(mode)
                    web_realtime.heuristic = (cheap_call_floor_control(old_heuristic)
                                              if label == args.baseline_label and args.baseline_cheap_call_floor_control
                                              else old_heuristic)
                    multiway_bluff = (args.baseline_multiway_bluff if label == args.baseline_label
                                      else args.candidate_multiway_bluff)
                    result = experiment.simulate(n, opponent, args.deals, args.seed, "network", args.seconds,
                                                 log=lambda message, **kwargs: print(f"{label} {message}", **kwargs),
                                                 capture_samples=True, multiway_bluff=multiway_bluff,
                                                 stack_bb=args.stack_bb, persistent_models=args.persistent_models,
                                                 adaptive=args.candidate_adaptive and label == args.candidate_label,
                                                 station_ranges=args.candidate_station_ranges)
                    variants[label] = result
                baseline, candidate = variants[args.baseline_label], variants[args.candidate_label]
                deltas = {"overall": paired_delta(baseline["_samples"]["profit"],
                                                   candidate["_samples"]["profit"])}
                deltas["by_position"] = {
                    pos: paired_delta(baseline["_samples"]["profit_by_position"][pos],
                                      candidate["_samples"]["profit_by_position"][pos])
                    for pos in baseline["_samples"]["profit_by_position"]
                }
                by_opponent[str(n)] = {
                    args.baseline_label: {k: v for k, v in baseline.items() if k != "_samples"},
                    args.candidate_label: {k: v for k, v in candidate.items() if k != "_samples"},
                    f"{args.candidate_label}_minus_{args.baseline_label}": deltas,
                }
                out.write_text(json.dumps(report, indent=2))
                print(f"{n}max vs {opponent}: paired {args.candidate_label}-{args.baseline_label} {deltas['overall']}", flush=True)
    finally:
        experiment.OPPONENTS = old_opponents
        web_realtime._public_ranges = old_range_source
        web_realtime.heuristic = old_heuristic
        web_preflop.TABLE_PATH = old_path
        web_preflop.tables.cache_clear()
    out.write_text(json.dumps(report, indent=2))
    print(f"Wrote paired CPU report to {out}", flush=True)


if __name__ == "__main__":
    main()
