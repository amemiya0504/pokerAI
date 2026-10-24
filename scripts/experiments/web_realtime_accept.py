"""Per-position web policy validation. CPU development and held-out Arena runs.

CPU mode uses the existing CPU policy network for HU postflop, so its profit
is diagnostic only. Arena mode runs the actual HU solver, one GPU job at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from backend.agents import CallingStationAgent, RuleBasedAgent
from backend.agents.web_preflop import POSITIONS, TABLE_PATH, context as preflop_context, position
from backend.agents.web_realtime import WebRealtimeAgent
from backend.engine.actions import ActionType
from backend.engine.cards import Deck, derive_seed
from backend.engine.hand import HandState, Street

# Registered before validation. Development deals must never be called acceptance.
ARENA_SEED = 20261029
CPU_SEED = 20261022
MIN_CHEAP = 30
MAX_CHEAP_FOLD = .65
OPPONENTS = {"rule": RuleBasedAgent, "station": lambda seed: CallingStationAgent()}


def interval(values):
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / len(values) ** .5 if len(values) > 1 else float("inf")
    return {"bb_per_100": 100 * mean, "ci": [100 * (mean - 1.96 * se), 100 * (mean + 1.96 * se)]}


def rates(counts):
    out = dict(counts)
    for label, num, den in (("vpip", "vpip_hands", "hands"), ("pfr", "pfr_hands", "hands"),
                            ("three_bet", "three_bets", "three_bet_opportunities"),
                            ("bb_defense", "bb_defends", "bb_defense_opportunities"),
                            ("pf_fold", "pf_folds", "pf_fold_opportunities"),
                            ("cheap_fold_to_bet", "cheap_folds", "cheap_opportunities")):
        out[num], out[den] = counts[num], counts[den]
        out[label] = counts[num] / counts[den] if counts[den] else None
    return out


def context_rates(counts):
    total = counts["decisions"]
    return {**counts,
            **{name: counts[name] / total if total else None
               for name in ("fold", "check", "call", "raise")}}


BB_CHIPS = 100

def simulate(n, opponent, deals, seed, mode, seconds, log=print, capture_samples=False, multiway_bluff=False, deal_start=0):
    if deal_start < 0:
        raise ValueError("deal_start must be nonnegative")
    by_position = {pos: Counter() for pos in POSITIONS[n]}
    by_context = {pos: {} for pos in POSITIONS[n]}
    profits = []
    position_profits = {pos: [] for pos in POSITIONS[n]}
    sources = Counter()
    latencies = []
    bluff = Counter()
    bluff_hand_net = []
    samples = {"profit": [], "profit_by_position": {pos: [] for pos in POSITIONS[n]}, "latencies": []}
    for offset in range(deals):
        deal = deal_start + offset
        net = 0
        deal_positions = {pos: 0.0 for pos in POSITIONS[n]}
        for hero in range(n):
            h = HandState({s: 10000 for s in range(n)}, 0, BB_CHIPS // 2, BB_CHIPS,
                          Deck(derive_seed(seed, n, deal)))
            hero_position = position(h.observation(hero))
            agent = WebRealtimeAgent(derive_seed(seed, n, deal, hero), mode=mode, seconds=seconds, multiway_bluff=multiway_bluff)
            others = {s: OPPONENTS[opponent](derive_seed(seed, n, deal, hero, s)) for s in range(n) if s != hero}
            stats = by_position[position(h.observation(hero))]
            stats["hands"] += 1
            voluntary, raised = False, False
            bluff_selected = False
            raises = 0
            while not h.is_over:
                seat, legal = h.to_act, h.legal_actions()
                view = h.observation(seat)
                started = time.perf_counter()
                d = (agent if seat == hero else others[seat]).act(view, legal)
                if seat == hero and view.street is not Street.PREFLOP:
                    latency = 1000 * (time.perf_counter() - started)
                    latencies.append(latency)
                    if capture_samples:
                        samples["latencies"].append(latency)
                action = d.action.type
                if seat == hero:
                    sources[str(d.strategy_source)] += 1
                    if d.reason and "estimated bluff mix" in d.reason:
                        raise_ev = (d.action_ev or {}).get("raise", 0.0)
                        check_ev = (d.action_ev or {}).get("check", 0.0)
                        bluff["opportunities"] += 1
                        bluff["profitable_opportunities"] += raise_ev > check_ev
                        bluff["estimated_raise_ev_sum_bb"] += raise_ev / BB_CHIPS
                        bluff["estimated_check_ev_sum_bb"] += check_ev / BB_CHIPS
                        bluff["mix_sum"] += (d.action_probs or {}).get("raise", 0.0)
                        if action in (ActionType.BET, ActionType.RAISE):
                            bluff["actions"] += 1
                            bluff["selected_ev_advantage_sum_bb"] += (raise_ev - check_ev) / BB_CHIPS
                            bluff_selected = True
                    if view.street is Street.PREFLOP:
                        node = preflop_context(view, legal)
                        node_counts = by_context[hero_position].setdefault(node, Counter())
                        node_counts["decisions"] += 1
                        node_counts["fold"] += action is ActionType.FOLD
                        node_counts["check"] += action is ActionType.CHECK
                        node_counts["call"] += action is ActionType.CALL
                        node_counts["raise"] += action in (ActionType.BET, ActionType.RAISE)
                        aggressive = action in (ActionType.BET, ActionType.RAISE)
                        voluntary |= action is ActionType.CALL or aggressive
                        raised |= aggressive
                        if raises == 1 and legal.can_raise:
                            stats["three_bet_opportunities"] += 1
                            stats["three_bets"] += aggressive
                        if legal.can_call and legal.can_fold:
                            stats["pf_fold_opportunities"] += 1
                            stats["pf_folds"] += action is ActionType.FOLD
                        if position(view) == "BB" and raises == 1 and legal.can_call:
                            stats["bb_defense_opportunities"] += 1
                            stats["bb_defends"] += action is not ActionType.FOLD
                    elif legal.can_call and legal.call_amount / (view.pot + legal.call_amount) <= .25:
                        stats["cheap_opportunities"] += 1
                        stats["cheap_folds"] += action is ActionType.FOLD
                if view.street is Street.PREFLOP and action in (ActionType.BET, ActionType.RAISE):
                    raises += 1
                h.apply(d.action)
            stats["vpip_hands"] += voluntary
            stats["pfr_hands"] += raised
            hand_net = h.result().net[hero] / BB_CHIPS
            if bluff_selected:
                bluff_hand_net.append(hand_net)
            net += hand_net
            deal_positions[hero_position] = hand_net
            agent.hand_over(h.observation(hero))
        profits.append(net / n)
        if capture_samples:
            samples["profit"].append(net / n)
        for pos, value in deal_positions.items():
            position_profits[pos].append(value)
            if capture_samples:
                samples["profit_by_position"][pos].append(value)
        if (offset + 1) % 20 == 0:
            log(f"{n}max vs {opponent}: {deal + 1}/{deal_start + deals} deals", flush=True)
    result = {"positions": {p: rates(c) for p, c in by_position.items()}, "profit": interval(profits),
              "profit_by_position": {p: interval(v) for p, v in position_profits.items()},
              "preflop_contexts": {p: {node: context_rates(c) for node, c in nodes.items()}
                                   for p, nodes in by_context.items()},
              "sources": dict(sources), "deals": deals, "hands": deals * n,
              "latency_postflop_p95_ms": sorted(latencies)[min(len(latencies) - 1, int(.95 * len(latencies)))] if latencies else None,
              "bluff": {"opportunities": bluff["opportunities"],
                        "profitable_opportunities": bluff["profitable_opportunities"],
                        "actions": bluff["actions"],
                        "sampled_frequency": bluff["actions"] / bluff["opportunities"] if bluff["opportunities"] else None,
                        "mean_strategy_frequency": bluff["mix_sum"] / bluff["opportunities"] if bluff["opportunities"] else None,
                        "estimated_raise_ev_bb_per_opportunity": bluff["estimated_raise_ev_sum_bb"] / bluff["opportunities"] if bluff["opportunities"] else None,
                        "estimated_check_ev_bb_per_opportunity": bluff["estimated_check_ev_sum_bb"] / bluff["opportunities"] if bluff["opportunities"] else None,
                        "estimated_raise_advantage_bb_per_opportunity": (bluff["estimated_raise_ev_sum_bb"] - bluff["estimated_check_ev_sum_bb"]) / bluff["opportunities"] if bluff["opportunities"] else None,
                        "estimated_raise_advantage_bb_when_selected": bluff["selected_ev_advantage_sum_bb"] / bluff["actions"] if bluff["actions"] else None,
                        "realized_hand_net_bb_when_bluff_selected": statistics.fmean(bluff_hand_net) if bluff_hand_net else None,
                        "hands_with_bluff": len(bluff_hand_net)}}
    if capture_samples:
        result["_samples"] = samples
    return result


def gate(rows, phase):
    details = {}
    for n, opponents in rows.items():
        rule, station = opponents["rule"], opponents["station"]
        # Free BB checks are not fold opportunities. Check the union of opponents.
        positions = {}
        for pos in rule["positions"]:
            cells = [r["positions"][pos] for r in opponents.values()]
            folds = sum(c.get("pf_folds", 0) for c in cells)
            opportunities = sum(c.get("pf_fold_opportunities", 0) for c in cells)
            cheap = sum(c.get("cheap_opportunities", 0) for c in cells)
            cheap_folds = sum(c.get("cheap_folds", 0) for c in cells)
            positions[pos] = {"nondegenerate_fold": 0 < folds < opportunities,
                              "cheap_samples": cheap, "cheap_fold_rate": cheap_folds / cheap if cheap else None,
                              "cheap_gate": cheap >= MIN_CHEAP and cheap_folds / cheap <= MAX_CHEAP_FOLD}
        details[n] = {"positions": positions, "rule_not_significantly_negative": rule["profit"]["ci"][1] >= 0,
                      "station_positive": station["profit"]["bb_per_100"] > 0,
                      "station_significantly_positive": station["profit"]["ci"][0] > 0}
    passed = (phase == "arena" and all(
        row["rule_not_significantly_negative"] and row["station_positive"] and
        all(p["nondegenerate_fold"] and p["cheap_gate"] for p in row["positions"].values())
        for row in details.values()))
    return {"passed": passed, "details": details, "note": "CPU runs are diagnostic, never final acceptance"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cpu", "arena"), default="cpu")
    parser.add_argument("--deals", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--players", type=int, nargs="+", default=list(range(2, 10)))
    parser.add_argument("--out", default="out/web-realtime/cpu.json")
    args = parser.parse_args()
    if args.deals < 2 or any(n not in range(2, 10) for n in args.players) or args.seconds <= 0:
        parser.error("need at least 2 deals, 2–9 players and positive seconds")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed = CPU_SEED if args.phase == "cpu" else ARENA_SEED
    table_hash = hashlib.sha256(TABLE_PATH.read_bytes()).hexdigest()
    report = {"config": vars(args), "seed": seed, "table_sha256": table_hash, "rows": {}}
    if out.exists():
        previous = json.loads(out.read_text())
        if (previous.get("config") == vars(args) and previous.get("seed") == seed
                and previous.get("table_sha256") == table_hash):
            report = previous
    for n in args.players:
        opponents = report["rows"].setdefault(str(n), {})
        for opponent in OPPONENTS:
            if opponent in opponents:
                continue
            row = simulate(n, opponent, args.deals, report["seed"],
                           "network" if args.phase == "cpu" else "solve", args.seconds)
            opponents[opponent] = row
            out.write_text(json.dumps(report, indent=2))
            print(f"{n}max vs {opponent}: {row['profit']}", flush=True)
    report["acceptance"] = gate(report["rows"], args.phase)
    report["acceptance"]["complete_player_coverage"] = set(args.players) == set(range(2, 10))
    report["acceptance"]["passed"] &= report["acceptance"]["complete_player_coverage"]
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["acceptance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
