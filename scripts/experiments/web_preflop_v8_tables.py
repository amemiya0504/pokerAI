"""Offline generator for the web's frozen v8 / v8.1 / v8.2 / v8.3 preflop tables (schema 2).

    python -m scripts.experiments.web_preflop_v8_tables equity   # one-off: data/preflop/hu_equity_169.npz
    python -m scripts.experiments.web_preflop_v8_tables          # writes the v8, v8.1, v8.2 and v8.3 .json.gz tables

Non-all-in nodes reuse the v7 width calibration with per-depth multipliers (HEURISTIC). All-in nodes
store each hand's equity against an assumed jam range, computed from the 169x169 class equity matrix
with card-removal weights; the runtime compares it with the actual price.

v8 failed its 40 BB gate vs station because of its short-stack open sizes and unopened / vs-open range
multipliers; v8.1 keeps v7's sizes and ranges there at every depth and keeps the new 3-bet / 4-bet /
5-bet / all-in nodes.

v8.2 = v8.1 plus capped-range defence nodes: v7's limp ranges are the weakest hands and its vs-open
tables fold all of them to an iso-raise; a flatter facing a squeeze used the cold 3-bet table and
folded nearly every flat. The new nodes rank each hand within the range that reached the node.

v8.3 = v8.2 with two fixes to the vs 3-bet / cold 3-bet / vs 4-bet / vs 5-bet nodes. Their continuing
ranges are narrow next to the logistic smoothing width, so even AA folded 4–31% of the time; now AA, KK,
QQ and the stronger half of each continuing range never fold. They also ignored the price (a min 3-bet used the
same row as a 3x one); now they store each hand's equity vs an assumed raiser range and the runtime
also continues when that equity, discounted for realisation, covers the actual price.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

from backend.agents.web_preflop import POSITIONS
from backend.agents.web_preflop_v8 import (ALLIN_GROUPS, DEFEND_GROUPS, DEPTHS, RAISE_GROUPS, SCHEMA, groups_for,
                                          side)
from backend.dataset.spots import CLASS_ORDER
from backend.solver.hands import CLASS_NAMES, COMBOS
from scripts.experiments.web_preflop_tables import BB_DEFENDS, WIDTHS, opener_position_factor

V8_PATH = Path("backend/agents/data/web_preflop_v8.json.gz")
EQUITY_PATH = Path("data/preflop/hu_equity_169.npz")
JAM_FRACTION = .40

# (continue multiplier, raise-width multiplier) per depth bucket and node family.
# shortcut: hand-set depth multipliers; replace with per-depth solver or Arena-calibrated values.
DEPTH_ADJ = {
    "unopened": {20: (.90, .90), 40: (.95, .95), 60: (1, 1), 100: (1, 1), 150: (1, 1)},
    "vs_open": {20: (.80, 1.10), 40: (.90, 1.05), 60: (.95, 1), 100: (1, 1), 150: (1.05, .95)},
    "vs_3bet": {20: (1.15, 2.2), 40: (1.05, 1.4), 60: (1, 1.15), 100: (1, 1), 150: (.92, .90)},
    "vs_4bet": {20: (1.5, 3.0), 40: (1.4, 2.5), 60: (1.2, 1.6), 100: (1, 1), 150: (.90, .90)},
}
# Open size (BB) cap per depth; v7's 100 BB sizes apply from 100 BB.
OPEN_CAP = {20: 2.0, 40: 2.2, 60: 2.3, 100: 99, 150: 99}
# Jam-range width (fraction of combos) at 100 BB and 20 BB; geometric interpolation in between.
JAM_100 = {"unopened": .025, "after_open": .045, "after_3bet": .06, "cold": .03}
JAM_20 = {"after_open": .16, "after_3bet": .22, "cold": .10}  # "unopened" uses the jammer's open width
LATE_JAM = 1.35
# v8.2 defence: share of MDF this player defends, and the re-raise width (fraction of all combos).
# shortcut: hand-set; replace with solver frequencies for capped ranges.
DEFEND_SCALE = {"limp_vs_raise": .75, "flat_vs_squeeze": .60}
DEFEND_RAISE_WIDTH = .025
# v8.3 priced raise nodes: raiser range width (fraction of combos) at 100 BB per node kind, scaled by the
# depth raise-width multiplier of the node the raiser acted in and by LATE_JAM for a late raiser.
# shortcut: hand-set; replace with the ranges opponents are observed to raise with.
RAISER_100 = {"vs_3bet": .08, "cold_vs_3bet": .06, "vs_4bet": .035, "vs_4bet_cold": .035, "vs_5bet": .025}
RAISER_FAMILY = {"vs_3bet": "vs_open", "cold_vs_3bet": "vs_open", "vs_4bet": "vs_3bet",
                 "vs_4bet_cold": "vs_3bet", "vs_5bet": "vs_4bet"}
# v8.3 raise nodes never fold AA, KK, QQ (this share of combos), nor the stronger half of the continuing range.
PREMIUM_SHARE = 18 / 1326
_FLAT = {d: (1, 1) for d in DEPTHS}
VARIANTS = {
    "v8": {"path": V8_PATH, "version": "web-v8-depth-allin", "depth_adj": DEPTH_ADJ, "open_cap": OPEN_CAP,
           "source": ("offline heuristic calibration (v7 widths x per-depth multipliers); vs 3-bet/4-bet/5-bet nodes; "
                      "all-in nodes from class equity vs assumed jam ranges; not GTO")},
    "v8.1": {"path": Path("backend/agents/data/web_preflop_v8_1.json.gz"), "version": "web-v8.1-depth-allin",
             "depth_adj": {**DEPTH_ADJ, "unopened": _FLAT, "vs_open": _FLAT}, "open_cap": {d: 99 for d in DEPTHS},
             "source": ("offline heuristic calibration; unopened / vs-open ranges and open sizes as v7 at every depth; "
                        "per-depth multipliers only on vs 3-bet/4-bet/5-bet nodes; all-in nodes from class equity "
                        "vs assumed jam ranges; not GTO")},
}
VARIANTS["v8.2"] = {**VARIANTS["v8.1"], "path": Path("backend/agents/data/web_preflop_v8_2.json.gz"),
                    "version": "web-v8.2-capped-defence", "defend": True,
                    "source": VARIANTS["v8.1"]["source"].removesuffix("; not GTO")
                    + "; limper-vs-raise and flatter-vs-squeeze nodes defend an MDF share of the capped range; not GTO"}
VARIANTS["v8.3"] = {**VARIANTS["v8.2"], "path": Path("backend/agents/data/web_preflop_v8_3.json.gz"),
                    "version": "web-v8.3-priced-raises", "priced_raises": True,
                    "source": VARIANTS["v8.2"]["source"].removesuffix("; not GTO")
                    + ("; vs 3-bet/4-bet/5-bet nodes never fold the stronger half of their continuing range and also "
                       "continue when equity vs an assumed raiser range covers the actual price; not GTO")}


def ranks() -> list[float]:
    rank, before = [0.0] * 169, 0
    for ci in CLASS_ORDER:
        rank[ci] = (before + len(COMBOS[ci]) / 2) / 1326
        before += len(COMBOS[ci])
    return rank


def smooth(rank: float, width: float, scale: float) -> float:
    return 1 / (1 + math.exp(max(-60, min(60, (rank - width) / scale))))


def base_widths(n: int, pos: str, group: str) -> tuple[float, float, str]:
    """(continue width, raise width, family) — v7 formulas for v7 groups, new ones for v8 nodes."""
    width, rfi = WIDTHS[n][POSITIONS[n].index(pos)]
    if group == "open":
        return width, rfi, "unopened"
    if group == "limped":
        return min(.85, width + .12), rfi * .8, "unopened"
    if group == "option":
        return 1, rfi * .7, "unopened"
    if group.startswith("vs_open"):
        defend = BB_DEFENDS[n] if pos == "BB" else width * .60
        if group.startswith("vs_open_early"):
            defend *= .78
        elif group == "vs_open_large":
            defend *= .50
        if group.endswith("small"):
            defend = min(.80, defend * 1.15)
        return defend, min(.13, defend * .27), "vs_open"
    if group.startswith("sb_vs_open_"):
        state = group.removeprefix("sb_vs_open_")
        multiway = state.startswith("multi_")
        state = state.removeprefix("multi_")
        defend = width * .60
        if state.startswith("early"):
            defend *= .78
        elif state == "large":
            defend *= .50
        elif state.endswith("small"):
            defend = min(.80, defend * 1.15)
        if multiway:
            defend *= .78
        return defend, min(.13, defend * .27), "vs_open"
    if group.startswith("bb_vs_open_"):
        state, *rest = group.removeprefix("bb_vs_open_").split("_")
        sizing = "_".join(rest)
        defend = BB_DEFENDS[n]
        if sizing.startswith("early"):
            defend *= .78
        elif sizing == "large":
            defend *= .50
        elif sizing.endswith("small"):
            defend = min(.80, defend * 1.15)
        if state == "multi":
            defend *= .75
        return defend, min(.13, defend * .27), "vs_open"
    if group in ("bb_limped_single", "bb_limped_multi"):
        return min(.90, width + .12), rfi * (.80 if group.endswith("single") else .60), "unopened"
    if group.startswith("vs_3bet_"):
        _, _, who, size = group.split("_")
        defend = max(.055, width * .24) * (1.25 if who == "late" else 1) * (.80 if size == "large" else 1)
        return defend, min(.045, defend * .35), "vs_3bet"
    if group == "cold_vs_3bet":
        return .03, .018, "vs_3bet"
    if group.startswith("vs_4bet_") and group != "vs_4bet_cold":
        defend = .045 * (1.20 if group.endswith("late") else 1)
        return defend, .015, "vs_4bet"
    if group == "vs_4bet_cold":
        return .025, .015, "vs_4bet"
    if group == "vs_5bet":
        return .03, .02, "vs_4bet"
    raise ValueError(f"unknown group {group!r}")


def sizing(n: int, pos: str, group: str, depth: int, open_cap: dict) -> dict:
    raise_bb = min(open_cap[depth], 2.5 if n == 2 or pos in ("BTN", "CO") else 3.0)
    if group.startswith(("vs_open", "sb_vs_open", "bb_vs_open")):
        multiple = 3.5 if pos in ("SB", "BB") else 3.0
    elif "3bet" in group:
        multiple = 2.3
    elif group.startswith(("vs_4bet", "vs_5bet")) or group == "flat_vs_squeeze":
        multiple = 2.2
    else:
        multiple = 3.5 if pos in ("SB", "BB") else 3.0
    return {"raise_bb": raise_bb, "raise_multiple": multiple, "jam_fraction": JAM_FRACTION}


def heuristic_table(rank, n, pos, group, depth, depth_adj: dict, open_cap: dict, top_never_folds: bool = False) -> dict:
    defend, raise_width, family = base_widths(n, pos, group)
    dm, rm = depth_adj[family][depth]
    defend = min(1, defend * dm)
    raise_width = min(defend, raise_width * rm)
    probs = []
    for ci in range(169):
        continuing = smooth(rank[ci], defend, .025)
        if top_never_folds and rank[ci] <= max(defend / 2, PREMIUM_SHARE):
            continuing = 1.0
        raising = min(smooth(rank[ci], raise_width, .018), continuing)
        fold, raising = round(1 - continuing, 4), round(raising, 4)
        probs.append([fold, round(1 - fold - raising, 4), raising])
    return {**sizing(n, pos, group, depth, open_cap), "probs": probs}


def jam_width(n: int, depth: int, jammer_side: str, state: str) -> float:
    if state == "unopened":
        seats = [p for p in POSITIONS[n] if p != "BB" and side(n, p) == jammer_side] or [POSITIONS[n][0]]
        w20 = float(np.mean([WIDTHS[n][POSITIONS[n].index(p)][1] for p in seats]))
    else:
        w20 = JAM_20[state]
    t = min(1, max(0, math.log(depth / 20) / math.log(5)))
    width = w20 ** (1 - t) * JAM_100[state] ** t
    if depth > 100:
        width *= .9
    if jammer_side == "late" and state != "unopened":  # open-jam widths already come from late seats
        width *= LATE_JAM
    return min(.6, width)


def range_equity(rank, E, W, width: float) -> list[float]:
    """Each class's equity vs the top ``width`` of combos (smoothed), with card-removal weights."""
    opp = np.array([smooth(rank[j], width, .025) for j in range(169)])
    mass = W * opp[None, :]
    equity = (E * mass).sum(axis=1) / np.maximum(mass.sum(axis=1), 1e-12)
    return [round(float(x), 4) for x in equity]


def allin_table(rank, E, W, n, depth, group) -> dict:
    _, _, jammer_side, *state = group.split("_")
    state = "_".join(state)
    width = jam_width(n, depth, jammer_side, state)
    return {"equity": range_equity(rank, E, W, width), "jam_width": round(width, 4),
            "decision_note": ("all-in node: call iff equity vs an assumed jam range covers the actual price; "
                              "jam range is HEURISTIC, multiway is approximated")}


def raiser_width(group: str, depth: int, depth_adj: dict) -> float:
    """Assumed range of the player whose raise this node faces (vs 3-bet / cold 3-bet / vs 4-bet / vs 5-bet)."""
    kind, raiser_side = group, None
    if group.startswith("vs_3bet_"):
        kind, raiser_side = "vs_3bet", group.split("_")[2]
    elif group.startswith("vs_4bet_") and group != "vs_4bet_cold":
        kind, raiser_side = "vs_4bet", group.split("_")[2]
    width = RAISER_100[kind] * depth_adj[RAISER_FAMILY[kind]][depth][1]
    if raiser_side == "late":
        width *= LATE_JAM
    return min(.6, width)


def defend_table(rank, strength, n, pos, group, depth, open_cap, arrival: np.ndarray) -> dict:
    """Percentile of each hand within the arrival range (0 = best), combo-weighted by equity vs a random hand."""
    w = np.array([len(c) for c in COMBOS]) * arrival
    total, before, percentile = max(float(w.sum()), 1e-12), 0.0, [0.0] * 169
    for ci in sorted(range(169), key=lambda c: -strength[c]):
        percentile[ci] = round(min(1.0, (before + w[ci] / 2) / total), 4)
        before += w[ci]
    return {**sizing(n, pos, group, depth, open_cap), "percentile": percentile,
            "raise": [round(smooth(rank[ci], DEFEND_RAISE_WIDTH, .008), 4) for ci in range(169)],
            "mdf_scale": DEFEND_SCALE[group],
            "decision_note": ("capped-range node: continue with the best mdf_scale x MDF share of the range that "
                              "reached it; HEURISTIC")}


def arrival_range(tables: dict, prefix: str, group: str) -> np.ndarray:
    """Passive-action weights that bring a hand to the node: an open-limp, or a flat of any open.

    shortcut: over-limpers use the open-limp ranking too (their stronger hands rank as top of range).
    """
    passive = lambda g: np.array([row[1] for row in tables[f"{prefix}/{g}:normal"]["probs"]])
    if group == "limp_vs_raise":
        return passive("open")
    opens = [k.removeprefix(prefix + "/").removesuffix(":normal") for k in tables
             if k.startswith(prefix + "/") and k.endswith(":normal")
             and k.removeprefix(prefix + "/").startswith(("vs_open", "sb_vs_open", "bb_vs_open"))]
    return np.mean([passive(g) for g in opens], axis=0)


def generate(E: np.ndarray, W: np.ndarray, variant: str = "v8.1") -> dict:
    v = VARIANTS[variant]
    rank = ranks()
    data = {"schema": SCHEMA, "version": v["version"], "source": v["source"],
            "depths_bb": list(DEPTHS),
            "stack_note": "effective-stack buckets <30, <50, <80, <125, >=125 BB; solved 5–20 BB push/fold cells take precedence",
            "actions": ["fold", "passive", "raise"], "hands": list(CLASS_NAMES), "tables": {}}
    if v.get("defend"):
        data["defend_groups"] = list(DEFEND_GROUPS)
        strength = (E * W).sum(axis=1) / W.sum(axis=1)  # HU equity vs a random hand
    priced = v.get("priced_raises", False)
    if priced:
        data["priced_raises"] = True
    for n, positions in POSITIONS.items():
        for depth in DEPTHS:
            allin = {g: allin_table(rank, E, W, n, depth, g) for g in ALLIN_GROUPS}
            raiser = {g: raiser_width(g, depth, v["depth_adj"]) for g in RAISE_GROUPS} if priced else {}
            equity = {g: range_equity(rank, E, W, w) for g, w in raiser.items()}
            for pos in positions:
                for group in groups_for(n, pos):
                    # Both price keys are kept for v7 parity; v8.3 raise nodes price calls at runtime instead.
                    table = heuristic_table(rank, n, pos, group, depth, v["depth_adj"], v["open_cap"],
                                            top_never_folds=group in raiser)
                    if group in raiser:
                        table |= {"equity": equity[group], "raiser_width": round(raiser[group], 4)}
                    for price in ("normal", "cheap"):
                        data["tables"][f"{n}/{depth}/{pos}/{group}:{price}"] = table
                for group, table in allin.items():
                    data["tables"][f"{n}/{depth}/{pos}/{group}:allin"] = table
                for group in (DEFEND_GROUPS if v.get("defend") else ()):
                    prefix = f"{n}/{depth}/{pos}"
                    data["tables"][f"{prefix}/{group}:mdf"] = defend_table(
                        rank, strength, n, pos, group, depth, v["open_cap"], arrival_range(data["tables"], prefix, group))
    return data


def load_equity(path: Path = EQUITY_PATH) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["E"], z["W"]


def write(data: dict, path: Path) -> None:
    raw = (json.dumps(data, separators=(",", ":")) + "\n").encode()
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))


def main() -> None:
    if sys.argv[1:] == ["equity"]:
        from backend.solver.pushfold import hu_equity_matrix

        # The Monte Carlo streams depend on the worker split, so the worker count is fixed and recorded.
        E, W = hu_equity_matrix(samples=2000, seed=0, workers=24)
        EQUITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EQUITY_PATH, E=E, W=W, samples=2000, seed=0, workers=24)
        print(f"wrote {EQUITY_PATH}")
        return
    E, W = load_equity()
    for variant, v in VARIANTS.items():
        data = generate(E, W, variant)
        write(data, v["path"])
        print(f"wrote {v['path']}: {len(data['tables'])} tables")


if __name__ == "__main__":
    main()
