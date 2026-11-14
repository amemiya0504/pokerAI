"""Offline generator for the web's frozen 44-position, 169-hand preflop tables."""

from __future__ import annotations

import json
import math

from backend.agents.web_preflop import (A3_CONTEXTS, BB_OPENERS, POSITIONS, TABLE_PATH,
                                       contexts_for)
from backend.dataset.spots import CLASS_ORDER
from backend.solver.hands import CLASS_NAMES, COMBOS

# Explicit per-position calibration: opening/limping widths and raise-first-in.
# shortcut: 100 BB calibration is shared from 20–100 BB; add depth buckets after per-depth Arena evidence.
WIDTHS = {
    2: ((.80, .67), (.62, .27)),
    3: ((.56, .49), (.50, .40), (.56, .23)),
    4: ((.35, .31), (.54, .48), (.49, .39), (.55, .22)),
    5: ((.29, .26), (.36, .32), (.54, .48), (.43, .34), (.54, .21)),
    6: ((.21, .20), (.28, .26), (.35, .32), (.50, .45), (.44, .35), (.53, .20)),
    7: ((.17, .17), (.20, .19), (.26, .24), (.33, .30), (.48, .43), (.40, .32), (.52, .20)),
    8: ((.17, .17), (.20, .19), (.23, .21), (.27, .25), (.32, .29), (.48, .43), (.42, .42), (.51, .19)),
    9: ((.16, .16), (.18, .17), (.20, .19), (.21, .20), (.23, .21), (.29, .27), (.46, .42), (.38, .30), (.50, .19)),
}


BB_DEFENDS = {2: .65, 3: .35, 4: .40, 5: .40, 6: .35, 7: .30, 8: .38, 9: .30}
def opener_position_factor(n: int, opener: str) -> float:
    if opener == "SB":
        return 1.05
    if opener == "BTN":
        return 1.02
    if opener == "CO":
        return .98
    positions = POSITIONS[n]
    early = positions[:positions.index("CO")] if "CO" in positions else ()
    if opener in early:
        return .78 + .04 * early.index(opener)
    # The frozen schema includes unreachable opener contexts for old-table fallback.
    fallback = {"UTG": .78, "UTG+1": .82, "UTG+2": .86, "LJ": .90, "HJ": .94}
    if opener in fallback:
        return fallback[opener]
    raise ValueError(f"unknown opener {opener!r} for {n}-max")


def generate(a3_runtime: bool = False) -> dict:
    rank = {}
    before = 0
    for ci in CLASS_ORDER:
        rank[ci] = (before + len(COMBOS[ci]) / 2) / 1326
        before += len(COMBOS[ci])
    data = {
            "schema": 1,
            "version": "web-v7-a3-no-floor" if a3_runtime else "web-v6-a4-relative-no-floor",
            "source": ("offline heuristic calibration; A1-A3 contexts retained; no per-hand continue/raise floor; not GTO"
                       if a3_runtime else
                       "offline heuristic calibration; SB/BB contexts and BB opener positions split; no per-hand floor; not GTO"),
            "stack_bb": 100, "stack_note": "100 BB calibration, approximate at other depths; solved 5–20 BB cells take precedence",
            "actions": ["fold", "passive", "raise"], "hands": list(CLASS_NAMES), "tables": {}}
    for n, positions in POSITIONS.items():
        for pi, pos in enumerate(positions):
            width, rfi = WIDTHS[n][pi]
            for ctx in (A3_CONTEXTS if a3_runtime else contexts_for(n, pos)):
                group, price = ctx.split(":")
                defend, raise_width = width, rfi
                if group == "limped":
                    defend, raise_width = min(.85, width + .12), rfi * .8
                elif group == "option":
                    defend, raise_width = 1, rfi * .7
                elif group.startswith("vs_open"):
                    defend = BB_DEFENDS[n] if pos == "BB" else width * .60
                    if group.startswith("vs_open_early"):
                        defend *= .78
                    elif group == "vs_open_large":
                        defend *= .50
                    if group.endswith("small"):
                        defend = min(.80, defend * 1.15)
                    raise_width = min(.13, defend * .27)
                elif group.startswith("sb_vs_open_"):
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
                    raise_width = min(.13, defend * .27)
                elif group in ("sb_vs_3bet", "sb_vs_3bet_multi"):
                    defend = max(.055, width * .24)
                    if group.endswith("multi"):
                        defend *= .80
                    raise_width = min(.035, defend * .25)
                elif group.startswith("bb_vs_open_"):
                    state, opener, *suffix = group.removeprefix("bb_vs_open_").split("_")
                    if opener in BB_OPENERS:
                        sizing = suffix[-1]
                        defend = BB_DEFENDS[n] * opener_position_factor(n, opener)
                        if sizing == "large":
                            defend *= .50
                        elif sizing == "small":
                            defend = min(.80, defend * 1.15)
                        if state == "multi":
                            defend *= .75
                    else:
                        sizing = "_".join((opener, *suffix))
                        defend = BB_DEFENDS[n]
                        if sizing.startswith("early"):
                            defend *= .78
                        elif sizing == "large":
                            defend *= .50
                        elif sizing.endswith("small"):
                            defend = min(.80, defend * 1.15)
                        if state == "multi":
                            defend *= .75
                    raise_width = min(.13, defend * .27)
                elif group in ("bb_vs_3bet_single", "bb_vs_3bet_multi"):
                    defend = max(.055, BB_DEFENDS[n] * .24)
                    if group.endswith("multi"):
                        defend *= .80
                    raise_width = min(.035, defend * .25)
                elif group in ("bb_limped_single", "bb_limped_multi"):
                    defend = min(.90, width + .12)
                    raise_width = rfi * (.80 if group.endswith("single") else .60)
                elif group == "vs_3bet":
                    defend, raise_width = max(.055, width * .24), .035
                elif group == "vs_4bet":
                    defend, raise_width = .045, .015
                probs = []
                for ci in range(169):
                    # Generator-only smooth boundaries, frozen as separate cells.
                    continuing = 1 / (1 + math.exp(max(-60, min(60, (rank[ci] - defend) / .025))))
                    raising = 1 / (1 + math.exp(max(-60, min(60, (rank[ci] - raise_width) / .018))))
                    raising = min(raising, continuing)
                    fold = 1 - continuing
                    fold, raising = round(fold, 6), round(raising, 6)
                    probs.append([fold, round(1 - fold - raising, 6), raising])
                table = {
                    "raise_bb": 2.5 if n == 2 or pos in ("BTN", "CO") else 3.0,
                    "raise_multiple": 3.5 if pos in ("SB", "BB") else 3.0, "probs": probs}
                if pos == "SB" and n > 2 and group.startswith("sb_vs_open_"):
                    table["decision_note"] = "SB faces an open; BB remains unacted and may call or squeeze."
                elif pos == "SB" and n > 2 and group.startswith("sb_vs_3bet"):
                    table["decision_note"] = "SB faces a 3-bet; BB remains unacted and may still enter."
                elif pos == "BB" and group.startswith("bb_vs_open_single_"):
                    table["decision_note"] = "BB faces one opener; calling closes preflop action."
                elif pos == "BB" and group.startswith("bb_vs_open_multi_"):
                    table["decision_note"] = "BB faces an opener and caller(s); calling closes preflop action."
                elif pos == "BB" and group.startswith("bb_vs_3bet"):
                    table["decision_note"] = "BB faces a 3-bet node; caller count is explicit."
                elif pos == "BB" and group.startswith("bb_limped_"):
                    table["decision_note"] = "BB has an option after one or more limpers."
                data["tables"][f"{n}/{pos}/{ctx}"] = table
    return data


if __name__ == "__main__":
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(json.dumps(generate(a3_runtime=True), separators=(",", ":")) + "\n")
    count = sum(len(A3_CONTEXTS) for n, positions in POSITIONS.items() for pos in positions)
    print(f"wrote {TABLE_PATH}: {count} position/context tables x 169 hands")
