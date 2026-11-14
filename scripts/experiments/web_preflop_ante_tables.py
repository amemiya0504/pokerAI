"""v8.3 with wider ranges for tables with a 0.5 BB ante.

Only open, limped, option, vs-open (SB/BB included) and BB-limped nodes change: their continue and raise
widths grow by ((1.5 + 0.5 n) / 1.5) ** k, the dead money per player count relative to blinds alone.
Sizes and every other node stay v8.3's. The second round (``defense=True``) also widens the calls of the vs-3-bet
and cheap vs-4-bet nodes those extra opens run into. Run from the repo root:

    python -m scripts.experiments.web_preflop_ante_tables
"""

from __future__ import annotations

from pathlib import Path

from backend.agents.web_preflop_v8 import validate
from scripts.experiments.web_preflop_v8_tables import (VARIANTS, generate, heuristic_table, load_equity, ranks, write)

K_CANDIDATES = (.15, .25, .35)
ANTE_BB = .5
WIDENED_FAMILIES = ("unopened", "vs_open")
CONTINUE_CAP = .85
VS_OPEN_RAISE_CAP = .13
TABLE_DIR = Path("backend/agents/data")


def table_path(k: float, defense: bool = False) -> Path:
    return TABLE_DIR / f"web_preflop_v8_3_ante05{'d' if defense else ''}_k{round(k * 100):02d}.json.gz"


def width_factor(n: int, k: float) -> float:
    return ((1.5 + ANTE_BB * n) / 1.5) ** k


def widen_for(n: int, k: float):
    factor = width_factor(n, k)

    def widen(family, defend, raise_width):
        if family not in WIDENED_FAMILIES:
            return defend, raise_width
        wider = max(defend, min(CONTINUE_CAP, defend * factor))  # never narrower, even above the cap
        raised = raise_width * factor
        if family == "vs_open":
            raised = max(raise_width, min(VS_OPEN_RAISE_CAP, raised))
        return wider, min(wider, raised)

    return widen


def widen_calls_for(n: int, k: float):
    factor = width_factor(n, k)

    def widen(_family, defend, raise_width):
        return max(defend, min(CONTINUE_CAP, defend * factor)), raise_width

    return widen


def is_defense(key: str) -> bool:
    """A node the wider opens run into: facing a 3-bet at any size, or a cheap 4-bet."""
    group, price = key.rsplit("/", 1)[1].split(":")
    if group.startswith("vs_3bet_"):
        return price in ("normal", "cheap")
    return group in ("vs_4bet_early", "vs_4bet_late") and price == "cheap"


def is_widened(key: str) -> bool:
    group, price = key.rsplit("/", 1)[1].split(":")
    if price not in ("normal", "cheap"):
        return False
    return group in ("open", "limped", "option") or group.startswith(
        ("vs_open_", "sb_vs_open_", "bb_vs_open_", "bb_limped_"))


def changed_by(defense: bool):
    return (lambda key: is_widened(key) or is_defense(key)) if defense else is_widened


def generate_ante(E, W, k: float, base: dict | None = None, defense: bool = False) -> dict:
    base = base or generate(E, W, "v8.3")
    v = VARIANTS["v8.3"]
    rank = ranks()
    tables = dict(base["tables"])
    for key in filter(changed_by(defense), base["tables"]):
        n, depth, rest = key.split("/", 2)  # the heads-up position itself contains a slash ("BTN/SB")
        pos, node = rest.rsplit("/", 1)
        calls_only = is_defense(key)
        row = heuristic_table(rank, int(n), pos, node.split(":")[0], int(depth), v["depth_adj"], v["open_cap"],
                              top_never_folds=calls_only, widen=(widen_calls_for if calls_only else widen_for)(int(n), k))
        tables[key] = {**base["tables"][key], "probs": row["probs"]}
    also = "; vs-3-bet and cheap vs-4-bet calls too" if defense else ""
    return {**base, "tables": tables, "version": f"web-v8.3-ante0.5{'-defense' if defense else ''}-k{k:g}", "ante_k": k,
            "source": base["source"].removesuffix("; not GTO")
            + f"; open, limped, option, vs-open and BB-limped widths x ((1.5+0.5n)/1.5)^{k:g} for a 0.5 BB ante{also}; not GTO"}


def static_problems(base: dict, candidate: dict, defense: bool = False) -> list[str]:
    """Why the candidate breaks the acceptance's static rules; empty when it does not."""
    if set(base["tables"]) != set(candidate["tables"]):
        return ["table keys differ"]
    problems, changed = [], 0
    for key, old in base["tables"].items():
        new = candidate["tables"][key]
        if not changed_by(defense)(key):
            if new != old:
                problems.append(f"{key}: an unmodified node changed")
            continue
        if {k: v for k, v in new.items() if k != "probs"} != {k: v for k, v in old.items() if k != "probs"}:
            problems.append(f"{key}: sizing changed")
        changed += new["probs"] != old["probs"]
        for hand, (was, now) in enumerate(zip(old["probs"], new["probs"])):
            if 1 - now[0] < 1 - was[0] - 1e-9 or now[2] < was[2] - 1e-9:
                problems.append(f"{key}: hand {hand} continues or raises less than v8.3")
                break
    return problems if changed else ["no node changed"]


def main() -> None:
    E, W = load_equity()
    base = generate(E, W, "v8.3")
    for defense in (False, True):
        for k in K_CANDIDATES:
            data = generate_ante(E, W, k, base, defense)
            validate(data)
            problems = static_problems(base, data, defense)
            if problems:
                raise SystemExit(f"k={k:g}: {len(problems)} static failures, first: {problems[0]}")
            path = table_path(k, defense)
            write(data, path)
            print(f"wrote {path}: {sum(map(changed_by(defense), data['tables']))} changed of {len(data['tables'])}", flush=True)


if __name__ == "__main__":
    main()
