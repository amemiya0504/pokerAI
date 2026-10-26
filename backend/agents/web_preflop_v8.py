"""Frozen web preflop v8 (schema 2): stack-depth buckets and explicit vs 3-bet / 4-bet / all-in nodes.

Runtime only reads the frozen table. Keys are ``{n}/{depth}/{position}/{group}:{price}``; all-in
nodes use the price token ``allin`` and store per-hand equity against an assumed jam range, so the
call/fold decision uses the actual pot odds of the hand being played.

v8.2 tables (``defend_groups`` present) add two ``:mdf`` nodes for a player whose range is capped by
an earlier passive action: a limper facing a raise and a flatter facing a squeeze. They store each
hand's percentile within that capped range and continue with the best MDF-scaled fraction of it.
"""

from __future__ import annotations

import math

import numpy as np

from backend.engine.actions import ActionType, LegalActions
from backend.engine.hand import EventType, PlayerView, Street

from .web_preflop import BB_A3_GROUPS, OPEN_GROUPS, POSITIONS, SB_GROUPS, context as v7_context
from .web_preflop import effective_bb, position

SCHEMA = 2
DEPTHS = (20, 40, 60, 100, 150)
# Upper effective-stack bound (BB, exclusive) of each bucket except the last.
DEPTH_BOUNDS = (30, 50, 80, 125)
SIDES = ("early", "late")
BASE_GROUPS = ("open", "limped", "option", *OPEN_GROUPS)
RAISE_GROUPS = (*(f"vs_3bet_{side}_{size}" for side in SIDES for size in ("small", "large")),
                "cold_vs_3bet", *(f"vs_4bet_{side}" for side in SIDES), "vs_4bet_cold", "vs_5bet")
ALLIN_STATES = ("unopened", "after_open", "after_3bet", "cold")
ALLIN_GROUPS = tuple(f"vs_allin_{side}_{state}" for side in SIDES for state in ALLIN_STATES)
PRICES = ("normal", "cheap")
# 3-bet "small" when the raise-to total is at most this multiple of the open.
SMALL_3BET_MULTIPLE = 3.5
# All-in call rule: smoothing width (equity) and extra equity required per opponent still to act.
CALL_SMOOTHING = .01
BEHIND_MARGIN = .03
# v8.2 capped-range defence: smoothing width (range percentile) and the share each extra live
# opponent besides the raiser takes off this player's defending fraction.
DEFEND_GROUPS = ("limp_vs_raise", "flat_vs_squeeze")
DEFEND_SMOOTHING = .03
OTHER_DEFENDER_DECAY = .75


def depth_bucket(bb: float) -> int:
    for depth, upper in zip(DEPTHS, DEPTH_BOUNDS):
        if bb < upper:
            return depth
    return DEPTHS[-1]


def side(n: int, pos: str) -> str:
    """Early = seats before CO; tables without a CO (2–4 handed) have no early seats."""
    positions = POSITIONS[n]
    return "early" if "CO" in positions and positions.index(pos) < positions.index("CO") else "late"


def groups_for(n: int, pos: str) -> tuple[str, ...]:
    groups = list(BASE_GROUPS)
    # v7's SB/BB 3-bet groups are replaced by the v8 vs 3-bet / cold 3-bet nodes.
    if n > 2 and pos == "SB":
        groups.extend(g for g in SB_GROUPS if "3bet" not in g)
    if n > 2 and pos == "BB":
        groups.extend(g for g in BB_A3_GROUPS if "3bet" not in g)
    groups.extend(RAISE_GROUPS)
    return tuple(groups)


def contexts_for(n: int, pos: str, defend: bool = False) -> tuple[str, ...]:
    return (*(f"{g}:{p}" for g in groups_for(n, pos) for p in PRICES),
            *(f"{g}:allin" for g in ALLIN_GROUPS),
            *(f"{g}:mdf" for g in (DEFEND_GROUPS if defend else ())))


def expected_keys(defend: bool = False) -> set[str]:
    return {f"{n}/{d}/{p}/{c}" for n, ps in POSITIONS.items() for d in DEPTHS for p in ps
            for c in contexts_for(n, p, defend)}


def _a3_group(view: PlayerView, group: str) -> str:
    """v7's BB opener-position keys collapse to the A3 early/late opener split."""
    for state in ("single", "multi"):
        prefix = f"bb_vs_open_{state}_"
        if group.startswith(prefix):
            opener, size = group.removeprefix(prefix).rsplit("_", 1)
            if size == "large":
                return f"{prefix}large"
            early = opener in ("UTG", "UTG+1", "UTG+2", "LJ", "HJ")
            return f"{prefix}{'early' if early else 'late'}_{size}"
    return group


def context(view: PlayerView, legal: LegalActions, defend: bool = False) -> str:
    n = len(view.seats)
    actions = [e for e in view.history if e.type is EventType.ACTION and e.street is Street.PREFLOP]
    raises = [e for e in actions if e.action in (ActionType.BET, ActionType.RAISE)]
    seats = {s.seat: s for s in view.seats}
    me = seats[view.seat]
    mine = [i for i, e in enumerate(raises) if e.seat == view.seat]
    if raises and legal.can_call and (seats[raises[-1].seat].all_in or legal.call_amount >= me.stack):
        if len(raises) == 1:
            state = "unopened"
        elif mine:
            state = "after_open" if mine[-1] == 0 else "after_3bet"
        else:
            state = "cold"
        return f"vs_allin_{side(n, position(view, raises[-1].seat))}_{state}:allin"
    if defend and raises and not mine and len(raises) <= 2:
        # Raises already made when this player last called: 0 = limped, 1 = flatted the open.
        seen, called = 0, None
        for e in actions:
            if e.action in (ActionType.BET, ActionType.RAISE):
                seen += 1
            elif e.seat == view.seat and e.action is ActionType.CALL:
                called = seen
        if called == len(raises) - 1:
            return f"{DEFEND_GROUPS[called]}:mdf"
    base = v7_context(view, legal)
    group, price = base.split(":")
    if len(raises) == 2:
        if mine == [0]:
            size = "small" if raises[1].amount <= SMALL_3BET_MULTIPLE * raises[0].amount else "large"
            group = f"vs_3bet_{side(n, position(view, raises[1].seat))}_{size}"
        else:
            group = "cold_vs_3bet"
    elif len(raises) == 3:
        group = f"vs_4bet_{side(n, position(view, raises[2].seat))}" if 1 in mine else "vs_4bet_cold"
    elif len(raises) > 3:
        group = "vs_5bet"
    else:
        group = _a3_group(view, group)
    return f"{group}:{price}"


def key(view: PlayerView, legal: LegalActions, defend: bool = False) -> str:
    return f"{len(view.seats)}/{depth_bucket(effective_bb(view))}/{position(view)}/{context(view, legal, defend)}"


def allin_probs(view: PlayerView, legal: LegalActions, table: dict) -> np.ndarray:
    """Call iff equity vs the assumed jam range covers the actual price (smoothed near breakeven)."""
    live = [s for s in view.seats if not s.folded and s.seat != view.seat]
    committed = [s for s in live if s.all_in or s.street_bet >= view.current_bet]
    behind = len(live) - len(committed)
    # shortcut: equity vs several committed ranges approximated as eq ** k (independent showdowns);
    # players still to act add a fixed margin. Replace with a multiway equity model when one exists.
    eq = np.asarray(table["equity"], dtype=float) ** max(1, len(committed))
    price = legal.call_amount / (view.pot + legal.call_amount)
    need = price + BEHIND_MARGIN * behind
    z = np.clip((eq - need) / CALL_SMOOTHING, -60, 60)
    call = 1 / (1 + np.exp(-z))
    return np.stack([1 - call, call, np.zeros_like(call)], axis=1)


def defend_probs(view: PlayerView, legal: LegalActions, table: dict) -> np.ndarray:
    """Continue with the best fraction of the capped range: mdf_scale x MDF, shared with other live players."""
    others = sum(1 for s in view.seats if not s.folded and s.seat != view.seat) - 1
    mdf = 1 - legal.call_amount / view.pot
    fraction = table["mdf_scale"] * mdf * OTHER_DEFENDER_DECAY ** max(0, others)
    z = np.clip((fraction - np.asarray(table["percentile"], dtype=float)) / DEFEND_SMOOTHING, -60, 60)
    cont = 1 / (1 + np.exp(-z))
    raising = np.minimum(np.asarray(table["raise"], dtype=float), cont)
    return np.stack([1 - cont, cont - raising, raising], axis=1)


def _unit_list(x) -> bool:
    return isinstance(x, list) and len(x) == 169 and all(math.isfinite(v) and 0 <= v <= 1 for v in x)


def validate(data: dict) -> None:
    if set(data["tables"]) != expected_keys("defend_groups" in data):
        raise ValueError("incomplete frozen web preflop v8 tables")
    for k, table in data["tables"].items():
        if k.endswith(":allin"):
            if not _unit_list(table.get("equity")):
                raise ValueError(f"invalid equity in {k}")
            continue
        if k.endswith(":mdf"):
            if (not _unit_list(table.get("percentile")) or not _unit_list(table.get("raise"))
                    or not 0 < table["mdf_scale"] <= 1):
                raise ValueError(f"invalid defence table {k}")
        else:
            probs = table["probs"]
            if len(probs) != 169 or any(len(row) != 3 or any(not math.isfinite(x) or x < 0 or x > 1 for x in row)
                                        or abs(sum(row) - 1) > 1e-6 for row in probs):
                raise ValueError(f"invalid probabilities in {k}")
        if (not math.isfinite(table["raise_bb"]) or table["raise_bb"] <= 0
                or not math.isfinite(table["raise_multiple"]) or table["raise_multiple"] <= 1
                or not 0 < table["jam_fraction"] <= 1):
            raise ValueError(f"invalid sizing in {k}")


def lookup(view: PlayerView, legal: LegalActions, data: dict) -> tuple[np.ndarray, dict, str]:
    k = key(view, legal, "defend_groups" in data)
    table = data["tables"][k]
    if k.endswith(":allin"):
        probs = allin_probs(view, legal, table)
    elif k.endswith(":mdf"):
        probs = defend_probs(view, legal, table)
    else:
        probs = np.array(table["probs"], dtype=float)
    return probs, table, k
