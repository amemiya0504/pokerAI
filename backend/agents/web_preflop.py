"""Frozen, position-specific web preflop policy. No table generation at runtime."""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.engine.actions import Action, ActionType, LegalActions
from backend.engine.cards import Deck
from backend.engine.hand import EventType, HandState, PlayerView, Street
from backend.solver.hands import CLASS_NAMES, COMBOS, class_index

from .pushfold import PushFoldAgent, preflop_order

# A8.3 passed (2026-10-27): v8.2 = v8.1 + defence nodes, read only by adaptive agents (defend=True).
TABLE_PATH = Path(__file__).with_name("data") / "web_preflop_v8_2.json.gz"
POSITIONS = {
    2: ("BTN/SB", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("CO", "BTN", "SB", "BB"),
    5: ("HJ", "CO", "BTN", "SB", "BB"),
    6: ("LJ", "HJ", "CO", "BTN", "SB", "BB"),
    7: ("UTG", "LJ", "HJ", "CO", "BTN", "SB", "BB"),
    8: ("UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"),
    9: ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB", "BB"),
}
OPEN_GROUPS = ("vs_open_early_small", "vs_open_early_medium", "vs_open_late_small",
               "vs_open_late_medium", "vs_open_large")
LEGACY_GROUPS = ("open", "limped", "option", *OPEN_GROUPS, "vs_3bet", "vs_4bet")
SB_GROUPS = tuple(f"sb_{state}_{group.removeprefix('vs_open_')}"
                  for state in ("vs_open", "vs_open_multi") for group in OPEN_GROUPS) + (
    "sb_vs_3bet", "sb_vs_3bet_multi")
BB_A3_GROUPS = tuple(f"bb_vs_open_{state}_{group.removeprefix('vs_open_')}"
                     for state in ("single", "multi") for group in OPEN_GROUPS) + (
    "bb_vs_3bet_single", "bb_vs_3bet_multi", "bb_limped_single", "bb_limped_multi")
PREVIOUS_GROUPS = (*LEGACY_GROUPS, *SB_GROUPS)
BB_OPENERS = ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN", "SB")
BB_OPEN_SIZES = ("small", "medium", "large")
BB_A4_GROUPS = tuple(f"bb_vs_open_{state}_{opener}_{size}"
                     for state in ("single", "multi") for opener in BB_OPENERS
                     for size in BB_OPEN_SIZES)
A3_GROUPS = (*PREVIOUS_GROUPS, *BB_A3_GROUPS)
GROUPS = (*A3_GROUPS, *BB_A4_GROUPS)
CONTEXTS = tuple(f"{g}:{price}" for g in GROUPS for price in ("normal", "cheap"))
LEGACY_CONTEXTS = tuple(f"{g}:{price}" for g in LEGACY_GROUPS for price in ("normal", "cheap"))
PREVIOUS_CONTEXTS = tuple(f"{g}:{price}" for g in PREVIOUS_GROUPS for price in ("normal", "cheap"))
A3_CONTEXTS = tuple(f"{g}:{price}" for g in A3_GROUPS for price in ("normal", "cheap"))


def contexts_for(n: int, pos: str) -> tuple[str, ...]:
    groups = list(LEGACY_GROUPS)
    if n > 2 and pos == "SB":
        groups.extend(SB_GROUPS)
    if n > 2 and pos == "BB":
        groups.extend(BB_A3_GROUPS)
        groups.extend(BB_A4_GROUPS)
    return tuple(f"{group}:{price}" for group in groups for price in ("normal", "cheap"))


def position(view: PlayerView, seat: int | None = None) -> str:
    return POSITIONS[len(view.seats)][preflop_order(view).index(view.seat if seat is None else seat)]


def context(view: PlayerView, legal: LegalActions) -> str:
    actions = [e for e in view.history if e.type is EventType.ACTION and e.street is Street.PREFLOP]
    raises = [e for e in actions if e.action in (ActionType.BET, ActionType.RAISE)]
    if len(raises) >= 3:
        group = "vs_4bet"
    elif len(raises) == 2:
        group = "vs_3bet"
    elif raises:
        opener = position(view, raises[0].seat)
        size = "small" if view.current_bet <= 2.5 * view.big_blind else "medium"
        group = ("vs_open_large" if view.current_bet > 4 * view.big_blind else
                 f"vs_open_late_{size}" if opener in ("CO", "BTN", "SB", "BTN/SB") else f"vs_open_early_{size}")
    elif legal.can_check:
        group = "option"
    else:
        group = "limped" if any(e.action is ActionType.CALL for e in actions) else "open"
    if len(view.seats) > 2 and position(view) == "SB":
        live = {s.seat for s in view.seats if not s.folded and s.seat != view.seat}
        last_action = {e.seat: e.action for e in actions}
        callers = {seat for seat, action in last_action.items()
                   if seat in live and action is ActionType.CALL}
        if len(raises) == 1:
            group = f"sb_vs_open_{'multi_' if callers else ''}{group.removeprefix('vs_open_')}"
        elif len(raises) == 2:
            group = "sb_vs_3bet_multi" if callers else "sb_vs_3bet"
    elif len(view.seats) > 2 and position(view) == "BB":
        live = {s.seat for s in view.seats if not s.folded and s.seat != view.seat}
        last_action = {e.seat: e.action for e in actions}
        callers = {seat for seat, action in last_action.items()
                   if seat in live and action is ActionType.CALL}
        if len(raises) == 1:
            state = "multi" if callers else "single"
            opener = position(view, raises[0].seat)
            size = ("small" if view.current_bet <= 2.5 * view.big_blind else
                    "medium" if view.current_bet <= 4 * view.big_blind else "large")
            group = f"bb_vs_open_{state}_{opener}_{size}"
        elif len(raises) == 2:
            group = "bb_vs_3bet_multi" if callers else "bb_vs_3bet_single"
        elif not raises and callers:
            group = "bb_limped_multi" if len(callers) > 1 else "bb_limped_single"
    cheap = legal.can_call and legal.call_amount / (view.pot + legal.call_amount) <= 0.25
    return f"{group}:{'cheap' if cheap else 'normal'}"


@lru_cache(maxsize=1)
def tables() -> dict:
    raw = TABLE_PATH.read_bytes()
    data = json.loads(gzip.decompress(raw) if TABLE_PATH.suffix == ".gz" else raw)
    if data.get("schema") not in (1, 2) or data.get("hands") != list(CLASS_NAMES) or data.get("actions") != ["fold", "passive", "raise"]:
        raise ValueError("invalid frozen web preflop schema")
    if data["schema"] == 2:
        from .web_preflop_v8 import validate

        validate(data)
        return data
    expected = {f"{n}/{p}/{c}" for n, ps in POSITIONS.items() for p in ps
                 for c in contexts_for(n, p)}
    legacy = {f"{n}/{p}/{c}" for n, ps in POSITIONS.items() for p in ps for c in LEGACY_CONTEXTS}
    previous = {f"{n}/{p}/{c}" for n, ps in POSITIONS.items() for p in ps for c in PREVIOUS_CONTEXTS}
    a3 = {f"{n}/{p}/{c}" for n, ps in POSITIONS.items() for p in ps for c in A3_CONTEXTS}
    if set(data["tables"]) not in (expected, a3, previous, legacy):
        raise ValueError("incomplete frozen web preflop tables")
    for key, table in data["tables"].items():
        probs = table["probs"]
        if len(probs) != 169 or any(len(row) != 3 or any(not math.isfinite(x) or x < 0 or x > 1 for x in row)
                                    or abs(sum(row) - 1) > 1e-6 for row in probs):
            raise ValueError(f"invalid probabilities in {key}")
        if not math.isfinite(table["raise_bb"]) or table["raise_bb"] <= 0 or not math.isfinite(table["raise_multiple"]) or table["raise_multiple"] <= 1:
            raise ValueError(f"invalid sizing in {key}")
    return data


def lookup(view: PlayerView, legal: LegalActions, defend: bool = False) -> tuple[np.ndarray, dict, str]:
    """v8.2 defence nodes only with ``defend=True`` (the adaptive agent's own choice); range
    estimates and every other caller read the v8.1 nodes, so a v8.2 table otherwise plays as v8.1."""
    data = tables()
    if data["schema"] == 2:
        from .web_preflop_v8 import lookup as v8_lookup

        probs, table, key = v8_lookup(view, legal, data, defend)
    else:
        probs, table, key = _v7_lookup(view, legal, data)
    if legal.can_check or not legal.can_fold:
        probs[:, 1] += probs[:, 0]
        probs[:, 0] = 0
    if not legal.can_raise:
        probs[:, 1] += probs[:, 2]
        probs[:, 2] = 0
    return probs, table, key


def _v7_lookup(view: PlayerView, legal: LegalActions, data: dict) -> tuple[np.ndarray, dict, str]:
    key = f"{len(view.seats)}/{position(view)}/{context(view, legal)}"
    if key not in data["tables"]:
        group, price = key.rsplit("/", 1)[-1].split(":")
        if group.startswith("bb_vs_open_single_") or group.startswith("bb_vs_open_multi_"):
            state = "multi" if group.startswith("bb_vs_open_multi_") else "single"
            opener, size = group.removeprefix(f"bb_vs_open_{state}_").rsplit("_", 1)
            early = opener in ("UTG", "UTG+1", "UTG+2", "LJ", "HJ")
            old_size = "large" if size == "large" else ("small" if size == "small" else "medium")
            old_group = (f"bb_vs_open_{state}_large" if old_size == "large" else
                         f"bb_vs_open_{state}_{'early' if early else 'late'}_{old_size}")
            old_key = f"{len(view.seats)}/{position(view)}/{old_group}:{price}"
            generic = ("vs_open_large" if old_size == "large" else
                       f"vs_open_{'early' if early else 'late'}_{old_size}")
            group = old_group if old_key in data["tables"] else generic
        elif group.startswith("sb_vs_open_multi_"):
            group = "vs_open_" + group.removeprefix("sb_vs_open_multi_")
        elif group.startswith("sb_vs_open_"):
            group = "vs_open_" + group.removeprefix("sb_vs_open_")
        elif group.startswith("sb_vs_3bet"):
            group = "vs_3bet"
        elif group.startswith("bb_vs_3bet"):
            group = "vs_3bet"
        elif group.startswith("bb_limped_"):
            group = "limped"
        key = f"{len(view.seats)}/{position(view)}/{group}:{price}"
    table = data["tables"][key]
    return np.array(table["probs"], dtype=float), table, key


def effective_bb(view: PlayerView) -> float:
    start = {s.seat: s.stack + s.total_bet for s in view.seats}
    others = [start[s.seat] for s in view.seats if s.seat != view.seat and not s.folded]
    return min(start[view.seat], max(others, default=0)) / view.big_blind


def probabilities(view: PlayerView, legal: LegalActions, pushfold: PushFoldAgent) -> np.ndarray:
    """169 x (fold, passive, raise), including the existing solved short-stack cells."""
    probs, _, _ = lookup(view, legal)
    if 5 <= effective_bb(view) <= 20:
        for ci, combos in enumerate(COMBOS):
            found = pushfold._lookup(replace(view, hole_cards=combos[0]))
            if isinstance(found, str):
                continue
            _, infoset, q, _ = found
            probs[ci] = (0 if legal.can_check else 1 - q,
                         (1 - q if legal.can_check else 0) + (q if "vs" in infoset or not legal.can_raise else 0),
                         q if "vs" not in infoset and legal.can_raise else 0)
    return probs


def ranges(view: PlayerView, pushfold: PushFoldAgent) -> dict[int, np.ndarray]:
    """Condition both HU ranges on the same action probabilities used by the web agent.

    A synthetic deal replays only public preflop actions to recover exact legal
    raises and prices. Its private cards are never used by the table lookup.
    """
    from backend.engine.fast_eval import CARD_A, CARD_B

    classes = np.array([class_index(int(a), int(b)) for a, b in zip(CARD_A, CARD_B)])
    weights = {s.seat: np.ones(1326) for s in view.seats}
    replay = HandState({s.seat: s.stack + s.total_bet for s in view.seats}, view.button,
                       view.small_blind, view.big_blind, Deck(0))
    for e in view.history:
        if e.type is not EventType.ACTION or e.street is not Street.PREFLOP:
            continue
        public = replay.observation(e.seat)
        probs = probabilities(public, replay.legal_actions(), pushfold)
        outcome = {ActionType.FOLD: 0, ActionType.CHECK: 1, ActionType.CALL: 1}.get(e.action, 2)
        weights[e.seat] *= probs[classes, outcome]
        replay.apply(Action(e.action, e.amount if outcome == 2 else 0))
    return weights
