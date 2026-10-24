"""How far a depth-limited flop solve is from the full solve.

The real-time agent solves the flop on a depth-limited tree: betting on the
first ``depth`` streets, then the remaining cards are dealt with no betting
(leaf value = exact all-in equity). That ignores the later betting, so it
misvalues draws and position. This module measures the cost on spots small
enough to solve the full tree (one 75% pot size per street, no raises,
SPR <= 4):

* ``root_tv``: total variation between the depth-limited and the full root
  strategy, weighted by the root player's range;
* ``vs_full_pct_pot``: duplicate head-to-head of the *hybrid* (flop from the
  depth-limited solve, turn and river from the full solve, i.e. what the
  agent does when it re-solves each street) against the full solve;
* ``exploitability_pct_pot``: exact exploitability of the hybrid.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from backend.dataset.spots import Spot, make_spot
from backend.engine.cards import parse_cards
from backend.solver.subgame import TreeConfig, compatible_mass
from backend.solver.subgame_gpu import BatchedSubgameCFR

from .exploit import average_table, head_to_head, install, table_policy

SIZE = 0.75


def bias_tree(spot: Spot) -> TreeConfig:
    """The small full flop tree: one bet size per street, no raises."""
    return TreeConfig(spot.pot, spot.stack, {s: [SIZE] for s in ("flop", "turn", "river")}, {}, 0, True)


def bias_spots(seed: int = 5, max_spr: float = 4.0) -> Iterator[Spot]:
    """Flop spots whose full tree fits in GPU memory (8 GB at SPR 4)."""
    idx = 0
    while True:
        spot = make_spot(seed, idx, {"flop": 1.0})
        idx += 1
        if spot.stack / spot.pot <= max_spr:
            yield spot


def _flop_node(path: str) -> bool:
    return "/" not in path and "*" not in path


@dataclass
class DepthBias:
    spot: int
    board: str
    spr: float
    full_pct_pot: float
    full_iterations: int
    depth: dict[int, dict[str, float]] = field(default_factory=dict)  # depth -> metrics


def evaluate_spot(spot: Spot, depths: tuple[int, ...] = (1, 2), full_target_pct_pot: float = 0.3,
                  full_max_iterations: int = 400, depth_iterations: int = 600, device: str | None = None,
                  dtype=torch.float32) -> DepthBias:
    board = parse_cards(spot.board)
    cfg = bias_tree(spot)
    full = BatchedSubgameCFR(board, spot.ranges(), cfg, device=device, dtype=dtype)
    fx = None
    while full.t < full_max_iterations:
        full.iterate(min(50, full_max_iterations - full.t))
        fx = full.exploitability()["exploitability_pct_pot"]
        if fx < full_target_pct_pot:
            break
    full_table = average_table(full)
    full_pol = table_policy(full_table)
    r0, r1 = spot.ranges()
    w = r0 * compatible_mass(r1)
    w = w / w.sum()
    out = DepthBias(spot.index, spot.board, spot.stack / spot.pot, fx, full.t)
    for depth in depths:
        dl = BatchedSubgameCFR(board, spot.ranges(), cfg, device=device, dtype=dtype, checkdown=depth)
        dl.iterate(depth_iterations)
        dl_table = average_table(dl)
        dl.release()
        tv = 0.5 * (w * np.abs(full_table["root"][:, 0, :] - dl_table["root"][:, 0, :]).sum(0)).sum()
        dl_pol = table_policy(dl_table)

        def hybrid(ctx, dl_pol=dl_pol):
            return (dl_pol if _flop_node(ctx["path"]) else full_pol)(ctx)

        h2h = head_to_head(spot, hybrid, full_pol, full, 0.0)
        install(full, spot, hybrid)
        hx = full.exploitability()["exploitability_pct_pot"]
        out.depth[depth] = {"root_tv": float(tv), "vs_full_pct_pot": float(h2h.a_advantage_pct_pot),
                            "exploitability_pct_pot": float(hx)}
    full.release()
    return out


def summarize(results: list[DepthBias]) -> dict[str, Any]:
    out: dict[str, Any] = {"spots": len(results),
                           "full_pct_pot_median": float(np.median([r.full_pct_pot for r in results]))}
    for d in sorted({d for r in results for d in r.depth}):
        col = {k: np.array([r.depth[d][k] for r in results]) for k in results[0].depth[d]}
        out[f"depth{d}"] = {
            "root_tv": {"mean": float(col["root_tv"].mean()), "max": float(col["root_tv"].max())},
            "vs_full_pct_pot": {"mean": float(col["vs_full_pct_pot"].mean()),
                                "min": float(col["vs_full_pct_pot"].min())},
            "exploitability_pct_pot": {"median": float(np.median(col["exploitability_pct_pot"])),
                                       "max": float(col["exploitability_pct_pot"].max())},
        }
    return out
