"""Dataset validation: every row must pass before a dataset is registered.

Checks (design doc M6 gate):

* probabilities: each strategy value in [0, 1]; for every combo the actor
  can hold at the node, the action probabilities sum to 1;
* legality: the node and its action list exist in the bet tree rebuilt from
  the spot, with the same investments; a fold is only offered facing a bet;
* card conflicts: no combo that shares a card with the board has reach;
* values: reach non-negative, EVs finite wherever the actor has reach and an
  opponent combo is compatible;
* convergence: the spot reached the target exploitability;
* completeness: every shard present and every spot has its root node.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.engine.cards import parse_cards
from backend.solver.subgame import N, TreeBuilder, board_mask, compatible_mass

from .pipeline import DatasetConfig, shard_path
from .spots import make_spot, tree_config

PROB_TOL = 1e-2  # float16 storage


def _tree_nodes(spot, cfg: DatasetConfig) -> dict[str, tuple[list[str], tuple[float, float]]]:
    """First-street decision nodes of the spot's bet tree: path -> (actions, inv)."""
    board = parse_cards(spot.board)
    root = TreeBuilder(tree_config(spot, cfg.trees)).build(board)
    out = {}

    def rec(node) -> None:
        if node.kind != "decision":
            return
        out[node.path or "root"] = (list(node.actions), node.inv)
        for c in node.children:
            rec(c)

    rec(root)
    return out


def validate(out: Path, max_errors: int = 20) -> dict[str, Any]:
    import pyarrow.parquet as pq

    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    cfg = DatasetConfig.from_json(m["config"])
    errors: list[str] = []
    counts = {"rows": 0, "spots": 0, "missing_shards": 0, "not_converged": 0}

    def fail(msg: str) -> None:
        if len(errors) < max_errors:
            errors.append(msg)

    for k in range(cfg.n_shards):
        path = shard_path(out, k)
        if not path.exists():
            counts["missing_shards"] += 1
            fail(f"missing {path.name}")
            continue
        t = pq.read_table(path).to_pylist()
        expected = set(range(k * cfg.shard_size, min((k + 1) * cfg.shard_size, cfg.spots)))
        by_spot: dict[int, list[dict]] = {}
        for row in t:
            by_spot.setdefault(row["spot"], []).append(row)
        if set(by_spot) != expected:
            fail(f"{path.name}: spots {sorted(set(by_spot) ^ expected)[:5]} missing or unexpected")
        for idx, rows in by_spot.items():
            counts["spots"] += 1
            spot = make_spot(cfg.seed, idx, cfg.streets)
            nodes = _tree_nodes(spot, cfg)
            mask = board_mask(parse_cards(spot.board))
            if "root" not in {r["path"] for r in rows}:
                fail(f"spot {idx}: no root row")
            if rows[0]["exploitability_pct_pot"] > cfg.target_pct_pot:
                counts["not_converged"] += 1
            for r in rows:
                counts["rows"] += 1
                where = f"spot {idx} node {r['path']}"
                if r["board"] != spot.board or r["street"] != spot.street:
                    fail(f"{where}: spot fields differ from the generator")
                node = nodes.get(r["path"])
                if node is None:
                    fail(f"{where}: node not in the bet tree")
                    continue
                actions, inv = node
                if r["actions"] != actions:
                    fail(f"{where}: actions {r['actions']} != tree {actions}")
                if not np.allclose([r["inv_oop"], r["inv_ip"]], inv, atol=1e-3):
                    fail(f"{where}: investments differ from the tree")
                facing = inv[0] != inv[1]
                if ("f" in r["actions"]) != facing or ("x" in r["actions"]) == facing:
                    fail(f"{where}: fold/check offered illegally")
                reach = [np.asarray(r["reach_oop"], dtype=np.float64),
                         np.asarray(r["reach_ip"], dtype=np.float64)]
                if any(len(x) != N for x in reach):
                    fail(f"{where}: reach has the wrong length")
                    continue
                if any((x < 0).any() for x in reach):
                    fail(f"{where}: negative reach")
                if any(((x > 0) & (mask == 0)).any() for x in reach):
                    fail(f"{where}: reach on a combo that conflicts with the board")
                a = len(actions)
                strat = np.asarray(r["strategy"], dtype=np.float64).reshape(a, N)
                ev = np.asarray(r["ev"], dtype=np.float64).reshape(a, N)
                if (strat < -1e-6).any() or (strat > 1 + PROB_TOL).any():
                    fail(f"{where}: probability outside [0, 1]")
                live = reach[r["player"]] > 0
                sums = strat.sum(axis=0)[live]
                if live.any() and np.abs(sums - 1).max() > PROB_TOL:
                    fail(f"{where}: action probabilities sum to {sums.min():.4f}..{sums.max():.4f}")
                defined = live & (compatible_mass(reach[1 - r["player"]]) > 0)
                if not np.isfinite(ev[:, defined]).all():
                    fail(f"{where}: non-finite EV for a reachable combo")
    counts["errors"] = errors
    counts["ok"] = not errors
    return counts
