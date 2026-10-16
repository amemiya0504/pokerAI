"""Turn a GTO dataset into training tensors (cached as .npz).

Node arrays are stored once per node; a sample is (node, combo) for every
combo the actor can hold at that node. Sample weight = probability of
reaching the node with that combo (actor reach x compatible opponent reach),
normalised so every spot weighs 1 in total.

Splits are by spot, never by row, so no spot appears in two splits:
``key % 10 == 0`` is the test split (reported), ``key % 10 == 1`` the
validation split (model selection), the rest is training. ``key`` is the spot
index offset by the dataset's position in the list, so several datasets can
be combined.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

from backend.solver.subgame import CARD_A, CARD_B, N, compatible_mass

from .features import N_ACTIONS, action_slots, board_features, equity, legal_mask, node_numeric, normalise, range_histograms

SPLIT_EVERY = 10
DATASET_OFFSET = 10_000_000  # spot keys: dataset position * offset + spot index


def prepare(dataset_dirs: list[Path] | Path, out_file: Path, progress=None) -> dict:
    import pyarrow.parquet as pq

    if isinstance(dataset_dirs, Path):
        dataset_dirs = [dataset_dirs]
    manifests = [json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in dataset_dirs]
    shards = [(di, f) for di, d in enumerate(dataset_dirs) for f in sorted(glob.glob(str(d / "shard-*.parquet")))]
    node_board, node_num, node_legal, node_spot, node_pot = [], [], [], [], []
    node_reach_actor, node_reach_opp, node_strength, node_category, node_equity = [], [], [], [], []
    s_node, s_combo, s_target, s_ev, s_evmask, s_weight = [], [], [], [], [], []
    n_nodes = 0
    for si, (di, shard) in enumerate(shards):
        rows = pq.read_table(shard).to_pylist()
        by_spot: dict[int, list[dict]] = {}
        for r in rows:
            by_spot.setdefault(r["spot"], []).append(r)
        for spot, srows in by_spot.items():
            bf = board_features(srows[0]["board"])
            root = next(r for r in srows if r["path"] == "root")
            r0 = np.asarray(root["reach_oop"], dtype=np.float64)
            r1 = np.asarray(root["reach_ip"], dtype=np.float64)
            z = float((r0 * compatible_mass(r1)).sum())
            for r in srows:
                p = r["player"]
                reach = (np.asarray(r["reach_oop"], dtype=np.float64), np.asarray(r["reach_ip"], dtype=np.float64))
                actor, opp = reach[p], reach[1 - p]
                mass = compatible_mass(opp)
                weight = actor * mass / z
                live = np.nonzero((actor > 0) & (weight > 0))[0]
                if len(live) == 0:
                    continue
                a = len(r["actions"])
                slots = action_slots(r["actions"])
                strat = np.asarray(r["strategy"], dtype=np.float32).reshape(a, N)
                ev = np.asarray(r["ev"], dtype=np.float32).reshape(a, N)
                target = np.zeros((len(live), N_ACTIONS), dtype=np.float32)
                evs = np.zeros((len(live), N_ACTIONS), dtype=np.float32)
                target[:, slots] = strat[:, live].T
                target /= target.sum(axis=1, keepdims=True)
                evs[:, slots] = np.nan_to_num(ev[:, live].T)
                evmask = np.isfinite(ev[:, live]).all(axis=0)
                board = list(bf.board) + [52] * (5 - len(bf.board))
                node_board.append(board)
                eq_actor = equity(bf, opp)
                hist = range_histograms(bf, actor, opp, eq_actor)
                node_num.append(node_numeric(r["street"], p, r["pot"], r["stack"], (r["inv_oop"], r["inv_ip"]), hist))
                node_legal.append(legal_mask(r["actions"]))
                node_spot.append(di * DATASET_OFFSET + spot)
                node_pot.append(r["pot"])
                node_reach_actor.append(normalise(actor).astype(np.float16))
                node_reach_opp.append(normalise(opp).astype(np.float16))
                node_strength.append(bf.strength.astype(np.float16))
                node_category.append(bf.category)
                node_equity.append(eq_actor.astype(np.float16))
                s_node.append(np.full(len(live), n_nodes, dtype=np.int32))
                s_combo.append(live.astype(np.int16))
                s_target.append(target.astype(np.float16))
                s_ev.append(evs)
                s_evmask.append(evmask)
                s_weight.append(weight[live].astype(np.float32))
                n_nodes += 1
        if progress:
            progress(si + 1, len(shards))
    arrays = {
        "node_board": np.array(node_board, dtype=np.int8),
        "node_numeric": np.stack(node_num),
        "node_legal": np.stack(node_legal),
        "node_spot": np.array(node_spot, dtype=np.int64),
        "node_pot": np.array(node_pot, dtype=np.float32),
        "node_reach_actor": np.stack(node_reach_actor),
        "node_reach_opp": np.stack(node_reach_opp),
        "node_strength": np.stack(node_strength),
        "node_category": np.stack(node_category),
        "node_equity": np.stack(node_equity),
        "sample_node": np.concatenate(s_node),
        "sample_combo": np.concatenate(s_combo),
        "sample_target": np.concatenate(s_target),
        "sample_ev": np.concatenate(s_ev),
        "sample_evmask": np.concatenate(s_evmask),
        "sample_weight": np.concatenate(s_weight),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, **arrays)
    meta = {"datasets": [{"dataset": m["dataset"], "version": m["version"], "dir": str(d)}
                         for m, d in zip(manifests, dataset_dirs)],
            "nodes": n_nodes, "samples": int(len(arrays["sample_node"]))}
    out_file.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class Tensors:
    """Training tensors on a device, with a spot-level train/validation split."""

    def __init__(self, npz_file: Path, device) -> None:
        import torch

        d = np.load(npz_file)
        t = lambda x, dt=None: torch.as_tensor(x, device=device, dtype=dt)  # noqa: E731
        self.device = device
        self.board_multi = torch.zeros(len(d["node_board"]), 53, device=device)
        self.board_multi.scatter_(1, t(d["node_board"].astype(np.int64)), 1.0)
        self.board_multi = self.board_multi[:, :52].contiguous()
        self.numeric = t(d["node_numeric"], torch.float32)
        self.legal = t(d["node_legal"])
        self.pot = t(d["node_pot"], torch.float32)
        self.reach_actor = t(d["node_reach_actor"])
        self.reach_opp = t(d["node_reach_opp"])
        self.strength = t(d["node_strength"])
        self.category = t(d["node_category"])  # int8 to save memory; widened per batch
        self.equity = t(d["node_equity"])
        self.is_river = self.numeric[:, 0]
        self.card_a = t(CARD_A.astype(np.int64))
        self.card_b = t(CARD_B.astype(np.int64))
        node = d["sample_node"].astype(np.int64)
        self.s_node = t(node)
        self.s_combo = t(d["sample_combo"].astype(np.int64))
        self.s_target = t(d["sample_target"])
        self.s_ev = t(d["sample_ev"])
        self.s_evmask = t(d["sample_evmask"])
        self.s_weight = t(d["sample_weight"])
        bucket = d["node_spot"][node] % SPLIT_EVERY
        self.test_idx = t(np.nonzero(bucket == 0)[0])
        self.val_idx = t(np.nonzero(bucket == 1)[0])
        self.train_idx = t(np.nonzero(bucket >= 2)[0])
        self.node_spot = d["node_spot"]

    def batch(self, idx):
        """Model inputs, targets, EVs, EV mask and weights for sample indices."""
        import torch

        node = self.s_node[idx]
        combo = self.s_combo[idx]
        hole = torch.zeros(len(idx), 52, device=self.device)
        hole.scatter_(1, self.card_a[combo][:, None], 1.0)
        hole.scatter_(1, self.card_b[combo][:, None], 1.0)
        cat = torch.nn.functional.one_hot(self.category[node, combo].long().clamp_min(0), 9).float()
        river = self.is_river[node]
        scalars = torch.stack([self.strength[node, combo].float(),
                               self.equity[node, combo].float(), river], dim=1)
        inputs = (hole, self.board_multi[node], cat, scalars, self.numeric[node], self.legal[node],
                  self.reach_actor[node].float(), self.reach_opp[node].float())
        return (inputs, self.s_target[idx].float(), self.s_ev[idx], self.s_evmask[idx],
                self.s_weight[idx], self.pot[node])
