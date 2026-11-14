"""Inference: action probabilities for a heads-up postflop node.

Features are computed from raw inputs with the same code as training
(``features.py``). Board-only work (hand ranks, showdown tables) is cached
per board, so repeated queries on one board are cheap.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from backend.solver.subgame import CARD_A, CARD_B

from .features import SLOT, BoardFeatures, action_slots, board_features, legal_mask, node_numeric, normalise, range_histograms
from .model import build_model


class PolicyRunner:
    def __init__(self, model_path: str | Path, device: str = "cpu", cache_boards: int = 64) -> None:
        state = torch.load(model_path, map_location=device, weights_only=False)
        self.model = build_model(state["model_config"]).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.device = torch.device(device)
        self.report = state.get("report")
        self._boards: OrderedDict[str, BoardFeatures] = OrderedDict()
        self._cache_boards = cache_boards

    def _board(self, board: str) -> BoardFeatures:
        bf = self._boards.get(board)
        if bf is None:
            bf = board_features(board)
            self._boards[board] = bf
            if len(self._boards) > self._cache_boards:
                self._boards.popitem(last=False)
        else:
            self._boards.move_to_end(board)
        return bf

    @torch.no_grad()
    def predict(self, *, board: str, street: str, pot: float, stack: float, inv: tuple[float, float],
                player: int, actions: list[str], reach_oop: np.ndarray, reach_ip: np.ndarray,
                combos: list[int] | np.ndarray) -> np.ndarray:
        """Probabilities (len(combos), len(actions)) for the actor holding each combo."""
        bf = self._board(board)
        combos = np.asarray(combos, dtype=np.int64)
        reach = (np.asarray(reach_oop, dtype=np.float64), np.asarray(reach_ip, dtype=np.float64))
        opp = reach[1 - player]
        n = len(combos)
        dev = self.device
        hole = torch.zeros(n, 52, device=dev)
        rows = torch.arange(n, device=dev)
        hole[rows, torch.as_tensor(CARD_A[combos], device=dev)] = 1.0
        hole[rows, torch.as_tensor(CARD_B[combos], device=dev)] = 1.0
        board_multi = torch.zeros(1, 52, device=dev)
        board_multi[0, list(bf.board)] = 1.0
        cat = torch.nn.functional.one_hot(torch.as_tensor(bf.category[combos].clip(0), dtype=torch.long,
                                                          device=dev), 9).float()
        river = 1.0 if street == "river" else 0.0
        eq_all = bf.runout(opp)
        eq = eq_all[combos]
        hist = range_histograms(bf, reach[player], opp, eq_all)
        scalars = torch.as_tensor(np.stack([bf.strength[combos], eq, np.full(n, river)], axis=1),
                                  dtype=torch.float32, device=dev)
        num = torch.as_tensor(node_numeric(street, player, pot, stack, inv, hist), device=dev)[None].expand(n, -1)
        legal = torch.as_tensor(legal_mask(actions), device=dev)[None].expand(n, -1)
        ra = torch.as_tensor(normalise(reach[player]), device=dev)[None]  # one row: embedded once
        ro = torch.as_tensor(normalise(opp), device=dev)[None]
        probs = torch.softmax(self.model(hole, board_multi.expand(n, -1), cat, scalars, num, legal, ra, ro), dim=1)
        return probs[:, action_slots(actions)].cpu().numpy()


def benchmark(runner: PolicyRunner, spots: list[dict], repeats: int = 1) -> dict[str, float]:
    """Wall time per decision (one combo, fresh board each time: features included)."""
    times = []
    for _ in range(repeats):
        for s in spots:
            runner._boards.clear()  # worst case: no cached board work
            t0 = time.perf_counter()
            runner.predict(**s)
            times.append(time.perf_counter() - t0)
    ms = np.array(times) * 1000
    return {"decisions": len(ms), "p50_ms": float(np.percentile(ms, 50)),
            "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max())}


__all__ = ["PolicyRunner", "benchmark", "SLOT"]
