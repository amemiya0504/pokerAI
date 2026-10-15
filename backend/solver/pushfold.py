"""Preflop push/fold equilibria for 2-9 players (an abstracted preflop game).

Game: every player has the same effective stack S (in big blinds); blinds
0.5/1, no ante. Players act once in preflop order (UTG .. SB, BB). Until
someone moves all-in, a player may fold or push; after the first push each
later player may fold or call all-in. If everyone folds to the big blind it
wins the small blind. All-ins run to a showdown, so payoffs are exact chip
EV: there is no model of postflop play to approximate.

Solver: external-sampling Monte Carlo CFR with linear weighting, run in
synchronous batches across processes (all workers sample with the same
current strategy, their regret increments are summed). Each player acts at
most once, so a player's own reach probability is always 1 and the average
strategy is the linearly weighted mean of the current strategies.

The output is labelled SOLVER_APPROX: it is an approximate equilibrium of
this abstracted game, with a Monte Carlo estimate of NashConv (the sum of
every player's best-response gain), not exact GTO for full no-limit play.
"""

from __future__ import annotations

import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import numpy as np

from backend.agents.base import StrategySource
from backend.engine.cards import derive_seed
from backend.engine.evaluator import evaluate

from .base import SolveRequest, SolveResult
from .hands import N_CLASSES, class_index

POSITIONS_9 = ("UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN", "SB", "BB")
FOLD, AGGRESS = 0, 1  # AGGRESS = push when unopened, call when facing a push


def positions(n: int) -> tuple[str, ...]:
    return POSITIONS_9[-n:]


@dataclass(frozen=True)
class PushFoldGame:
    players: int
    stack_bb: float

    def __post_init__(self) -> None:
        if not 2 <= self.players <= 9:
            raise ValueError("players must be 2..9")
        if self.stack_bb <= 1:
            raise ValueError("stack must exceed the big blind")

    @property
    def blinds(self) -> list[float]:
        b = [0.0] * self.players
        b[-2], b[-1] = 0.5, 1.0
        return b

    def infosets(self) -> list[tuple[int, int, int]]:
        """(actor, first pusher or -1, bitmask of callers so far)."""
        n = self.players
        out = [(p, -1, 0) for p in range(n - 1)]
        for f in range(n - 1):
            for p in range(f + 1, n):
                between = list(range(f + 1, p))
                for m in range(1 << len(between)):
                    mask = sum(1 << between[i] for i in range(len(between)) if m >> i & 1)
                    out.append((p, f, mask))
        return out

    def infoset_name(self, key: tuple[int, int, int]) -> str:
        pos = positions(self.players)
        p, f, mask = key
        if f < 0:
            return f"{pos[p]}|open"
        callers = [pos[i] for i in range(self.players) if mask >> i & 1]
        return f"{pos[p]}|vs {pos[f]} push" + (f"|callers {','.join(callers)}" if callers else "")

    def actions(self, key: tuple[int, int, int]) -> tuple[str, str]:
        return ("fold", "push") if key[1] < 0 else ("fold", "call")


# ---------------------------------------------------------------------------
# Sampled traversal (runs in worker processes)


class _Traverser:
    def __init__(self, game: PushFoldGame) -> None:
        self.n = game.players
        self.s = float(game.stack_bb)
        self.blinds = game.blinds
        keys = game.infosets()
        self.index = {k: i for i, k in enumerate(keys)}
        self.n_inf = len(keys)

    def payoff(self, t: int, f: int, mask: int, scores: list[int]) -> float:
        n, blinds, s = self.n, self.blinds, self.s
        if f < 0:  # everyone folded to the big blind
            return 0.5 if t == n - 1 else -blinds[t]
        players = [f] + [i for i in range(n) if mask >> i & 1]
        if t not in players:
            return -blinds[t]
        if len(players) == 1:
            return sum(blinds) - blinds[f]
        pot = s * len(players) + sum(blinds[i] for i in range(n) if i not in players)
        best = max(scores[i] for i in players)
        winners = [i for i in players if scores[i] == best]
        return (pot / len(winners) if t in winners else 0.0) - s

    def walk(self, t: int, p: int, f: int, mask: int, strat, cls, scores, rng, sink) -> float:
        """Value of the subtree for traverser t; other players are sampled.
        ``sink(i, h, values, probs)`` receives t's action values at t's node."""
        n = self.n
        while True:
            if (f < 0 and p == n - 1) or p == n:
                return self.payoff(t, f, mask, scores)
            i = self.index[(p, f, mask)]
            h = cls[p]
            q = strat[i][h]
            if p == t:
                v0 = self.walk(t, p + 1, f, mask, strat, cls, scores, rng, sink)
                if f < 0:
                    v1 = self.walk(t, p + 1, p, 0, strat, cls, scores, rng, sink)
                else:
                    v1 = self.walk(t, p + 1, f, mask | 1 << p, strat, cls, scores, rng, sink)
                sink(i, h, v0, v1, q)
                return (1 - q) * v0 + q * v1
            if rng.random() < q:
                if f < 0:
                    f, mask = p, 0
                else:
                    mask |= 1 << p
            p += 1


def _deal(rng: random.Random, n: int):
    cards = rng.sample(range(52), 2 * n + 5)
    board = cards[2 * n:]
    holes = [(cards[2 * i], cards[2 * i + 1]) for i in range(n)]
    cls = [class_index(a, b) for a, b in holes]
    scores = [evaluate((a, b, *board)) for a, b in holes]
    return cls, scores


def _regret_batch(game: PushFoldGame, strat: list, seed: int, deals: int):
    """Summed regret increments and visit counts for ``deals`` sampled deals."""
    tr = _Traverser(game)
    rng = random.Random(seed)
    regret = np.zeros((tr.n_inf, N_CLASSES, 2))
    visits = np.zeros((tr.n_inf, N_CLASSES), dtype=np.int64)
    r_flat: dict[tuple[int, int], list[float]] = {}

    def sink(i, h, v0, v1, q):
        v = (1 - q) * v0 + q * v1
        acc = r_flat.get((i, h))
        if acc is None:
            r_flat[(i, h)] = [v0 - v, v1 - v, 1]
        else:
            acc[0] += v0 - v
            acc[1] += v1 - v
            acc[2] += 1

    for _ in range(deals):
        cls, scores = _deal(rng, game.players)
        for t in range(game.players):
            tr.walk(t, 0, -1, 0, strat, cls, scores, rng, sink)
    for (i, h), (a, b, c) in r_flat.items():
        regret[i, h, 0] = a
        regret[i, h, 1] = b
        visits[i, h] = c
    return regret, visits


def _eval_batch(game: PushFoldGame, strat: list, seed: int, deals: int):
    """Summed action values per (infoset, class) for best-response estimates."""
    tr = _Traverser(game)
    rng = random.Random(seed)
    values = np.zeros((tr.n_inf, N_CLASSES, 2))
    counts = np.zeros((tr.n_inf, N_CLASSES), dtype=np.int64)
    no_decision = np.zeros(game.players)  # payoffs reached without the player's own decision

    for _ in range(deals):
        cls, scores = _deal(rng, game.players)
        for t in range(game.players):
            got: list = []
            v = tr.walk(t, 0, -1, 0, strat, cls, scores, rng,
                        lambda i, h, v0, v1, q: got.append((i, h, v0, v1)))
            if got:
                i, h, v0, v1 = got[0]
                values[i, h, 0] += v0
                values[i, h, 1] += v1
                counts[i, h] += 1
            else:
                no_decision[t] += v
    return values, counts, no_decision


def _strategy_lists(probs_aggress: np.ndarray) -> list:
    return probs_aggress.tolist()


def regret_matching(regret: np.ndarray) -> np.ndarray:
    """Probability of the aggressive action for every (infoset, class)."""
    pos = np.maximum(regret, 0.0)
    total = pos.sum(axis=2)
    return np.where(total > 0, pos[:, :, 1] / np.where(total > 0, total, 1), 0.5)


class PushFoldSolver:
    def __init__(self, workers: int | None = None) -> None:
        self.workers = max(1, workers or os.cpu_count() or 1)

    def _map(self, pool, fn, game, strat, seeds, per_worker):
        if pool is None:
            return [fn(game, strat, s, per_worker) for s in seeds]
        futures = [pool.submit(fn, game, strat, s, per_worker) for s in seeds]
        return [f.result() for f in futures]

    def solve(self, request: SolveRequest, progress=None) -> SolveResult:
        if request.game != "pushfold":
            raise ValueError(f"PushFoldSolver cannot solve {request.game!r}")
        game = PushFoldGame(int(request.params["players"]), float(request.params["stack_bb"]))
        if request.params.get("ante", 0):
            raise ValueError("antes are not supported yet")
        cfg = request.solver
        batches = int(cfg.get("batches", 200))
        deals_per_batch = int(cfg.get("deals_per_batch", 20_000))
        eval_deals = int(cfg.get("eval_deals", 200_000))
        seed = int(cfg.get("seed", 0))
        workers = min(self.workers, max(1, deals_per_batch // 500))
        per_worker = -(-deals_per_batch // workers)

        tr = _Traverser(game)
        regret = np.zeros((tr.n_inf, N_CLASSES, 2))
        strat_sum = np.zeros((tr.n_inf, N_CLASSES))
        visits = np.zeros((tr.n_inf, N_CLASSES), dtype=np.int64)
        weight_sum = 0.0
        t0 = time.perf_counter()
        pool = (ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
                if workers > 1 else None)
        try:
            for b in range(1, batches + 1):
                current = regret_matching(regret)
                strat = _strategy_lists(current)
                seeds = [derive_seed(seed, "batch", b, w) for w in range(workers)]
                for inc, vis in self._map(pool, _regret_batch, game, strat, seeds, per_worker):
                    regret += b * inc  # linear CFR: later batches weigh more
                    visits += vis
                strat_sum += b * current
                weight_sum += b
                if progress:
                    progress(b, batches)
            average = strat_sum / weight_sum

            # Evaluate on two independent halves so the best-response gain can
            # be bracketed: same-sample (upper, noise inflates the max) and
            # cross-fitted (choose on one half, score on the other: lower).
            halves = []
            for half in ("A", "B"):
                n_w = workers
                per_eval = -(-eval_deals // (2 * n_w))
                seeds = [derive_seed(seed, "eval", half, w) for w in range(n_w)]
                v_sum = np.zeros((tr.n_inf, N_CLASSES, 2))
                c_sum = np.zeros((tr.n_inf, N_CLASSES), dtype=np.int64)
                nd_sum = np.zeros(game.players)
                for v, c, nd in self._map(pool, _eval_batch, game, _strategy_lists(average), seeds, per_eval):
                    v_sum += v
                    c_sum += c
                    nd_sum += nd
                halves.append((v_sum, c_sum, nd_sum, per_eval * n_w))
        finally:
            if pool is not None:
                pool.shutdown()

        actor = np.array([k[0] for k in game.infosets()])
        q = average

        def on_policy(v):
            return (1 - q) * v[:, :, 0] + q * v[:, :, 1]

        (vA, cA, ndA, nA), (vB, cB, ndB, nB) = halves
        values, counts, no_dec, total_eval = vA + vB, cA + cB, ndA + ndB, nA + nB
        pickA = vA.argmax(axis=2)
        pickB = vB.argmax(axis=2)
        crossB = np.take_along_axis(vB, pickA[:, :, None], 2)[:, :, 0]
        crossA = np.take_along_axis(vA, pickB[:, :, None], 2)[:, :, 0]
        per_player = []
        for pl in range(game.players):
            rows = actor == pl
            u = (on_policy(values)[rows].sum() + no_dec[pl]) / total_eval
            upper = (values.max(axis=2)[rows].sum() + no_dec[pl]) / total_eval - u
            lower = 0.5 * ((crossB[rows].sum() - on_policy(vB)[rows].sum()) / nB
                           + (crossA[rows].sum() - on_policy(vA)[rows].sum()) / nA)
            per_player.append({"position": positions(game.players)[pl], "value_bb": float(u),
                               "gain_upper_bb": float(upper), "gain_lower_bb": float(lower)})
        nash_upper = sum(p["gain_upper_bb"] for p in per_player)
        nash_lower = sum(p["gain_lower_bb"] for p in per_player)

        keys = game.infosets()
        strategy = {}
        with np.errstate(invalid="ignore", divide="ignore"):
            ev = values / counts[:, :, None]
        for i, key in enumerate(keys):
            strategy[game.infoset_name(key)] = {
                "actions": list(game.actions(key)),
                "aggress_prob": [round(float(x), 4) for x in average[i]],
                "ev_bb": [[None if not np.isfinite(e) else round(float(e), 4) for e in row]
                          for row in ev[i]],
                "visits": [int(x) for x in visits[i]],
            }
        return SolveResult(
            request=request,
            strategy=strategy,
            iterations=batches * per_worker * workers,
            convergence={
                "metric": "nash_conv_bb_per_hand",
                "nash_conv_upper_bb": nash_upper,
                "nash_conv_lower_bb": nash_lower,
                "method": f"Monte Carlo best responses over {total_eval:,} deals in two halves; "
                          "upper = same-sample (noise inflates it), lower = cross-fitted",
                "per_player": per_player,
            },
            seconds=time.perf_counter() - t0,
            strategy_source=StrategySource.SOLVER_APPROX,
            notes=[
                "Abstracted game: push/fold only, equal stacks, blinds 0.5/1, no ante.",
                "Chip EV (no ICM). Approximate equilibrium; not exact GTO for full no-limit play.",
                "Cells with few visits are rarely reached and less converged; see 'visits'.",
            ],
        )


# ---------------------------------------------------------------------------
# Heads-up: exact best responses from a class-vs-class equity matrix


def _equity_rows(rows: list[int], samples: int, seed: int):
    from .hands import COMBOS

    rng = random.Random(seed)
    out = []
    for a in rows:
        eq = np.zeros(N_CLASSES)
        for b in range(N_CLASSES):
            if b < a:
                continue
            if b == a:
                eq[b] = 0.5  # identical classes split by symmetry
                continue
            pairs = [(x, y) for x in COMBOS[a] for y in COMBOS[b] if not set(x) & set(y)]
            win = 0.0
            for k in range(samples):
                x, y = pairs[k % len(pairs)]
                rest = [c for c in range(52) if c not in x and c not in y]
                board = rng.sample(rest, 5)
                sx, sy = evaluate((*x, *board)), evaluate((*y, *board))
                win += 1.0 if sx > sy else 0.5 if sx == sy else 0.0
            eq[b] = win / samples
        out.append((a, eq))
    return out


def hu_equity_matrix(samples: int = 2000, seed: int = 0, workers: int | None = None):
    """E[a, b] = equity of class a vs class b; W[a, b] = disjoint combo pairs."""
    from .hands import COMBOS

    W = np.array([[sum(1 for x in COMBOS[a] for y in COMBOS[b] if not set(x) & set(y))
                   for b in range(N_CLASSES)] for a in range(N_CLASSES)], dtype=float)
    E = np.zeros((N_CLASSES, N_CLASSES))
    workers = max(1, workers or os.cpu_count() or 1)
    chunks = [list(range(i, N_CLASSES, workers)) for i in range(workers)]
    seeds = [derive_seed(seed, "equity", i) for i in range(workers)]
    if workers == 1:
        results = [_equity_rows(chunks[0], samples, seeds[0])]
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            results = list(pool.map(_equity_rows, chunks, [samples] * workers, seeds))
    for part in results:
        for a, eq in part:
            for b in range(a, N_CLASSES):
                E[a, b] = eq[b]
                E[b, a] = 1 - eq[b]
    return E, W


def hu_exploitability(E: np.ndarray, W: np.ndarray, stack_bb: float,
                      push: np.ndarray, call: np.ndarray) -> dict[str, float]:
    """Exact NashConv (bb/hand) of a heads-up push/fold profile."""
    s = stack_bb
    P = W / W.sum()
    allin = 2 * s * E - s  # SB payoff when both are all-in
    # SB: per class a, value of push vs fold given BB's call probabilities.
    push_val = (P * ((1 - call)[None, :] * 1.0 + call[None, :] * allin)).sum(axis=1)
    fold_val = -0.5 * P.sum(axis=1)
    sb_u = (push * push_val + (1 - push) * fold_val).sum()
    sb_br = np.maximum(push_val, fold_val).sum()
    # BB: per class b, facing a push (weighted by the SB's push probability).
    reach = P * push[:, None]
    call_val = (reach * -allin).sum(axis=0)
    fold_bb = (reach * -1.0).sum(axis=0)
    bb_unopened = (P * (1 - push)[:, None] * 0.5).sum()
    bb_u = (call * call_val + (1 - call) * fold_bb).sum() + bb_unopened
    bb_br = np.maximum(call_val, fold_bb).sum() + bb_unopened
    return {"sb_gain_bb": float(sb_br - sb_u), "bb_gain_bb": float(bb_br - bb_u),
            "nash_conv_bb": float(sb_br - sb_u + bb_br - bb_u), "sb_value_bb": float(sb_u)}


def hu_cfr_plus(E: np.ndarray, W: np.ndarray, stack_bb: float, iterations: int = 5000):
    """Full-width CFR+ on the heads-up push/fold game (per-class matrix form)."""
    s = stack_bb
    P = W / W.sum()
    allin = 2 * s * E - s
    r_sb = np.zeros((N_CLASSES, 2))
    r_bb = np.zeros((N_CLASSES, 2))
    sum_push = np.zeros(N_CLASSES)
    sum_call = np.zeros(N_CLASSES)

    def rm(r):
        pos = np.maximum(r, 0)
        tot = pos.sum(axis=1)
        return np.where(tot > 0, pos[:, 1] / np.where(tot > 0, tot, 1), 0.5)

    for t in range(1, iterations + 1):
        push, call = rm(r_sb), rm(r_bb)
        push_val = (P * ((1 - call)[None, :] + call[None, :] * allin)).sum(axis=1)
        fold_val = -0.5 * P.sum(axis=1)
        v = push * push_val + (1 - push) * fold_val
        r_sb = np.maximum(r_sb + np.stack([fold_val - v, push_val - v], axis=1), 0)
        push = rm(r_sb)  # alternating updates
        reach = P * push[:, None]
        call_val = (reach * -allin).sum(axis=0)
        fold_bb = -reach.sum(axis=0)
        v = call * call_val + (1 - call) * fold_bb
        r_bb = np.maximum(r_bb + np.stack([fold_bb - v, call_val - v], axis=1), 0)
        sum_push += t * push
        sum_call += t * rm(r_bb)
    norm = iterations * (iterations + 1) / 2
    return sum_push / norm, sum_call / norm


def request(players: int, stack_bb: float, **solver: Any) -> SolveRequest:
    return SolveRequest("pushfold", {"players": players, "stack_bb": stack_bb,
                                     "small_blind_bb": 0.5, "ante": 0}, solver)
