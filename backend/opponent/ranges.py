"""Opponent model -> range tracking (replaces the self-play assumption).

The real-time agent updates the opponent's range with P(observed action |
combo). Under the self-play assumption that is the equilibrium strategy of
the re-solve. With an opponent model the strategy is *tilted*: the smallest
change (in KL divergence) that makes the range-weighted frequency of each
action class (fold / check / call / bet / raise) equal a target, while
keeping which combos the equilibrium sends to which action. The target is
the opponent's observed frequency in this decision context, shrunk toward
the equilibrium's own frequency with ``strength`` pseudo-observations
("adjust the GTO baseline by confidence", design doc section 4): with no
data it is the equilibrium itself.

Preflop, the top-X% range width of each line comes from the opponent's
frequencies in the context of his last preflop action, shrunk toward the
heuristic width the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .model import OpponentModel
from .stats import GROUPS

EPSILON = 0.02  # share of every action kept for mistakes, so no observed action is impossible


def action_class(label: str, facing_bet: bool) -> str:
    """Solver action label -> statistics outcome."""
    if label == "f":
        return "fold"
    if label == "x":
        return "check"
    if label == "c":
        return "call"
    if label == "a":
        return "raise" if facing_bet else "bet"
    return "raise" if label[0] == "r" else "bet"


def postflop_group(street: str, facing_bet: bool, raises: int) -> str:
    """Context group of a postflop decision (``raises``: raises this street, bets excluded)."""
    if not facing_bet:
        return f"{street}_unbet"
    return f"{street}_vs_bet" if raises == 0 else f"{street}_vs_raise"


def target_frequencies(counts: Sequence[int], outcomes: Sequence[str], baseline: Mapping[str, float],
                       strength: float) -> dict[str, float]:
    """Posterior mean of each available outcome's frequency with a Dirichlet
    prior of weight ``strength`` centred on ``baseline`` (renormalised over
    the outcomes in ``baseline``, i.e. the ones available here)."""
    avail = [o for o in outcomes if o in baseline]
    n = {o: counts[list(outcomes).index(o)] for o in avail}
    total = sum(n.values())
    base = sum(baseline[o] for o in avail)
    out = {o: n[o] + strength * baseline[o] / base for o in avail}
    z = total + strength
    return {o: v / z for o, v in out.items()}


def tilt(sigma: np.ndarray, reach: np.ndarray, classes: Sequence[str], target: Mapping[str, float],
         iterations: int = 200) -> np.ndarray:
    """``sigma`` (A, N) action probabilities per combo; returns the I-projection
    whose reach-weighted class frequencies equal ``target`` (iterative scaling)."""
    classes = list(classes)
    names = sorted(set(classes))
    idx = np.array([names.index(c) for c in classes])
    w = np.ones(len(names))
    mass = reach / max(reach.sum(), 1e-300)
    want = np.array([target.get(c, 0.0) for c in names])
    want = want / want.sum()
    p = sigma
    for _ in range(iterations):
        p = sigma * w[idx][:, None]
        p = p / np.maximum(p.sum(axis=0, keepdims=True), 1e-300)
        marg = np.zeros(len(names))
        np.add.at(marg, idx, p @ mass)
        ratio = np.where(marg > 0, want / np.maximum(marg, 1e-300), 1.0)
        if np.max(np.abs(ratio - 1)) < 1e-9:
            break
        w = np.clip(w * ratio, 1e-9, 1e9)
    return p


def modelled_strategy(sigma: np.ndarray, reach: np.ndarray, labels: Sequence[str], facing_bet: bool,
                      group: str, model: OpponentModel | None, strength: float,
                      fold_only: bool = False) -> tuple[np.ndarray, str]:
    """The opponent's action probabilities per combo at a solved node: the
    equilibrium ``sigma`` mixed with ``EPSILON`` uniform and tilted to the
    model's frequencies. ``fold_only`` raises only the fold frequency, preserving
    the baseline's relative non-fold mix. Returns (probabilities, note)."""
    a = sigma.shape[0]
    mixed = (1 - EPSILON) * sigma + EPSILON / a
    classes = [action_class(lb, facing_bet) for lb in labels]
    mass = reach / max(reach.sum(), 1e-300)
    baseline: dict[str, float] = {}
    for c, row in zip(classes, mixed):
        baseline[c] = baseline.get(c, 0.0) + float(row @ mass)
    counts = model.total(group) if model is not None and group in GROUPS else [0] * len(GROUPS.get(group, ()))
    if model is None or sum(counts) == 0 or set(baseline) - set(GROUPS[group]):
        return mixed, f"{group}: no observations, equilibrium frequencies"
    target = target_frequencies(counts, GROUPS[group], baseline, strength)
    if fold_only and "fold" in target:
        fold = max(baseline["fold"], target["fold"])
        remaining = max(1 - baseline["fold"], 1e-30)
        target = {o: fold if o == "fold" else (1 - fold) * baseline[o] / remaining for o in target}
    note = f"{group}: {sum(counts)} observations, " + ", ".join(
        f"{o} {baseline[o]:.2f}->{target[o]:.2f}" for o in target)
    return tilt(mixed, reach, classes, target), note


# ----------------------------------------------------------------------------- preflop

# Heuristic frequency priors per preflop context (fold, call, raise) /
# (check, raise); line widths of the self-play tracker are sums of these.
PREFLOP_PRIOR = {
    "pf_open": {"fold": 0.30, "call": 0.50, "raise": 0.20},
    "pf_option": {"check": 0.80, "raise": 0.20},
    "pf_vs_raise": {"fold": 0.68, "call": 0.24, "raise": 0.08},
    "pf_vs_3bet": {"fold": 0.74, "call": 0.20, "raise": 0.06},
    "pf_vs_4bet": {"fold": 0.74, "call": 0.20, "raise": 0.06},
}
CALLER_TOP_WEIGHT = 0.3  # a calling range still holds some of the hands that usually raise


def preflop_width(model: OpponentModel | None, group: str, outcome: str, strength: float
                  ) -> tuple[float, float, int]:
    """(width of the range, width of its down-weighted top, observations) for
    a seat whose last preflop action was ``outcome`` in ``group``."""
    prior = PREFLOP_PRIOR[group]
    counts = model.total(group) if model is not None else [0] * len(GROUPS[group])
    f = target_frequencies(counts, GROUPS[group], prior, strength)
    if outcome == "raise":
        return f["raise"], 0.0, sum(counts)
    return min(1.0, f.get("call", f.get("check", 0.0)) + f["raise"]), f["raise"], sum(counts)
