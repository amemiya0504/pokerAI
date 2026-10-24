"""Bayesian opponent model (Milestone 8): Dirichlet posteriors over each
statistic's outcomes, shrunk toward a population prior.

For a group with outcomes o_1..o_k the model keeps raw counts per position
(the denominator is their sum). Two levels of shrinkage:

* player: Dirichlet(prior alpha + counts over all positions), where the
  prior is the population prior (``Prior``);
* position: Dirichlet(m * k + counts at the position), where m is the mean
  of the player's posterior from his *other* positions and k =
  min(``position_strength``, that posterior's total weight): positions are
  assumed similar, up to ``position_strength`` hands' worth of evidence.

Posterior means and equal-tailed credible intervals (each outcome's marginal
is a Beta distribution) come with the number of opportunities, so callers
(the exploit engine, M9) can scale their deviation with the confidence.
Pure Python: it runs in the web server, which has no NumPy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .stats import GROUPS, Observation, PublicHand, group_street, observations

MODEL_VERSION = "dirichlet-hier/1"


# ----------------------------------------------------------------------------- special functions


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(q: float, a: float, b: float) -> float:
    """Quantile of Beta(a, b) by bisection on the CDF."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def digamma(x: float) -> float:
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


# ----------------------------------------------------------------------------- prior


@dataclass
class Prior:
    """Population prior: Dirichlet pseudo-counts per group, with its source."""

    alpha: dict[str, tuple[float, ...]]
    source: str

    @classmethod
    def weak(cls) -> Prior:
        """One pseudo-count per outcome (Laplace). Not data: a placeholder
        until a population prior is fitted from real players' hands."""
        return cls({g: tuple(1.0 for _ in outs) for g, outs in GROUPS.items()},
                   "uninformative: one pseudo-count per outcome (no population data)")

    def get(self, group: str) -> tuple[float, ...]:
        return self.alpha.get(group) or tuple(1.0 for _ in GROUPS[group])

    def to_json(self) -> dict[str, Any]:
        return {"alpha": {g: list(a) for g, a in self.alpha.items()}, "source": self.source}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Prior:
        return cls({g: tuple(float(x) for x in a) for g, a in d["alpha"].items()}, d["source"])


def fit_prior(models: Iterable[OpponentModel], source: str, max_strength: float = 50.0,
              min_players: int = 3) -> Prior:
    """Empirical-Bayes population prior: per group, the Dirichlet-multinomial
    maximum-likelihood alpha over the players' pooled counts (Minka's fixed
    point), total weight capped at ``max_strength``. Groups with fewer than
    ``min_players`` players with data keep the weak prior."""
    models = list(models)
    weak = Prior.weak()
    alpha: dict[str, tuple[float, ...]] = {}
    for g, outs in GROUPS.items():
        data = [m.total(g) for m in models]
        data = [c for c in data if sum(c) > 0]
        if len(data) < min_players:
            alpha[g] = weak.get(g)
            continue
        k = len(outs)
        # Start from the pooled mean with weight 2 per outcome.
        pooled = [sum(c[i] for c in data) + 0.5 for i in range(k)]
        tot = sum(pooled)
        a = [2.0 * k * p / tot for p in pooled]
        for _ in range(500):
            a0 = sum(a)
            den = sum(digamma(sum(c) + a0) - digamma(a0) for c in data)
            new = []
            for i in range(k):
                num = sum(digamma(c[i] + a[i]) - digamma(a[i]) for c in data)
                new.append(max(a[i] * num / den, 1e-3) if den > 0 else a[i])
            done = max(abs(x - y) for x, y in zip(new, a)) < 1e-9
            a = new
            if sum(a) > max_strength * 10 or done:
                break
        if sum(a) > max_strength:
            a = [x * max_strength / sum(a) for x in a]
        alpha[g] = tuple(a)
    return Prior(alpha, source)


# ----------------------------------------------------------------------------- posterior


@dataclass(frozen=True)
class Estimate:
    group: str
    position: str | None  # None = all positions
    outcomes: tuple[str, ...]
    mean: dict[str, float]
    low: dict[str, float]
    high: dict[str, float]
    level: float
    opportunities: int  # observed decisions in this group (and position)
    alpha: tuple[float, ...]  # posterior Dirichlet parameters

    def sd(self, outcome: str) -> float:
        a0 = sum(self.alpha)
        ai = self.alpha[self.outcomes.index(outcome)]
        return math.sqrt(ai * (a0 - ai) / (a0 * a0 * (a0 + 1)))


def _estimate(group: str, position: str | None, alpha: list[float], n: int, level: float) -> Estimate:
    outs = GROUPS[group]
    a0 = sum(alpha)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    mean, low, high = {}, {}, {}
    for o, ai in zip(outs, alpha):
        mean[o] = ai / a0
        low[o] = beta_ppf(lo_q, ai, a0 - ai)
        high[o] = beta_ppf(hi_q, ai, a0 - ai)
    return Estimate(group, position, outs, mean, low, high, level, n, tuple(alpha))


@dataclass
class OpponentModel:
    """Counts and posteriors for one player."""

    prior: Prior = field(default_factory=Prior.weak)
    position_strength: float = 20.0
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)  # group -> position -> counts
    hands: int = 0

    def add(self, obs: Observation) -> None:
        by_pos = self.counts.setdefault(obs.group, {})
        c = by_pos.setdefault(obs.position, [0] * len(GROUPS[obs.group]))
        c[GROUPS[obs.group].index(obs.outcome)] += 1

    def observe(self, hand: PublicHand, seat: int) -> None:
        """Add one finished hand of the player sitting in ``seat``."""
        if seat not in hand.start_stacks:
            raise ValueError(f"seat {seat} is not in this hand")
        for o in observations(hand):
            if o.seat == seat:
                self.add(o)
        self.hands += 1

    def total(self, group: str, exclude: str | None = None) -> list[int]:
        k = len(GROUPS[group])
        out = [0] * k
        for pos, c in self.counts.get(group, {}).items():
            if pos != exclude:
                for i in range(k):
                    out[i] += c[i]
        return out

    def estimate(self, group: str, position: str | None = None, level: float = 0.9) -> Estimate:
        prior = list(self.prior.get(group))
        if position is None:
            c = self.total(group)
            return _estimate(group, None, [p + x for p, x in zip(prior, c)], sum(c), level)
        others = [p + x for p, x in zip(prior, self.total(group, exclude=position))]
        w = min(self.position_strength, sum(others))
        base = [w * a / sum(others) for a in others]
        c = self.counts.get(group, {}).get(position, [0] * len(prior))
        return _estimate(group, position, [b + x for b, x in zip(base, c)], sum(c), level)

    def estimates(self, level: float = 0.9) -> dict[str, Estimate]:
        return {g: self.estimate(g, level=level) for g in GROUPS}

    # --------------------------------------------------------------- storage

    def to_json(self) -> dict[str, Any]:
        return {"version": MODEL_VERSION, "prior": self.prior.to_json(), "position_strength": self.position_strength,
                "hands": self.hands, "counts": self.counts}

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> OpponentModel:
        if d.get("version") != MODEL_VERSION:
            raise ValueError(f"unsupported opponent model version {d.get('version')!r}")
        return cls(Prior.from_json(d["prior"]), float(d["position_strength"]),
                   {g: {p: list(c) for p, c in by.items()} for g, by in d["counts"].items()}, int(d["hands"]))

    def statistic_rows(self) -> list[dict[str, Any]]:
        """Rows for the ``player_statistics`` table: one per (outcome,
        position), with the group's opportunities as the denominator."""
        rows = []
        for g, by_pos in sorted(self.counts.items()):
            for pos, c in sorted(by_pos.items()):
                for o, hits in zip(GROUPS[g], c):
                    rows.append({"stat": f"{g}:{o}", "position": pos, "street": group_street(g),
                                 "n_opportunities": sum(c), "n_hits": hits})
        return rows

    @classmethod
    def from_statistic_rows(cls, rows: Iterable[Mapping[str, Any]], prior: Prior | None = None,
                            hands: int = 0) -> OpponentModel:
        m = cls(prior or Prior.weak())
        for r in rows:
            g, o = r["stat"].split(":")
            c = m.counts.setdefault(g, {}).setdefault(r["position"], [0] * len(GROUPS[g]))
            c[GROUPS[g].index(o)] = int(r["n_hits"])
        m.hands = hands
        return m
