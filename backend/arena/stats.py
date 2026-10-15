"""bb/100 with confidence intervals from duplicate blocks.

In duplicate play each deal is replayed once per seat rotation, so hands of
one deal are correlated. The independent unit is the deal (block): an
agent's mean result over the deal's rotations. The mean of the block means
is the per-hand estimate and its standard error comes from the spread of the
block means, with a Student-t quantile for the interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


def t_quantile(p: float, df: int) -> float:
    """Student-t quantile via the Cornish-Fisher expansion (good for df >= 3)."""
    z = NormalDist().inv_cdf(p)
    if df <= 0:
        raise ValueError("df must be positive")
    g1 = (z**3 + z) / 4
    g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    g3 = (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / 384
    g4 = (79 * z**9 + 776 * z**7 + 1482 * z**5 - 1920 * z**3 - 945 * z) / 92160
    return z + g1 / df + g2 / df**2 + g3 / df**3 + g4 / df**4


@dataclass(frozen=True)
class AgentStats:
    agent: str
    hands: int
    blocks: int
    bb_per_100: float
    stddev: float  # standard deviation of block means, in bb/100
    ci_low: float
    ci_high: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def summarize(agent: str, block_means_bb: list[float], hands: int,
              confidence: float = 0.95) -> AgentStats:
    """``block_means_bb``: per-deal mean result in big blinds per hand."""
    n = len(block_means_bb)
    if n == 0:
        raise ValueError("no blocks")
    mean = sum(block_means_bb) / n
    if n == 1:
        return AgentStats(agent, hands, n, mean * 100, math.nan, -math.inf, math.inf)
    var = sum((x - mean) ** 2 for x in block_means_bb) / (n - 1)
    sd = math.sqrt(var)
    half = t_quantile(0.5 + confidence / 2, n - 1) * sd / math.sqrt(n)
    return AgentStats(agent, hands, n, mean * 100, sd * 100, (mean - half) * 100, (mean + half) * 100)
