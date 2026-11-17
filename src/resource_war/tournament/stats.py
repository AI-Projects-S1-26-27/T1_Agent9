"""Confidence intervals and the decision rule for "X beats Y".

Owner: Manas Tripathi, Reviewer: Jainam Parekh Naresh

- Win rates use the Wilson score interval. Unlike the textbook "p +- 1.96 *
  sqrt(p(1-p)/n)" interval, it stays inside [0, 1] and behaves well near 0
  and 1 and for moderate n.
- Differences in means use a percentile bootstrap that resamples whole
  CLUSTERS (a seed's two side-swapped games, a battle start position's two
  first-mover orders, or one game's encounters). Observations inside a
  cluster share a map or a start and are correlated; resampling single
  observations would understate the uncertainty.
- Decision rule (pre-registered): X beats Y iff the 95% CI of the
  difference lies entirely above 0. Ties are always reported separately,
  never folded into wins.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def above_zero(self) -> bool:
        return self.low > 0

    def below_zero(self) -> bool:
        return self.high < 0

    def fmt(self, digits: int = 3) -> str:
        return f"{self.estimate:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    if n <= 0:
        raise ValueError("Wilson interval needs at least one trial")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be in [0, {n}], got {successes}")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_mean(
    clusters: Sequence[Sequence[float]], rng: np.random.Generator, resamples: int = 10000, confidence: float = 0.95
) -> Interval:
    """Percentile-bootstrap CI for the mean of all observations, resampling whole clusters."""
    clusters = [np.asarray(c, dtype=float) for c in clusters if len(c)]
    if not clusters:
        raise ValueError("no data to bootstrap")
    sums = np.array([c.sum() for c in clusters])
    sizes = np.array([len(c) for c in clusters])
    estimate = float(sums.sum() / sizes.sum())
    idx = rng.integers(0, len(clusters), size=(resamples, len(clusters)))
    means = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Interval(estimate, float(low), float(high))


def bootstrap_mean(values: Sequence[float], rng: np.random.Generator, resamples: int = 10000, confidence: float = 0.95) -> Interval:
    """Bootstrap CI for a mean of independent observations (each its own cluster)."""
    return cluster_bootstrap_mean([[v] for v in values], rng, resamples, confidence)


def verdict(interval: Interval) -> str:
    """'X beats Y' / 'Y beats X' / 'no detectable difference' for a CI of (X - Y)."""
    if interval.above_zero():
        return "first beats second"
    if interval.below_zero():
        return "second beats first"
    return "no detectable difference"
