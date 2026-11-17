"""Exact linear-system solving over the rationals (Gauss-Jordan elimination).

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

numpy.linalg.solve works in floats and cannot tell "singular" from "nearly
singular". Equilibrium computation needs that distinction: a singular
indifference system is the signature of a degenerate game (a whole line of
solutions instead of one point). With Fractions the rank is exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class LinearSolution:
    status: str  # "unique", "none" (inconsistent) or "infinite" (singular, consistent)
    x: tuple[Fraction, ...] | None = None


def solve_exact(a: Sequence[Sequence[Fraction]], b: Sequence[Fraction]) -> LinearSolution:
    """Solve a x = b exactly. ``a`` may be non-square."""
    rows = len(a)
    cols = len(a[0]) if rows else 0
    m = [list(map(Fraction, a[i])) + [Fraction(b[i])] for i in range(rows)]
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        pivot = next((i for i in range(r, rows) if m[i][c] != 0), None)
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        pv = m[r][c]
        m[r] = [v / pv for v in m[r]]
        for i in range(rows):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    if any(all(v == 0 for v in row[:-1]) and row[-1] != 0 for row in m):
        return LinearSolution("none")
    if len(pivots) < cols:
        return LinearSolution("infinite")
    x = [Fraction(0)] * cols
    for i, c in enumerate(pivots):
        x[c] = m[i][-1]
    return LinearSolution("unique", tuple(x))
