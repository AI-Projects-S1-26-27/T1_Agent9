"""Independent best-response verifier for claimed equilibria.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Deliberately written WITHOUT reusing anything from the solvers (no shared
helpers, no linear algebra). It only computes, by direct summation, what each
player would earn from every pure action against the other's mix, and checks
that no action beats what the claimed strategy earns. That is the definition
of a Nash equilibrium: no profitable unilateral deviation. Checking against
pure deviations is enough, because a mixed deviation earns a weighted
average of pure ones and so can never beat the best pure deviation.

If a solver bug produced a wrong "equilibrium", this check would catch it,
because the check does not share the solver's code paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from resource_war.encounter.matrix import BimatrixGame


def _check_distribution(p: Sequence[Fraction], size: int, who: str) -> None:
    if len(p) != size:
        raise ValueError(f"{who} strategy has {len(p)} entries; the game needs {size}")
    if any(v < 0 for v in p) or sum(p) != 1:
        raise ValueError(f"{who} strategy must be non-negative and sum to exactly 1, got {list(map(str, p))}")


def deviation_gains(game: BimatrixGame, x: Sequence[Fraction], y: Sequence[Fraction]) -> tuple[Fraction, Fraction]:
    """Best gain each player could get by switching alone (0 means no profitable deviation)."""
    rows = game.row_payoffs
    cols = game.col_payoffs
    m, n = len(rows), len(rows[0])
    _check_distribution(x, m, "row")
    _check_distribution(y, n, "column")
    row_action_values = []
    for i in range(m):
        total = Fraction(0)
        for j in range(n):
            total += rows[i][j] * y[j]
        row_action_values.append(total)
    col_action_values = []
    for j in range(n):
        total = Fraction(0)
        for i in range(m):
            total += x[i] * cols[i][j]
        col_action_values.append(total)
    row_current = sum((x[i] * row_action_values[i] for i in range(m)), Fraction(0))
    col_current = sum((y[j] * col_action_values[j] for j in range(n)), Fraction(0))
    return max(row_action_values) - row_current, max(col_action_values) - col_current


def is_nash(game: BimatrixGame, x: Sequence[Fraction], y: Sequence[Fraction]) -> bool:
    row_gain, col_gain = deviation_gains(game, x, y)
    return row_gain == 0 and col_gain == 0
