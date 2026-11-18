
"""Exact equal-support enumeration for two-player bimatrix games.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

For each support pair, solve the indifference equations exactly.
Keep positive distributions only when unused actions cannot improve
either player's payoff.

Equal-sized supports suffice for nondegenerate games. Degenerate games
may need vertex enumeration, coordinated by solve.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

from resource_war.equilibrium.linalg import solve_exact

if TYPE_CHECKING:
    from resource_war.encounter.matrix import BimatrixGame


@dataclass(frozen=True)
class Equilibrium:
    row: tuple[Fraction, ...]
    col: tuple[Fraction, ...]
    row_payoff: Fraction
    col_payoff: Fraction

    @property
    def row_support(self) -> tuple[int, ...]:
        return tuple(i for i, probability in enumerate(self.row) if probability > 0)

    @property
    def col_support(self) -> tuple[int, ...]:
        return tuple(j for j, probability in enumerate(self.col) if probability > 0)

    @property
    def is_pure(self) -> bool:
        return len(self.row_support) == 1 and len(self.col_support) == 1


@dataclass
class SupportEnumResult:
    equilibria: list[Equilibrium]
    singular_supports: list[
        tuple[tuple[int, ...], tuple[int, ...]]
    ] = field(default_factory=list)
    extra_best_responses: bool = False

    @property
    def degenerate(self) -> bool:
        """Conservative signal; solve.py uses the exact polytope test."""
        return bool(self.singular_supports) or self.extra_best_responses


def _indifferent_mix(
    payoffs,
    own_support,
    other_support,
    n_other: int,
    transpose: bool,
):
    """Find an opponent mix making supported own actions equally good.

    Unknowns are the supported probabilities and a common payoff.
    The final equation requires the probabilities to sum to one.
    """
    k = len(other_support)
    coefficients = []
    constants = []

    for own_action in own_support:
        row = [
            payoffs[other_action][own_action]
            if transpose
            else payoffs[own_action][other_action]
            for other_action in other_support
        ]
        coefficients.append(row + [Fraction(-1)])
        constants.append(Fraction(0))

    coefficients.append([Fraction(1)] * k + [Fraction(0)])
    constants.append(Fraction(1))

    solution = solve_exact(coefficients, constants)
    if solution.status != "unique":
        return solution.status, None, None

    assert solution.x is not None
    mix = [Fraction(0)] * n_other
    for index, action in enumerate(other_support):
        mix[action] = solution.x[index]

    return "unique", tuple(mix), solution.x[k]


def support_enumeration(game: BimatrixGame) -> SupportEnumResult:
    """Enumerate candidates with equally sized row and column supports."""
    m, n = game.shape
    a, b = game.row_payoffs, game.col_payoffs
    result = SupportEnumResult([])

    for k in range(1, min(m, n) + 1):
        for rows in combinations(range(m), k):
            for cols in combinations(range(n), k):
                y_status, y, u = _indifferent_mix(
                    a, rows, cols, n, transpose=False
                )
                x_status, x, v = _indifferent_mix(
                    b, cols, rows, m, transpose=True
                )

                if "infinite" in (y_status, x_status):
                    result.singular_supports.append((rows, cols))
                    continue

                if x is None or y is None:
                    continue

                if any(x[i] <= 0 for i in rows):
                    continue
                if any(y[j] <= 0 for j in cols):
                    continue

                row_values = [
                    sum(
                        (a[i][j] * y[j] for j in range(n)),
                        Fraction(0),
                    )
                    for i in range(m)
                ]
                col_values = [
                    sum(
                        (x[i] * b[i][j] for i in range(m)),
                        Fraction(0),
                    )
                    for j in range(n)
                ]

                if max(row_values) > u or max(col_values) > v:
                    continue

                if row_values.count(u) > k or col_values.count(v) > k:
                    result.extra_best_responses = True

                result.equilibria.append(Equilibrium(x, y, u, v))

    return result
