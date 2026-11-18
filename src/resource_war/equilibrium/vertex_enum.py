
"""Exact vertex enumeration of labelled best-response polytopes.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Positive payoff shifts preserve best responses. We construct:
    P = {x >= 0: B_positive.T*x <= 1}
    Q = {y >= 0: A_positive*y <= 1}

Completely labelled pairs of nonzero vertices yield equilibria after
normalization. In degenerate games this lists extreme equilibria,
not every member of a possible continuum.

Designed for the project's small games; enumeration is combinatorial.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import TYPE_CHECKING

from resource_war.equilibrium.linalg import solve_exact
from resource_war.equilibrium.support_enum import Equilibrium

if TYPE_CHECKING:
    from resource_war.encounter.matrix import BimatrixGame


def _vertices(
    constraints: list[tuple[list[Fraction], Fraction]],
    dim: int,
) -> dict[tuple[Fraction, ...], frozenset[int]]:
    """Enumerate feasible vertices and all their tight-constraint labels."""
    found: dict[tuple[Fraction, ...], frozenset[int]] = {}

    for tight in combinations(range(len(constraints)), dim):
        solution = solve_exact(
            [constraints[index][0] for index in tight],
            [constraints[index][1] for index in tight],
        )
        if solution.status != "unique":
            continue

        assert solution.x is not None
        point = solution.x

        values = [
            sum(
                (coefficients[k] * point[k] for k in range(dim)),
                Fraction(0),
            )
            for coefficients, _ in constraints
        ]

        if any(
            value > bound
            for value, (_, bound) in zip(values, constraints)
        ):
            continue

        labels = frozenset(
            index
            for index, (value, (_, bound)) in enumerate(
                zip(values, constraints)
            )
            if value == bound
        )
        found[point] = labels

    return found


def _polytopes(game: BimatrixGame):
    """Construct the vertices of both best-response polytopes."""
    m, n = game.shape
    a, b = game.row_payoffs, game.col_payoffs

    smallest = min(
        min(min(row) for row in a),
        min(min(row) for row in b),
    )
    shift = Fraction(1) - smallest

    a_positive = [[value + shift for value in row] for row in a]
    b_positive = [[value + shift for value in row] for row in b]

    one, zero = Fraction(1), Fraction(0)

    # P labels 0..m-1: row probabilities are zero.
    p_constraints = [
        ([-one if k == i else zero for k in range(m)], zero)
        for i in range(m)
    ]

    # P labels m..m+n-1: column payoff constraints are tight.
    p_constraints += [
        ([b_positive[i][j] for i in range(m)], one)
        for j in range(n)
    ]

    # Q labels 0..m-1: row payoff constraints are tight.
    q_constraints = [
        ([a_positive[i][j] for j in range(n)], one)
        for i in range(m)
    ]

    # Q labels m..m+n-1: column probabilities are zero.
    q_constraints += [
        ([-one if k == j else zero for k in range(n)], zero)
        for j in range(n)
    ]

    return (
        _vertices(p_constraints, m),
        _vertices(q_constraints, n),
    )


def is_degenerate(game: BimatrixGame) -> bool:
    """Check whether a polytope vertex has more labels than its dimension."""
    m, n = game.shape
    p_vertices, q_vertices = _polytopes(game)

    return (
        any(len(labels) > m for labels in p_vertices.values())
        or any(len(labels) > n for labels in q_vertices.values())
    )


def vertex_enumeration(game: BimatrixGame) -> list[Equilibrium]:
    """Return normalized, completely labelled nonzero vertex pairs."""
    m, n = game.shape
    a, b = game.row_payoffs, game.col_payoffs
    p_vertices, q_vertices = _polytopes(game)
    all_labels = frozenset(range(m + n))

    found = {}

    for x, x_labels in p_vertices.items():
        if all(value == 0 for value in x):
            continue

        for y, y_labels in q_vertices.items():
            if all(value == 0 for value in y):
                continue
            if (x_labels | y_labels) != all_labels:
                continue

            x_total, y_total = sum(x), sum(y)
            row_mix = tuple(value / x_total for value in x)
            col_mix = tuple(value / y_total for value in y)

            row_payoff = sum(
                (
                    row_mix[i] * a[i][j] * col_mix[j]
                    for i in range(m)
                    for j in range(n)
                ),
                Fraction(0),
            )
            col_payoff = sum(
                (
                    row_mix[i] * b[i][j] * col_mix[j]
                    for i in range(m)
                    for j in range(n)
                ),
                Fraction(0),
            )

            found[(row_mix, col_mix)] = Equilibrium(
                row_mix, col_mix, row_payoff, col_payoff
            )

    return sorted(found.values(), key=lambda eq: (eq.row, eq.col))
