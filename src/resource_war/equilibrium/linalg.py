
"""Exact linear-system solving using Gauss-Jordan elimination.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Uses Fraction arithmetic. Returns one of three statuses:
unique, none (inconsistent), or infinite (underdetermined).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class LinearSolution:
    status: str
    x: tuple[Fraction, ...] | None = None


def solve_exact(
    a: Sequence[Sequence[Fraction]],
    b: Sequence[Fraction],
) -> LinearSolution:
    """Solve a*x=b exactly, including rectangular systems.

    Inputs should contain integers or exact rational values.
    Malformed dimensions raise ValueError.
    """
    rows = len(a)
    cols = len(a[0]) if rows else 0

    if len(b) != rows:
        raise ValueError("right-hand side must have one entry per row")
    if any(len(row) != cols for row in a):
        raise ValueError("coefficient matrix must not be ragged")

    augmented = [
        [Fraction(value) for value in a[i]] + [Fraction(b[i])]
        for i in range(rows)
    ]

    pivots: list[int] = []
    pivot_row = 0

    for col in range(cols):
        pivot = next(
            (
                i for i in range(pivot_row, rows)
                if augmented[i][col] != 0
            ),
            None,
        )
        if pivot is None:
            continue

        augmented[pivot_row], augmented[pivot] = (
            augmented[pivot],
            augmented[pivot_row],
        )

        divisor = augmented[pivot_row][col]
        augmented[pivot_row] = [
            value / divisor for value in augmented[pivot_row]
        ]

        for i in range(rows):
            if i == pivot_row or augmented[i][col] == 0:
                continue
            factor = augmented[i][col]
            augmented[i] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[i], augmented[pivot_row]
                )
            ]

        pivots.append(col)
        pivot_row += 1
        if pivot_row == rows:
            break

    for row in augmented:
        if all(value == 0 for value in row[:-1]) and row[-1] != 0:
            return LinearSolution("none")

    if len(pivots) < cols:
        return LinearSolution("infinite")

    solution = [Fraction(0)] * cols
    for row_index, col_index in enumerate(pivots):
        solution[col_index] = augmented[row_index][-1]

    return LinearSolution("unique", tuple(solution))
