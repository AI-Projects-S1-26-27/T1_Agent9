"""Pure-strategy Nash equilibria by best-response checking.

Owner: Atharv Singhal, Reviewer: Gursidak Singh Bassi

A cell (i, j) is a pure Nash equilibrium when i is a best response to j AND
j is a best response to i: neither player can gain by switching alone.
We report all pure best responses (ties included), because a tie between
best responses is exactly what makes a game degenerate (e.g. A and E both
answer E at V = 4).
"""

from __future__ import annotations

from resource_war.encounter.matrix import BimatrixGame


def row_best_responses(game: BimatrixGame, col: int) -> list[int]:
    m, _ = game.shape
    best = max(game.row_payoffs[i][col] for i in range(m))
    return [i for i in range(m) if game.row_payoffs[i][col] == best]


def col_best_responses(game: BimatrixGame, row: int) -> list[int]:
    _, n = game.shape
    best = max(game.col_payoffs[row][j] for j in range(n))
    return [j for j in range(n) if game.col_payoffs[row][j] == best]


def pure_nash_equilibria(game: BimatrixGame) -> list[tuple[int, int]]:
    """All pure equilibria (i, j) in row-major order."""
    m, n = game.shape
    return [
        (i, j)
        for i in range(m)
        for j in range(n)
        if i in row_best_responses(game, j) and j in col_best_responses(game, i)
    ]


def is_strict_pure_equilibrium(game: BimatrixGame, i: int, j: int) -> bool:
    """Strict: each player's equilibrium action is its UNIQUE best response."""
    return row_best_responses(game, j) == [i] and col_best_responses(game, i) == [j]

# Verified: Atharv (pure Nash)
