"""Independent best-response verifier for claimed Nash equilibria.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

The verifier does not import solver helpers or linear algebra. It directly
calculates each pure action's expected payoff against the opponent's mix.

A profile is a Nash equilibrium when neither player can improve by
changing strategy alone. Checking pure deviations is sufficient because
a mixed deviation is a weighted average of pure-action payoffs.

Inputs must use exact rational values, normally fractions.Fraction.
The game is expected to contain validated, nonempty payoff matrices
of the same shape.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resource_war.encounter.matrix import BimatrixGame


def _check_distribution(
    p: Sequence[Fraction],
    size: int,
    who: str,
) -> None:
    """Reject a strategy with the wrong length or invalid probabilities."""
    if len(p) != size:
        raise ValueError(
            f"{who} strategy has {len(p)} entries; the game needs {size}"
        )

    if any(v < 0 for v in p) or sum(p) != 1:
        raise ValueError(
            f"{who} strategy must be non-negative and sum to exactly 1, "
            f"got {list(map(str, p))}"
        )


def deviation_gains(
    game: BimatrixGame,
    x: Sequence[Fraction],
    y: Sequence[Fraction],
) -> tuple[Fraction, Fraction]:
    """Return each player's maximum gain from changing strategy alone.

    x is the row player's distribution; y is the column player's.
    A gain of zero means that player has no profitable deviation.

    Raises:
        ValueError: If either probability distribution is invalid.
    """
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

    row_current = sum(
        (x[i] * row_action_values[i] for i in range(m)),
        Fraction(0),
    )
    col_current = sum(
        (y[j] * col_action_values[j] for j in range(n)),
        Fraction(0),
    )

    return (
        max(row_action_values) - row_current,
        max(col_action_values) - col_current,
    )


def is_nash(
    game: BimatrixGame,
    x: Sequence[Fraction],
    y: Sequence[Fraction],
) -> bool:
    """Return whether both players have zero profitable deviation.

    Invalid distributions raise ValueError rather than returning False.
    """
    row_gain, col_gain = deviation_gains(game, x, y)
    return row_gain == 0 and col_gain == 0