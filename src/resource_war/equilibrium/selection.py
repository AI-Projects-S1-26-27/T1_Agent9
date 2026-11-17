
"""Deterministic selection from the row player's perspective.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Prefer the largest own expected payoff, then the lexicographically
smallest (own strategy, opponent strategy) pair.

Selecting independently does not guarantee coordination in arbitrary
games where players prefer different equilibria.
"""

from __future__ import annotations

from collections.abc import Sequence

from resource_war.equilibrium.support_enum import Equilibrium


def select_equilibrium(
    equilibria: Sequence[Equilibrium],
) -> Equilibrium:
    """Select maximum row payoff, then lexicographically smallest profile."""
    if not equilibria:
        raise ValueError("no equilibria to select from")

    best_payoff = max(eq.row_payoff for eq in equilibria)

    return min(
        (
            eq for eq in equilibria
            if eq.row_payoff == best_payoff
        ),
        key=lambda eq: (eq.row, eq.col),
    )
