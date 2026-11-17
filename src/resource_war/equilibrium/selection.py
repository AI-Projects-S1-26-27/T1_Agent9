"""Deterministic equilibrium selection for the Equilibrium agent.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

When a game has several equilibria, "play your equilibrium strategy" is not
well defined. The rule (config/agents.yaml):
1. prefer the equilibrium with the highest expected payoff for ME;
2. break remaining ties by the lexicographically SMALLEST (my strategy,
   opponent strategy) pair of probability vectors in A, D, E order.

At V = 4 both (E,E) and (1/2 A + 1/2 E, E) give the row player 2; the
lexicographic rule picks (0, 0, 1) = pure E. Because the encounter game is
symmetric, the opponent's rule (run from its own point of view) also picks
E, so two Equilibrium agents coordinate on (E,E).

Known caveat (DESIGN 6): in games such as Battle of the Sexes the two
players' favourite equilibria differ, so two agents using this rule
independently can mis-coordinate. Our encounter matrices only have multiple
equilibria at V = 2c, where both pick (E,E).
"""

from __future__ import annotations

from collections.abc import Sequence

from resource_war.equilibrium.support_enum import Equilibrium


def select_equilibrium(equilibria: Sequence[Equilibrium]) -> Equilibrium:
    """Pick one equilibrium from the ROW player's point of view."""
    if not equilibria:
        raise ValueError("no equilibria to select from (every finite game has one; this is a bug)")
    best = max(e.row_payoff for e in equilibria)
    return min((e for e in equilibria if e.row_payoff == best), key=lambda e: (e.row, e.col))
