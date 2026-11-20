"""Pareto optimality over the pure outcomes of a bimatrix game.

Owner: Atharv Singhal, Reviewer: Gursidak Singh Bassi

An outcome (u1, u2) Pareto-dominates (v1, v2) when u1 >= v1 and u2 >= v2 with
at least one strict inequality: one player is better off and nobody is worse
off. An outcome is Pareto-optimal when no outcome dominates it.

Scope: we compare the pure outcomes (the cells of the matrix), which is
what the analysis card shows. ``pareto_dominating_outcomes`` answers the
question the proposal promised: is an equilibrium's payoff vector beaten by
some cell for BOTH players? At V = 6 the answer is yes: (E,E) = (3,3) beats
the equilibrium (51/27, 51/27), so rational play leaves both players worse
off, the same inefficiency as in the Prisoner's Dilemma.
"""

from __future__ import annotations

from fractions import Fraction

from resource_war.encounter.matrix import BimatrixGame


def pareto_dominates(u: tuple[Fraction, Fraction], v: tuple[Fraction, Fraction]) -> bool:
    return u[0] >= v[0] and u[1] >= v[1] and (u[0] > v[0] or u[1] > v[1])


def pareto_optimal_outcomes(game: BimatrixGame) -> list[tuple[int, int]]:
    m, n = game.shape
    cells = [(i, j) for i in range(m) for j in range(n)]
    return [
        cell
        for cell in cells
        if not any(pareto_dominates(game.payoff(*other), game.payoff(*cell)) for other in cells)
    ]


def pareto_dominating_outcomes(game: BimatrixGame, payoff: tuple[Fraction, Fraction]) -> list[tuple[int, int]]:
    """Cells whose payoffs Pareto-dominate ``payoff`` (e.g. an equilibrium's payoffs)."""
    m, n = game.shape
    return [(i, j) for i in range(m) for j in range(n) if pareto_dominates(game.payoff(i, j), payoff)]

# Verified: Atharv (Pareto)
