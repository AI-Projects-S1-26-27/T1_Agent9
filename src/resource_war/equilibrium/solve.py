
"""Coordinate exact equilibrium enumeration and independent verification.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Nondegenerate games use support enumeration.
Degenerate games use vertex enumeration to return extreme equilibria.
Every returned strategy pair passes the independent best-response check.

The game must be validated, immutable, and hashable for caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from resource_war.equilibrium.support_enum import (
    Equilibrium,
    support_enumeration,
)
from resource_war.equilibrium.verify import is_nash
from resource_war.equilibrium.vertex_enum import (
    is_degenerate,
    vertex_enumeration,
)

if TYPE_CHECKING:
    from resource_war.encounter.matrix import BimatrixGame


@dataclass(frozen=True)
class EquilibriumSet:
    equilibria: tuple[Equilibrium, ...]
    degenerate: bool
    method: str
    support_enum_found: int
    singular_supports: tuple[
        tuple[tuple[int, ...], tuple[int, ...]], ...
    ] = field(default=())

    def notes(self) -> list[str]:
        if self.degenerate:
            notes = [
                "Degenerate game: equilibria may form a continuum.",
                (
                    f"Support enumeration found {self.support_enum_found}; "
                    f"vertex enumeration returns {len(self.equilibria)} "
                    "extreme equilibria, not the entire continuum."
                ),
            ]
        else:
            notes = [
                "Nondegenerate game: support enumeration finds every equilibrium."
            ]

        if self.singular_supports:
            notes.append(
                "Singular indifference systems encountered for "
                f"{len(self.singular_supports)} support pair(s)."
            )

        return notes


@lru_cache(maxsize=4096)
def solve_game(game: BimatrixGame) -> EquilibriumSet:
    """Find equilibria and independently check every returned profile."""
    support_result = support_enumeration(game)

    # Singular candidate systems alone do not prove game degeneracy:
    # their solutions may not be feasible equilibrium strategies.
    # Use the labelled-polytope criterion for the final classification.
    degenerate = is_degenerate(game)

    if degenerate:
        candidates = vertex_enumeration(game)
        method = "vertex_enumeration"
    else:
        candidates = support_result.equilibria
        method = "support_enumeration"

    unique = {(eq.row, eq.col): eq for eq in candidates}
    equilibria = tuple(
        sorted(unique.values(), key=lambda eq: (eq.row, eq.col))
    )

    if not equilibria:
        raise RuntimeError(
            "solver bug: no equilibrium found for a validated finite game"
        )

    for equilibrium in equilibria:
        if not is_nash(game, equilibrium.row, equilibrium.col):
            raise RuntimeError(
                "solver bug: claimed equilibrium fails "
                f"the independent best-response check: {equilibrium}"
            )

    return EquilibriumSet(
        equilibria=equilibria,
        degenerate=degenerate,
        method=method,
        support_enum_found=len(support_result.equilibria),
        singular_supports=tuple(support_result.singular_supports),
    )
