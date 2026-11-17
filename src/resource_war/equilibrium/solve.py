"""One entry point for "all equilibria of this game", handling degenerate games.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Policy:
1. Run exact support enumeration (the method named in the proposal).
2. Decide degeneracy exactly: support enumeration's own signals (singular
   systems, extra best responses at a found equilibrium) OR the polytope
   label test in vertex_enum.is_degenerate. The second test is needed:
   the Hypothesis property test found a 2x3 game whose extra equilibrium
   has UNEQUAL supports (sizes 2 and 1), invisible to equal-size support
   enumeration and to its own signals.
3. If the game is degenerate, also run exact vertex enumeration and return
   every extreme equilibrium; the card explains that in a degenerate game
   equilibria can form a continuum whose corners are the listed points.
4. Every returned equilibrium is re-checked by the independent verifier
   (verify.py); a failure is a bug and raises immediately.

Results are cached per game: BimatrixGame is immutable and hashable, and the
Equilibrium agent asks for the same few matrices thousands of times in a
tournament.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from resource_war.encounter.matrix import BimatrixGame
from resource_war.equilibrium.support_enum import Equilibrium, support_enumeration
from resource_war.equilibrium.verify import is_nash
from resource_war.equilibrium.vertex_enum import is_degenerate, vertex_enumeration


@dataclass(frozen=True)
class EquilibriumSet:
    equilibria: tuple[Equilibrium, ...]
    degenerate: bool
    method: str
    support_enum_found: int
    singular_supports: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = field(default=())

    def notes(self) -> list[str]:
        if not self.degenerate:
            return ["Nondegenerate game: support enumeration finds every equilibrium."]
        notes = [
            "Degenerate game: some strategy has more pure best responses than its support size, "
            "so equilibria can form a continuum.",
            f"Support enumeration alone found {self.support_enum_found}; vertex enumeration lists "
            f"all {len(self.equilibria)} extreme equilibria (corners of the equilibrium set).",
        ]
        if self.singular_supports:
            notes.append(f"Singular indifference systems for {len(self.singular_supports)} support pair(s).")
        return notes


@lru_cache(maxsize=4096)
def solve_game(game: BimatrixGame) -> EquilibriumSet:
    se = support_enumeration(game)
    degenerate = se.degenerate or is_degenerate(game)
    if degenerate:
        found = {(e.row, e.col): e for e in se.equilibria}
        for e in vertex_enumeration(game):
            found.setdefault((e.row, e.col), e)
        equilibria = tuple(sorted(found.values(), key=lambda e: (e.row, e.col)))
        method = "vertex_enumeration"
    else:
        equilibria = tuple(se.equilibria)
        method = "support_enumeration"
    for e in equilibria:
        if not is_nash(game, e.row, e.col):
            raise RuntimeError(f"solver bug: claimed equilibrium {e} fails the best-response check")
    return EquilibriumSet(equilibria, degenerate, method, len(se.equilibria), tuple(se.singular_supports))
