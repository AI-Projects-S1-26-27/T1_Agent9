"""Exact support enumeration for mixed Nash equilibria (Porter, Nudelman & Shoham [3]).

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Idea: guess which actions each player uses (the supports I and J). In a
mixed equilibrium every action in a player's support must earn the SAME
expected payoff (otherwise the player would drop the worse one). That is the
indifference principle, and it turns each guess into a linear system:
- the COLUMN player's mix y must make every row in I equally good for the row player;
- the ROW player's mix x must make every column in J equally good for the column player.
So each player's mix is pinned down by the OTHER player's payoffs.

We try every pair of supports with |I| = |J| = k (enough for nondegenerate
games), solve both systems exactly, and keep a solution when:
1. all probabilities on the support are positive; and
2. no action outside the support does better (the best-response condition).

Degeneracy: if an indifference system is singular (infinitely many
solutions), or a found equilibrium has more pure best responses than
actions in its support, the game is degenerate. Support enumeration with
equal-size supports can then miss equilibria, so we flag it and the caller
falls back to vertex enumeration (vertex_enum.py) instead of crashing or
silently under-reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations

from resource_war.encounter.matrix import BimatrixGame
from resource_war.equilibrium.linalg import solve_exact


@dataclass(frozen=True)
class Equilibrium:
    row: tuple[Fraction, ...]
    col: tuple[Fraction, ...]
    row_payoff: Fraction
    col_payoff: Fraction

    @property
    def row_support(self) -> tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.row) if p > 0)

    @property
    def col_support(self) -> tuple[int, ...]:
        return tuple(j for j, p in enumerate(self.col) if p > 0)

    @property
    def is_pure(self) -> bool:
        return len(self.row_support) == 1 and len(self.col_support) == 1


@dataclass
class SupportEnumResult:
    equilibria: list[Equilibrium]
    singular_supports: list[tuple[tuple[int, ...], tuple[int, ...]]] = field(default_factory=list)
    extra_best_responses: bool = False

    @property
    def degenerate(self) -> bool:
        return bool(self.singular_supports) or self.extra_best_responses


def _indifferent_mix(payoffs, own_support, other_support, n_other: int, transpose: bool):
    """Mix over ``other_support`` that makes every action in ``own_support`` equally good.

    Unknowns: the k probabilities plus the common payoff u. Equations: one
    indifference equation per own action, plus "probabilities sum to 1".
    """
    k = len(other_support)
    a, b = [], []
    for s in own_support:
        coeffs = [payoffs[o][s] if transpose else payoffs[s][o] for o in other_support]
        a.append(coeffs + [Fraction(-1)])
        b.append(Fraction(0))
    a.append([Fraction(1)] * k + [Fraction(0)])
    b.append(Fraction(1))
    sol = solve_exact(a, b)
    if sol.status != "unique":
        return sol.status, None, None
    mix = [Fraction(0)] * n_other
    for idx, o in enumerate(other_support):
        mix[o] = sol.x[idx]
    return "unique", tuple(mix), sol.x[k]


def support_enumeration(game: BimatrixGame) -> SupportEnumResult:
    m, n = game.shape
    a, b = game.row_payoffs, game.col_payoffs
    result = SupportEnumResult([])
    for k in range(1, min(m, n) + 1):
        for rows in combinations(range(m), k):
            for cols in combinations(range(n), k):
                y_status, y, u = _indifferent_mix(a, rows, cols, n, transpose=False)
                x_status, x, v = _indifferent_mix(b, cols, rows, m, transpose=True)
                if "infinite" in (y_status, x_status):
                    result.singular_supports.append((rows, cols))
                    continue
                if y is None or x is None:
                    continue
                if any(y[j] <= 0 for j in cols) or any(x[i] <= 0 for i in rows):
                    continue
                row_vals = [sum((a[i][j] * y[j] for j in range(n)), Fraction(0)) for i in range(m)]
                col_vals = [sum((x[i] * b[i][j] for i in range(m)), Fraction(0)) for j in range(n)]
                if max(row_vals) > u or max(col_vals) > v:
                    continue
                if row_vals.count(u) > k or col_vals.count(v) > k:
                    result.extra_best_responses = True
                result.equilibria.append(Equilibrium(x, y, u, v))
    return result
