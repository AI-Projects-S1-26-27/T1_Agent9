"""Strict, weak and iterated dominance between pure strategies, plus a mixed-dominance check.

Owner: Atharv Singhal, Reviewer: Gursidak Singh Bassi

Definitions (for the row player; the column player is symmetric):
- s strictly dominates t if s pays MORE than t against every remaining
  column strategy.
- s weakly dominates t if s pays AT LEAST as much against every remaining
  column strategy and strictly more against at least one.

Why iterated STRICT elimination is safe but WEAK is not:
- A strictly dominated strategy is never a best response, so it is never
  played in any Nash equilibrium. Deleting it cannot remove an equilibrium,
  and the final result does not depend on the order of deletion.
- Deleting weakly dominated strategies can delete equilibria, and the result
  can depend on the order. At V = 4 in our encounter game, A is only weakly
  dominated, yet A is played in a whole continuum of equilibria. We
  therefore report weak elimination as one documented order (row player
  first, then column, all dominated strategies of a player at once per
  round), always with that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from resource_war.encounter.matrix import BimatrixGame


@dataclass(frozen=True)
class Domination:
    player: int  # 0 = row, 1 = column
    dominated: int
    dominator: int
    strict: bool


def _payoff(game: BimatrixGame, player: int, own: int, other: int) -> Fraction:
    return game.row_payoffs[own][other] if player == 0 else game.col_payoffs[other][own]


def dominated_strategies(
    game: BimatrixGame,
    player: int,
    strict: bool = True,
    rows: tuple[int, ...] | None = None,
    cols: tuple[int, ...] | None = None,
) -> list[Domination]:
    """All (dominated, first dominator) pairs for ``player`` within the given sub-game."""
    m, n = game.shape
    rows = tuple(range(m)) if rows is None else rows
    cols = tuple(range(n)) if cols is None else cols
    own, other = (rows, cols) if player == 0 else (cols, rows)
    found = []
    for t in own:
        for s in own:
            if s == t:
                continue
            diffs = [_payoff(game, player, s, o) - _payoff(game, player, t, o) for o in other]
            if strict and all(d > 0 for d in diffs):
                found.append(Domination(player, t, s, True))
                break
            if not strict and all(d >= 0 for d in diffs) and any(d > 0 for d in diffs):
                found.append(Domination(player, t, s, False))
                break
    return found


def dominant_strategy(game: BimatrixGame, player: int) -> int | None:
    """A strategy that strictly dominates every other strategy of ``player``, if one exists."""
    m, n = game.shape
    own, other = (range(m), range(n)) if player == 0 else (range(n), range(m))
    for s in own:
        if all(
            all(_payoff(game, player, s, o) > _payoff(game, player, t, o) for o in other)
            for t in own
            if t != s
        ):
            return s
    return None


@dataclass(frozen=True)
class EliminationResult:
    steps: tuple[Domination, ...]
    remaining_rows: tuple[int, ...]
    remaining_cols: tuple[int, ...]
    strict: bool

    @property
    def solved(self) -> bool:
        """True if elimination leaves a single strategy profile."""
        return len(self.remaining_rows) == 1 and len(self.remaining_cols) == 1


def iterated_elimination(game: BimatrixGame, strict: bool = True) -> EliminationResult:
    """Repeatedly delete dominated strategies: each round row first, then column."""
    m, n = game.shape
    rows, cols = tuple(range(m)), tuple(range(n))
    steps: list[Domination] = []
    while True:
        row_out = dominated_strategies(game, 0, strict, rows, cols)
        if len(row_out) == len(rows):  # never delete a player's last strategy
            row_out = row_out[:-1]
        rows = tuple(r for r in rows if r not in {d.dominated for d in row_out})
        col_out = dominated_strategies(game, 1, strict, rows, cols)
        if len(col_out) == len(cols):
            col_out = col_out[:-1]
        cols = tuple(c for c in cols if c not in {d.dominated for d in col_out})
        steps += row_out + col_out
        if not row_out and not col_out:
            return EliminationResult(tuple(steps), rows, cols, strict)


def dominated_by_mixed(game: BimatrixGame, player: int, tol: float = 1e-9) -> list[int]:
    """Pure strategies strictly dominated by some MIXTURE of the player's other strategies.

    A strategy can be dominated by a mixture even when no single pure strategy
    dominates it. For each strategy t we solve the linear program
        maximise eps  s.t.  sum_k sigma_k * u(k, o) >= u(t, o) + eps  for all o,
                            sigma >= 0, sum sigma = 1 (k != t)
    and report t when the optimum eps is positive. Floats are used here (an LP
    solver), so this result is reported separately from the exact checks above.
    """
    from scipy.optimize import linprog  # imported lazily: scipy.optimize takes ~0.7 s to import

    m, n = game.shape
    own, other = (list(range(m)), list(range(n))) if player == 0 else (list(range(n)), list(range(m)))
    out = []
    for t in own:
        ks = [k for k in own if k != t]
        if not ks:
            continue
        # variables: sigma_k for k in ks, then eps; minimise -eps
        c = np.zeros(len(ks) + 1)
        c[-1] = -1.0
        a_ub = []
        b_ub = []
        for o in other:
            a_ub.append([-float(_payoff(game, player, k, o)) for k in ks] + [1.0])
            b_ub.append(-float(_payoff(game, player, t, o)))
        a_eq = [[1.0] * len(ks) + [0.0]]
        bounds = [(0, None)] * len(ks) + [(None, None)]
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
        if res.status == 0 and -res.fun > tol:
            out.append(t)
    return out
