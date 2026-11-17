"""Cross-check our exact equilibria against nashpy.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

nashpy [4] is an independent implementation in floating point. Agreement
between two independent implementations is strong evidence that both are
right.

What we compare against, and why:
- PRIMARY: nashpy.vertex_enumeration. In a nondegenerate game every
  equilibrium is a vertex pair, so it lists all of them; in a degenerate game
  (V = 4) it lists exactly the extreme equilibria, which is what our solver
  returns there. ``agree`` is defined against this method.
- SECONDARY (reported, not required): nashpy.support_enumeration, the same
  algorithm as ours. It is unreliable on some games because of float
  rounding, found by the Hypothesis property test on 2026-11-13:
  1. It defaults to tol=1e-16, below rounding error. For
     A = [[0,0],[1,0],[0,1]], B = [[0,-1],[0,2],[0,-2]] it computes 1.1e-16
     instead of 0 for an unused action, fails its support check, and drops
     the true equilibrium ((0, 1/2, 1/2), (1/2, 1/2)).
  2. Even with a larger tol it rejects any solution with a negative entry
     (hard-coded ``prob >= 0``). For B = [[0,-2],[0,3],[0,-1]] the solve gives
     -5.55e-17 instead of 0, so the equilibrium ((0, 1/4, 3/4), (1/2, 1/2))
     is dropped.
  Our exact Fraction solver cannot make either mistake, which is the reason
  this project computes equilibria exactly. On every matrix the project
  actually uses, both nashpy methods agree with us (tests/test_equilibrium.py).
Matching uses a 1e-7 tolerance, since nashpy works in floats.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import nashpy
import numpy as np

from resource_war.encounter.matrix import BimatrixGame
from resource_war.equilibrium.solve import solve_game

TOLERANCE = 1e-7
NASHPY_SUPPORT_TOL = 1e-10

Profile = tuple[tuple[float, ...], tuple[float, ...]]


@dataclass(frozen=True)
class CrossCheck:
    agree: bool  # our equilibria == nashpy vertex_enumeration
    support_enum_agree: bool  # our equilibria == nashpy support_enumeration (informational)
    ours: tuple[Profile, ...]
    vertex: tuple[Profile, ...]
    support: tuple[Profile, ...]


def _close(a: Profile, b: Profile) -> bool:
    return all(abs(p - q) <= TOLERANCE for p, q in zip(a[0] + a[1], b[0] + b[1]))


def _dedupe(found) -> tuple[Profile, ...]:
    out: list[Profile] = []
    for x, y in found:
        pair = (
            tuple(float(v) for v in np.where(np.abs(x) < TOLERANCE, 0.0, x)),
            tuple(float(v) for v in np.where(np.abs(y) < TOLERANCE, 0.0, y)),
        )
        if not any(_close(pair, t) for t in out):
            out.append(pair)
    return tuple(out)


def _same(ours: tuple[Profile, ...], theirs: tuple[Profile, ...]) -> bool:
    return len(ours) == len(theirs) and all(any(_close(o, t) for t in theirs) for o in ours)


def crosscheck(game: BimatrixGame) -> CrossCheck:
    ours = tuple((tuple(map(float, e.row)), tuple(map(float, e.col))) for e in solve_game(game).equilibria)
    a, b = game.to_numpy()
    g = nashpy.Game(a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vertex = _dedupe(g.vertex_enumeration())
        support = _dedupe(g.support_enumeration(tol=NASHPY_SUPPORT_TOL))
    return CrossCheck(_same(ours, vertex), _same(ours, support), ours, vertex, support)
