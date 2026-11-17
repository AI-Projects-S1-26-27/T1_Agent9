"""Compare exact equilibria with Nashpy's floating-point results.

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Vertex enumeration is the primary comparison. Support enumeration is
reported separately because it can miss equilibria in degenerate games
or reject candidates affected by floating-point rounding.

Agreement is numerical evidence, not a proof of correctness.
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
    agree: bool
    support_enum_agree: bool
    ours: tuple[Profile, ...]
    vertex: tuple[Profile, ...]
    support: tuple[Profile, ...]


def _close(a: Profile, b: Profile) -> bool:
    if len(a[0]) != len(b[0]) or len(a[1]) != len(b[1]):
        return False

    return all(
        abs(p - q) <= TOLERANCE
        for p, q in zip(a[0] + a[1], b[0] + b[1])
    )


def _dedupe(found) -> tuple[Profile, ...]:
    profiles: list[Profile] = []

    for x, y in found:
        profile = (
            tuple(
                float(value)
                for value in np.where(
                    np.abs(x) < TOLERANCE, 0.0, x
                )
            ),
            tuple(
                float(value)
                for value in np.where(
                    np.abs(y) < TOLERANCE, 0.0, y
                )
            ),
        )

        if not any(_close(profile, existing) for existing in profiles):
            profiles.append(profile)

    return tuple(profiles)


def _same(
    ours: tuple[Profile, ...],
    theirs: tuple[Profile, ...],
) -> bool:
    """Match every profile to a distinct profile within tolerance."""
    if len(ours) != len(theirs):
        return False

    matches: dict[int, int] = {}

    def assign(i: int, seen: set[int]) -> bool:
        for j, candidate in enumerate(theirs):
            if j in seen or not _close(ours[i], candidate):
                continue

            seen.add(j)

            # Reassign an earlier match when another pairing is possible.
            if j not in matches or assign(matches[j], seen):
                matches[j] = i
                return True

        return False

    return all(assign(i, set()) for i in range(len(ours)))


def crosscheck(game: BimatrixGame) -> CrossCheck:
    """Compare our profiles with Nashpy's vertex and support methods.

    Enumeration errors propagate to the caller. A failed library call
    must not be reported as successful agreement.
    """
    ours = tuple(
        (
            tuple(map(float, equilibrium.row)),
            tuple(map(float, equilibrium.col)),
        )
        for equilibrium in solve_game(game).equilibria
    )

    a, b = game.to_numpy()
    reference = nashpy.Game(a, b)

    # Keep the existing quiet behavior for analysis cards. Exceptions
    # still propagate; warning suppression does not imply success.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vertex = _dedupe(reference.vertex_enumeration())
        support = _dedupe(
            reference.support_enumeration(tol=NASHPY_SUPPORT_TOL)
        )

    return CrossCheck(
        agree=_same(ours, vertex),
        support_enum_agree=_same(ours, support),
        ours=ours,
        vertex=vertex,
        support=support,
    )
