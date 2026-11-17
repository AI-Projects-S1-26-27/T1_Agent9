"""Exact vertex enumeration of best-response polytopes (for degenerate games).

Owner: Gursidak Singh Bassi, Reviewer: Atharv Singhal

Why we need it: in a degenerate game (such as V = 4 in our encounter game)
the equilibria can form a continuum. Support enumeration with equal-size
supports then returns only part of the answer; at V = 4 it finds (E,E) but
misses three of the four extreme equilibria. Vertex enumeration finds every
EXTREME equilibrium, and every equilibrium is a convex combination of
extreme ones within its component.

Method (labelled polytopes, as in nashpy's vertex_enumeration [4]):
1. Shift both payoff matrices so all entries are positive (adding a constant
   does not change best responses).
2. P = {x >= 0 : B^T x <= 1} and Q = {y >= 0 : A y <= 1}.
3. Label a point of P with i when x_i = 0, and with m + j when column j is a
   best response (its constraint is tight). Label Q symmetrically.
4. A pair of vertices (x, y), not both zero, is an equilibrium exactly when
   every label 0 .. m+n-1 appears on x or on y ("completely labelled"):
   each action is either unused or a best response.
5. Normalising x and y to sum to 1 gives the mixed strategies.

Everything is exact (Fractions), so the vertices and labels are exact too.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from resource_war.encounter.matrix import BimatrixGame
from resource_war.equilibrium.linalg import solve_exact
from resource_war.equilibrium.support_enum import Equilibrium


def _vertices(constraints: list[tuple[list[Fraction], Fraction]], dim: int) -> dict[tuple[Fraction, ...], frozenset[int]]:
    """All vertices of {z : g . z <= h for every (g, h)}, with their tight-constraint labels."""
    found: dict[tuple[Fraction, ...], frozenset[int]] = {}
    for tight in combinations(range(len(constraints)), dim):
        sol = solve_exact([constraints[t][0] for t in tight], [constraints[t][1] for t in tight])
        if sol.status != "unique":
            continue
        z = sol.x
        values = [sum((g[k] * z[k] for k in range(dim)), Fraction(0)) for g, _ in constraints]
        if any(val > h for val, (_, h) in zip(values, constraints)):
            continue
        labels = frozenset(i for i, (val, (_, h)) in enumerate(zip(values, constraints)) if val == h)
        found[z] = labels
    return found


def _polytopes(game: BimatrixGame):
    """Vertices (with labels) of the two best-response polytopes P and Q."""
    m, n = game.shape
    a, b = game.row_payoffs, game.col_payoffs
    shift = 1 - min(min(min(r) for r in a), min(min(r) for r in b))
    a_pos = [[v + shift for v in r] for r in a]
    b_pos = [[v + shift for v in r] for r in b]
    one, zero = Fraction(1), Fraction(0)
    # P (row player's x): labels 0..m-1 for x_i >= 0, m..m+n-1 for columns' best-response constraints
    p_cons = [([-one if k == i else zero for k in range(m)], zero) for i in range(m)]
    p_cons += [([b_pos[i][j] for i in range(m)], one) for j in range(n)]
    # Q (column player's y): labels 0..m-1 for rows' best-response constraints, m..m+n-1 for y_j >= 0
    q_cons = [([a_pos[i][j] for j in range(n)], one) for i in range(m)]
    q_cons += [([-one if k == j else zero for k in range(n)], zero) for j in range(n)]
    return _vertices(p_cons, m), _vertices(q_cons, n)


def is_degenerate(game: BimatrixGame) -> bool:
    """Exact degeneracy test: some vertex of P or Q has more labels than its dimension.

    A label on a point of P means "this row is unused" or "this column is a
    best response". More than m labels at a point x means x (a mix over k
    rows) has more than k pure best responses: the textbook definition of a
    degenerate game. Any point with too many labels lies on a face whose
    vertices carry those labels too, so checking vertices is enough.
    TODO: citation needed (labelled-polytope characterisation of degeneracy).
    """
    m, n = game.shape
    p_vertices, q_vertices = _polytopes(game)
    return any(len(lab) > m for lab in p_vertices.values()) or any(len(lab) > n for lab in q_vertices.values())


def vertex_enumeration(game: BimatrixGame) -> list[Equilibrium]:
    m, n = game.shape
    a, b = game.row_payoffs, game.col_payoffs
    p_vertices, q_vertices = _polytopes(game)
    all_labels = frozenset(range(m + n))
    out = []
    for x, lx in p_vertices.items():
        if all(v == 0 for v in x):
            continue
        for y, ly in q_vertices.items():
            if all(v == 0 for v in y) or (lx | ly) != all_labels:
                continue
            sx, sy = sum(x), sum(y)
            xs = tuple(v / sx for v in x)
            ys = tuple(v / sy for v in y)
            u = sum((xs[i] * a[i][j] * ys[j] for i in range(m) for j in range(n)), Fraction(0))
            w = sum((xs[i] * b[i][j] * ys[j] for i in range(m) for j in range(n)), Fraction(0))
            out.append(Equilibrium(xs, ys, u, w))
    out.sort(key=lambda e: (e.row, e.col))
    return out
