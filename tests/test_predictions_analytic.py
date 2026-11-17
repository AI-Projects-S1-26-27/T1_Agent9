"""Pins every exact number quoted in docs/PREDICTIONS.md and DESIGN 5.

Owner: Jainam Parekh Naresh, Reviewer: Gursidak Singh Bassi

These are mathematical facts derived before any experiment ran: the H3
closed form for the coupled encounter game and the H4 per-encounter payoff
table. If a payoff or a solver changes, these tests show exactly which
pre-registered number no longer holds.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from resource_war.config import load_payoffs
from resource_war.encounter.matrix import build_encounter_game
from resource_war.equilibrium.selection import select_equilibrium
from resource_war.equilibrium.solve import solve_game

PAYOFFS = load_payoffs()


def f_mix(r: F) -> tuple[F, F, F]:
    """H3 closed form at V=6, c=2, d=1: a player's mix as a function of the OPPONENT's win prob."""
    den = 90 - 18 * r
    return ((39) / den, (24 * r - 3) / den, (54 - 42 * r) / den)


def general_mix(v: F, c: F, d: F, p: F) -> tuple[F, F, F]:
    """DESIGN 5: the opponent's mix as a function of MY win probability p (any V, c, d)."""
    den = v * (5 - p)
    return ((2 * v - c + 3 * d) / den, (v * (p + 1) - 3 * c + d * (2 * p - 1)) / den, 2 * (v * (1 - p) + 2 * c - d * (p + 1)) / den)


GRID = [F(k, 20) for k in range(3, 18)]  # 0.15 .. 0.85, all above 1/8


VALID_PAIRS = [(p, q) for p in GRID for q in (F(1, 5), F(2, 5), F(1, 2)) if p + q <= 1]


@pytest.mark.parametrize(("p", "q"), VALID_PAIRS)
def test_h3_closed_form_v6(p: F, q: F) -> None:
    (eq,) = solve_game(build_encounter_game(6, PAYOFFS, p, q)).equilibria
    assert eq.row == f_mix(q)  # my mix depends only on the opponent's win probability
    assert eq.col == f_mix(p)
    assert eq.row_payoff == (p + 8) / (5 - p)
    assert eq.col_payoff == (q + 8) / (5 - q)


@pytest.mark.parametrize("v", [5, 6, 7, 8, 10])
@pytest.mark.parametrize("p", [F(1, 4), F(1, 2), F(3, 4)])
def test_general_closed_form(v: int, p: F) -> None:
    """The closed form holds exactly where it is valid (all probabilities > 0, DESIGN 5).

    Outside that region (e.g. V = 5, p = 1/4, where the formula gives D a
    negative probability) the equilibrium is not fully mixed: the support shrinks.
    """
    q = F(1, 2) if p <= F(1, 2) else F(1, 4)
    (eq,) = solve_game(build_encounter_game(v, PAYOFFS, p, q)).equilibria
    formula = general_mix(F(v), F(2), F(1), p)
    # Valid only if BOTH players' closed-form mixes are strictly positive: each
    # player's mix is set by the other's win probability (V = 5, q = 1/4 fails).
    if all(x > 0 for x in formula) and all(x > 0 for x in general_mix(F(v), F(2), F(1), q)):
        assert eq.col == formula
        assert eq.row_payoff == (2 * v - 2 + 1 * (p - 2)) / (5 - p)
    else:
        assert 0 in eq.col, "outside the validity region the column's support must shrink"


def test_h3_worked_example_and_direction() -> None:
    deep, shallow = F(3, 5), F(2, 5)
    (eq,) = solve_game(build_encounter_game(6, PAYOFFS, deep, shallow)).equilibria
    assert eq.row == (F(65, 138), F(11, 138), F(31, 69)) and eq.row_payoff == F(43, 22)
    assert eq.col == (F(65, 132), F(19, 132), F(4, 11)) and eq.col_payoff == F(42, 23)
    assert eq.row[0] < eq.col[0] and eq.row[2] > eq.col[2]  # the stronger battler attacks LESS


FM = (F(2, 5), F(7, 20), F(1, 4))


def pure(i: int) -> tuple[F, F, F]:
    return tuple(F(int(k == i)) for k in range(3))  # type: ignore[return-value]


def heuristic(v: int) -> tuple[F, F, F]:
    return pure(0) if v > 5 else pure(2)


def eq_strategy(v: int) -> tuple[F, ...]:
    return select_equilibrium(solve_game(build_encounter_game(v, PAYOFFS)).equilibria).row


def best_response(v: int, opp) -> tuple[F, F, F]:
    g = build_encounter_game(v, PAYOFFS)
    values = [sum(g.row_payoffs[i][j] * opp[j] for j in range(3)) for i in range(3)]
    return pure(values.index(max(values)))  # first maximum = lexicographic tie-break


@pytest.mark.parametrize(
    ("v", "eq_fm", "eq_ph", "eu_fm", "eu_ph"),
    [
        (2, (F(19, 20), F(-1, 10)), (1, 1), (F(19, 20), F(-1, 10)), (1, 1)),
        (4, (F(19, 10), F(19, 20)), (2, 2), (F(19, 10), F(19, 20)), (2, 0)),
        (6, (F(893, 540), F(17, 9)), (F(28, 27), F(17, 9)), (F(57, 20), 2), (5, -2)),
        (8, (F(449, 180), F(25, 9)), (F(19, 9), F(25, 9)), (F(19, 5), F(61, 20)), (7, -2)),
    ],
)
def test_h4_table(v, eq_fm, eq_ph, eu_fm, eu_ph) -> None:
    g = build_encounter_game(v, PAYOFFS)
    assert g.expected_payoffs(eq_strategy(v), FM) == eq_fm
    assert g.expected_payoffs(eq_strategy(v), heuristic(v)) == eq_ph
    assert g.expected_payoffs(best_response(v, FM), FM) == eu_fm
    assert g.expected_payoffs(best_response(v, heuristic(v)), heuristic(v)) == eu_ph


def test_dirichlet_prior_already_implies_e_at_v6() -> None:
    uniform = (F(1, 3), F(1, 3), F(1, 3))
    assert best_response(6, uniform) == pure(2)


def test_equilibrium_agent_payoff_against_a_pure_expander_at_v6() -> None:
    """Quoted in the report (H4d): against pure E the Equilibrium agent earns 82/27, not 17/9."""
    g = build_encounter_game(6, PAYOFFS)
    assert g.expected_payoffs(eq_strategy(6), pure(2)) == (F(82, 27), F(17, 9))
