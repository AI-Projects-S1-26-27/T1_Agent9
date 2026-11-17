"""Known-answer fixtures, written BEFORE the analysis code (P2/P3).

Owner: Jainam Parekh Naresh, Reviewer: Atharv Singhal

Every expected value here was written down from the textbook definition of
each game (or, for the encounter game, the team's hand calculation in
DESIGN 2.3) before any solver existed. The solvers must reproduce them
exactly, in Fraction arithmetic.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest

from resource_war.config import load_payoffs
from resource_war.encounter.dominance import dominant_strategy, dominated_strategies, iterated_elimination
from resource_war.encounter.matrix import build_encounter_game, make_game
from resource_war.encounter.pareto import pareto_dominating_outcomes, pareto_optimal_outcomes
from resource_war.encounter.pure_nash import pure_nash_equilibria
from resource_war.equilibrium.solve import solve_game

PD = make_game([[3, 0], [5, 1]], [[3, 5], [0, 1]], ["C", "D"], ["C", "D"], "Prisoner's Dilemma")
MP = make_game([[1, -1], [-1, 1]], [[-1, 1], [1, -1]], ["H", "T"], ["H", "T"], "Matching Pennies")
BOS = make_game([[2, 0], [0, 1]], [[1, 0], [0, 2]], ["O", "F"], ["O", "F"], "Battle of the Sexes")
STAG = make_game([[4, 0], [3, 2]], [[4, 3], [0, 2]], ["S", "H"], ["S", "H"], "Stag Hunt")


def _profiles(game):
    return {(eq.row, eq.col) for eq in solve_game(game).equilibria}


def enc(v):
    return build_encounter_game(v, load_payoffs())


# ---------------------------------------------------------------- classics
def test_prisoners_dilemma() -> None:
    assert dominant_strategy(PD, 0) == 1 and dominant_strategy(PD, 1) == 1  # Defect
    assert pure_nash_equilibria(PD) == [(1, 1)]
    assert _profiles(PD) == {((F(0), F(1)), (F(0), F(1)))}
    assert set(pareto_optimal_outcomes(PD)) == {(0, 0), (0, 1), (1, 0)}
    assert pareto_dominating_outcomes(PD, (F(1), F(1))) == [(0, 0)]  # equilibrium is Pareto-inefficient


def test_matching_pennies() -> None:
    assert pure_nash_equilibria(MP) == []
    half = (F(1, 2), F(1, 2))
    result = solve_game(MP)
    assert _profiles(MP) == {(half, half)}
    assert (result.equilibria[0].row_payoff, result.equilibria[0].col_payoff) == (0, 0)
    assert not result.degenerate


def test_battle_of_the_sexes() -> None:
    assert pure_nash_equilibria(BOS) == [(0, 0), (1, 1)]
    assert _profiles(BOS) == {
        ((F(1), F(0)), (F(1), F(0))),
        ((F(0), F(1)), (F(0), F(1))),
        ((F(2, 3), F(1, 3)), (F(1, 3), F(2, 3))),
    }
    mixed = [e for e in solve_game(BOS).equilibria if len(e.row_support) == 2][0]
    assert (mixed.row_payoff, mixed.col_payoff) == (F(2, 3), F(2, 3))


def test_stag_hunt() -> None:
    assert pure_nash_equilibria(STAG) == [(0, 0), (1, 1)]
    assert _profiles(STAG) == {
        ((F(1), F(0)), (F(1), F(0))),
        ((F(0), F(1)), (F(0), F(1))),
        ((F(2, 3), F(1, 3)), (F(2, 3), F(1, 3))),
    }
    assert pareto_dominating_outcomes(STAG, (F(2), F(2))) == [(0, 0)]  # (Hare, Hare) is Pareto-dominated


# ---------------------------------------------------------------- our encounter game (DESIGN 2.3)
def test_encounter_v6_cyclic_unique_mixed() -> None:
    g = enc(6)
    assert dominated_strategies(g, 0, strict=True) == []
    assert dominated_strategies(g, 0, strict=False) == []
    assert pure_nash_equilibria(g) == []
    eq_set = solve_game(g)
    star = (F(13, 27), F(3, 27), F(11, 27))
    assert [(e.row, e.col) for e in eq_set.equilibria] == [(star, star)]
    assert eq_set.equilibria[0].row_payoff == F(51, 27) == eq_set.equilibria[0].col_payoff
    assert not eq_set.degenerate
    assert pareto_dominating_outcomes(g, (F(51, 27), F(51, 27))) == [(2, 2)]  # (E,E) = (3,3)
    assert set(pareto_optimal_outcomes(g)) == {(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)}


def test_encounter_v2_iterated_strict_dominance_gives_peaceful_expansion() -> None:
    g = enc(2)
    elim = iterated_elimination(g, strict=True)
    assert (elim.remaining_rows, elim.remaining_cols) == ((2,), (2,))
    first_round = [(s.player, s.dominated, s.dominator) for s in elim.steps[:2]]
    assert first_round == [(0, 0, 2), (1, 0, 2)]  # A eliminated by E for both players first
    assert [(e.row, e.col) for e in solve_game(g).equilibria] == [((F(0), F(0), F(1)), (F(0), F(0), F(1)))]
    assert (2, 2) in pareto_optimal_outcomes(g)  # here peaceful expansion is efficient


def test_encounter_v4_weak_dominance_and_degenerate_continuum() -> None:
    g = enc(4)
    assert dominated_strategies(g, 0, strict=True) == []
    weak = dominated_strategies(g, 0, strict=False)
    assert [(s.dominated, s.dominator) for s in weak] == [(0, 2)]  # E weakly dominates A
    assert pure_nash_equilibria(g) == [(2, 2)]
    eq_set = solve_game(g)
    assert eq_set.degenerate
    e = (F(0), F(0), F(1))
    ae = (F(1, 2), F(0), F(1, 2))
    assert {(x.row, x.col) for x in eq_set.equilibria} == {(e, e), (e, ae), (ae, e), (ae, ae)}


def test_encounter_v8_unique_mixed() -> None:
    star = (F(17, 36), F(1, 6), F(13, 36))
    eq_set = solve_game(enc(8))
    assert [(e.row, e.col) for e in eq_set.equilibria] == [(star, star)]
    assert eq_set.equilibria[0].row_payoff == F(25, 9)


@pytest.mark.parametrize("game", [PD, MP, BOS, STAG, enc(2), enc(6), enc(8)], ids=lambda g: g.name)
def test_nondegenerate_fixtures_are_flagged_nondegenerate(game) -> None:
    assert not solve_game(game).degenerate
