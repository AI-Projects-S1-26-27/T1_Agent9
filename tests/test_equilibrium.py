"""Equilibrium solvers: nashpy agreement, independent verification, degenerate games.

Owner: Jainam Parekh Naresh, Reviewer: Gursidak Singh Bassi
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import product

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from resource_war.config import load_payoffs
from resource_war.encounter.analysis import analyze
from resource_war.encounter.matrix import build_encounter_game, make_game
from resource_war.equilibrium.crosscheck import crosscheck
from resource_war.equilibrium.selection import select_equilibrium
from resource_war.equilibrium.solve import solve_game
from resource_war.equilibrium.support_enum import support_enumeration
from resource_war.equilibrium.verify import deviation_gains, is_nash
from test_known_answers import BOS, MP, PD, STAG

PAYOFFS = load_payoffs()


def sweep_games():
    """Every matrix the E5 sensitivity sweep will use (99 distinct games)."""
    games = {}
    for v0, kv, kc, kd in product((2, 4, 6, 8), (F(3, 4), 1, F(5, 4)), (F(3, 4), 1, F(5, 4)), (F(3, 4), 1, F(5, 4))):
        cfg = PAYOFFS.with_overrides(attack_cost=2 * kc, defend_cost=1 * kd)
        g = build_encounter_game(v0 * kv, cfg)
        games[g] = None
    return list(games)


def coupled_games():
    """Asymmetric matrices of the kind E4 builds (battle win probabilities with ties)."""
    out = []
    for p, q in [(F(1, 2), F(1, 2)), (F(3, 5), F(2, 5)), (F(7, 10), F(1, 5)), (F(2, 5), F(2, 5)), (F(9, 10), F(1, 20)), (F(1, 10), F(1, 10))]:
        for v in (2, 4, 6, 8):
            out.append(build_encounter_game(v, PAYOFFS, p, q))
    return out


ALL_GAMES = [PD, MP, BOS, STAG] + sweep_games() + coupled_games()


@pytest.mark.parametrize("game", ALL_GAMES, ids=lambda g: g.name)
def test_matches_nashpy_and_passes_independent_verifier(game) -> None:
    check = crosscheck(game)
    assert check.agree, f"ours={check.ours} nashpy vertex_enumeration={check.vertex}"
    if not solve_game(game).degenerate and check.support:
        # Nashpy support enumeration may return no equilibria due to floating-point errors.
        # When it returns results, they must agree with our exact solver.
        assert check.support_enum_agree, (
            f"ours={check.ours} nashpy support_enumeration={check.support}"
        )
    for e in solve_game(game).equilibria:
        assert is_nash(game, e.row, e.col)


def test_sweep_contains_the_degenerate_edges() -> None:
    degenerate = [g for g in sweep_games() if solve_game(g).degenerate]
    assert len(degenerate) == 9  # V = 2c at c in {1.5, 2, 2.5}, for each of 3 values of d


def test_verifier_rejects_non_equilibria_and_bad_input() -> None:
    g = build_encounter_game(6, PAYOFFS)
    a = (F(1), F(0), F(0))
    assert not is_nash(g, a, a)
    gains = deviation_gains(g, a, a)
    assert gains == (F(4), F(4))  # against A, D earns 5 instead of 1
    with pytest.raises(ValueError, match="sum to exactly 1"):
        is_nash(g, (F(1, 2), F(1, 2), F(1, 2)), a)
    with pytest.raises(ValueError, match="entries"):
        is_nash(g, (F(1), F(0)), a)


def test_support_enumeration_on_v4_reports_what_it_finds() -> None:
    se = support_enumeration(build_encounter_game(4, PAYOFFS))
    assert [(e.row, e.col) for e in se.equilibria] == [((0, 0, 1), (0, 0, 1))]
    assert se.degenerate and se.singular_supports == [((0, 2), (0, 2))]


def test_selection_rule() -> None:
    eqs = solve_game(build_encounter_game(4, PAYOFFS)).equilibria
    assert select_equilibrium(eqs).row == (0, 0, 1)
    assert select_equilibrium(solve_game(BOS).equilibria).row == (1, 0)  # row's favourite: Opera
    with pytest.raises(ValueError):
        select_equilibrium([])


def test_analysis_flags_bos_miscoordination_but_not_encounters() -> None:
    assert not analyze(BOS).played_is_equilibrium
    for v in (2, 4, 6, 8):
        assert analyze(build_encounter_game(v, PAYOFFS)).played_is_equilibrium


def test_custom_matrix_validation() -> None:
    from resource_war.config import ConfigError

    with pytest.raises(ConfigError, match="same shape"):
        make_game([[1, 2]], [[1], [2]])
    with pytest.raises(ConfigError, match="ragged"):
        make_game([[1, 2], [3]], [[1, 2], [3, 4]])
    with pytest.raises(ConfigError, match="finite"):
        make_game([[float("nan")]], [[1]])
    with pytest.raises(ConfigError, match="expected a number"):
        make_game([["x"]], [[1]])


small_payoff = st.integers(min_value=-30, max_value=30)


@st.composite
def random_games(draw):
    m = draw(st.integers(2, 3))
    n = draw(st.integers(2, 3))
    a = [[draw(small_payoff) for _ in range(n)] for _ in range(m)]
    b = [[draw(small_payoff) for _ in range(n)] for _ in range(m)]
    return make_game(a, b, name="random")


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(random_games())
def test_random_games_always_have_verified_equilibria(game) -> None:
    """Nash's theorem: every finite game has an equilibrium. Ours must find one and verify it.

    The nashpy comparison is limited to nondegenerate random games: on highly
    degenerate ones (e.g. all payoffs equal) nashpy's vertex enumeration
    itself crashes in its feasibility LP. Degenerate matrices that the project
    actually uses are cross-checked in the dedicated test above.
    """
    result = solve_game(game)
    assert result.equilibria
    assert all(is_nash(game, e.row, e.col) for e in result.equilibria)
    if not result.degenerate:
        assert crosscheck(game).agree


def test_exact_solver_finds_equilibrium_that_nashpy_default_tolerance_drops() -> None:
    """Regression for the nashpy tol=1e-16 finding documented in crosscheck.py."""
    import warnings

    import nashpy

    game = make_game([[0, 0], [1, 0], [0, 1]], [[0, -1], [0, 2], [0, -2]], name="tolerance case")
    (eq,) = solve_game(game).equilibria
    assert (eq.row, eq.col) == ((0, F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)))
    a, b = game.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert list(nashpy.Game(a, b).support_enumeration()) == []  # nashpy default drops it
    assert crosscheck(game).agree  # nashpy's vertex enumeration agrees with us


def test_nashpy_negative_rounding_case() -> None:
    """Second nashpy finding: a -5.55e-17 'probability' makes support_enumeration drop the equilibrium."""
    game = make_game([[0, 0], [1, 0], [0, 1]], [[0, -2], [0, 3], [0, -1]], name="negative rounding case")
    (eq,) = solve_game(game).equilibria
    assert (eq.row, eq.col) == ((0, F(1, 4), F(3, 4)), (F(1, 2), F(1, 2)))
    check = crosscheck(game)
    assert check.agree and not check.support_enum_agree


def test_degeneracy_with_unequal_supports_is_detected() -> None:
    """Hypothesis found this: an extra equilibrium with supports of sizes 2 and 1."""
    game = make_game([[0, 0, 0], [-1, -1, 0]], [[0, -2, -1], [-1, 1, 0]], name="unequal supports")
    result = solve_game(game)
    assert result.degenerate
    profiles = {(e.row, e.col) for e in result.equilibria}
    assert ((F(1, 2), F(1, 2)), (0, 0, 1)) in profiles
    assert ((1, 0), (1, 0, 0)) in profiles
    assert crosscheck(game).agree
