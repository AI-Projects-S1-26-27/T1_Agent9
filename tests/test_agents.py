"""Agent library: each agent's policy, learning, and full-game determinism.

Owner: Jainam Parekh Naresh, Reviewer: Tushya Jain
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction as F

import numpy as np
import pytest

from resource_war.agents.base import Agent, EncounterState, sample_action
from resource_war.agents.library import AGENT_TYPES, ExpectedUtilityAgent, make_agent
from resource_war.config import ConfigError
from resource_war.encounter.matrix import ATTACK, DEFEND, EXPAND, build_encounter_game
from resource_war.engine.game import Game


def state_for(configs, v, p_row=None, p_col=None) -> EncounterState:
    game = build_encounter_game(v, configs.payoffs, p_row, p_col)
    return EncounterState(F(v), game, 1, (3, 4), 0, 3, 3)


def frequencies(agent, state, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    c = Counter(agent.choose(state, rng) for _ in range(n))
    return [c[a] / n for a in (ATTACK, DEFEND, EXPAND)]


@pytest.mark.parametrize("kind", AGENT_TYPES)
def test_factory_builds_protocol_conforming_agents(configs, kind) -> None:
    agent = make_agent(kind, configs.agents, 3)
    assert isinstance(agent, Agent) and agent.battle_depth == 3


def test_factory_rejects_bad_input(configs) -> None:
    with pytest.raises(ConfigError, match="unknown agent type"):
        make_agent("random_walk", configs.agents, 3)
    with pytest.raises(ConfigError, match="must be one of"):
        make_agent("equilibrium", configs.agents, 4)


def test_pure_heuristic_attacks_only_above_threshold(configs) -> None:
    agent = make_agent("pure_heuristic", configs.agents, 1)
    rng = np.random.default_rng(0)
    assert agent.choose(state_for(configs, 6), rng) == ATTACK
    assert agent.choose(state_for(configs, 5), rng) == EXPAND  # "above", strictly (D16)
    assert agent.choose(state_for(configs, 4), rng) == EXPAND


def test_fixed_mixed_frequencies(configs) -> None:
    freq = frequencies(make_agent("fixed_mixed", configs.agents, 1), state_for(configs, 6))
    assert freq == pytest.approx([0.40, 0.35, 0.25], abs=0.01)


def test_equilibrium_agent_strategies(configs) -> None:
    agent = make_agent("equilibrium", configs.agents, 3)
    assert agent.strategy(state_for(configs, 2)) == (0, 0, 1)
    assert agent.strategy(state_for(configs, 4)) == (0, 0, 1)  # selection rule picks (E,E)
    assert agent.strategy(state_for(configs, 6)) == (F(13, 27), F(3, 27), F(11, 27))
    freq = frequencies(agent, state_for(configs, 6))
    assert freq == pytest.approx([13 / 27, 3 / 27, 11 / 27], abs=0.01)
    # coupled game: my mix depends on the OPPONENT's win probability (H3)
    assert agent.strategy(state_for(configs, 6, F(3, 5), F(2, 5))) == (F(65, 138), F(11, 138), F(31, 69))


def test_expected_utility_learns_per_value(configs) -> None:
    agent = make_agent("expected_utility", configs.agents, 1)
    assert isinstance(agent, ExpectedUtilityAgent)
    s6, s2 = state_for(configs, 6), state_for(configs, 2)
    rng = np.random.default_rng(0)
    assert agent.choose(s6, rng) == EXPAND  # uniform prior: best response is E at V = 6
    for _ in range(10):
        agent.observe(s6, EXPAND, ATTACK, F(0))  # opponent keeps attacking at V = 6
    assert agent.choose(s6, rng) == DEFEND  # best response to Attack is Defend
    assert agent.belief(s2) == (F(1, 3), F(1, 3), F(1, 3))  # V = 2 model untouched
    assert agent.belief(s6) == (F(11, 13), F(1, 13), F(1, 13))


def test_expected_utility_tie_break_at_v4(configs) -> None:
    agent = make_agent("expected_utility", configs.agents, 1)
    s4 = state_for(configs, 4)
    for _ in range(200):
        agent.observe(s4, EXPAND, EXPAND, F(2))
    values = agent.expected_values(s4)
    assert values[0] < values[2]  # with a finite prior, E is still strictly better
    pure_e = ExpectedUtilityAgent((F(0), F(0), F(1)), True, 1)  # degenerate belief: opponent surely Expands
    assert pure_e.expected_values(s4)[0] == pure_e.expected_values(s4)[2]
    assert pure_e.choose(s4, np.random.default_rng(0)) == ATTACK  # tie -> first action (DESIGN 6)


@pytest.mark.parametrize(("a", "b"), [("equilibrium", "fixed_mixed"), ("expected_utility", "pure_heuristic"), ("depth_limited", "equilibrium")])
def test_full_games_replay_byte_identically(configs, a, b) -> None:
    def play():
        return Game(configs, (make_agent(a, configs.agents, 1), make_agent(b, configs.agents, 3)), 2024).run()

    first, second = play(), play()
    assert first.log_bytes() == second.log_bytes()
    assert first.encounters


def test_every_pair_of_agent_types_completes_a_game(configs) -> None:
    for a in AGENT_TYPES:
        for b in AGENT_TYPES:
            result = Game(configs, (make_agent(a, configs.agents, 1), make_agent(b, configs.agents, 1)), 5).run()
            assert result.turns == configs.game.turn_limit


@pytest.mark.parametrize(
    ("probabilities", "message"),
    [
        ((), "must not be empty"),
        ((F(0), F(0), F(0)), "sum to 1"),
        ((F(-1, 4), F(1, 2), F(3, 4)), "non-negative"),
        ((F(1, 2), F(1, 4), F(1, 8)), "sum to 1"),
        ((F(1, 2), F(1, 2), F(1, 2)), "sum to 1"),
    ],
)
def test_sample_action_rejects_malformed_distributions(probabilities, message) -> None:
    with pytest.raises(ValueError, match=message):
        sample_action(probabilities, np.random.default_rng(0))


def test_sample_action_validation_does_not_change_the_random_stream() -> None:
    """Valid input must draw exactly one uniform, so replays stay byte-identical."""
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    sample_action((F(2, 5), F(7, 20), F(1, 4)), a)
    b.random()
    assert a.random() == b.random()


def test_sample_action_never_returns_a_zero_probability_action() -> None:
    rng = np.random.default_rng(1)
    assert {sample_action((F(1, 2), F(0), F(1, 2)), rng) for _ in range(500)} <= {0, 2}


@pytest.mark.parametrize("bad", [-1, 3, 7])
def test_expected_utility_rejects_an_invalid_opponent_action(configs, bad) -> None:
    agent = make_agent("expected_utility", configs.agents, 1)
    with pytest.raises(ValueError, match="opp_action"):
        agent.observe(state_for(configs, 6), ATTACK, bad, F(0))
    assert agent.counts == {}  # nothing was recorded


def test_depth_limited_plays_the_equilibrium_policy_and_only_the_depth_differs(configs) -> None:
    eq = make_agent("equilibrium", configs.agents, 3)
    for depth in (1, 3, 5):
        dl = make_agent("depth_limited", configs.agents, depth)
        assert dl.battle_depth == depth and dl.name == f"depth_limited_{depth}"
        for v in (2, 4, 6, 8):
            assert dl.strategy(state_for(configs, v)) == eq.strategy(state_for(configs, v))


@pytest.mark.parametrize("v", [2, 4, 6, 8])
def test_expected_utility_defends_against_an_opponent_that_always_attacks(configs, v) -> None:
    agent = make_agent("expected_utility", configs.agents, 1)
    s = state_for(configs, v)
    for _ in range(30):
        agent.observe(s, EXPAND, ATTACK, F(0))
    assert agent.belief(s)[ATTACK] > F(9, 10)
    assert agent.choose(s, np.random.default_rng(0)) == DEFEND
