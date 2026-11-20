"""The five agent types and a factory that builds them from config/agents.yaml.

Owner: Tushya Jain, Reviewer: Gursidak Singh Bassi

Each agent answers one question, "Attack, Defend or Expand?", and carries a
Battle Mode search depth. They embody different ideas of rationality:

- PureHeuristic: a fixed rule ("attack anything worth more than T"). No
  game theory, no learning; cheap and predictable.
- FixedMixed: always randomises with the same probabilities (0.40/0.35/0.25).
  Unpredictable, but ignores the cell value and the opponent.
- Equilibrium: plays its Nash equilibrium strategy for the current matrix.
  It is unexploitable (the opponent cannot gain by deviating), but it is
  also non-exploiting: it makes the opponent indifferent and does not take
  advantage of a weak opponent. That is the point of H4.
- ExpectedUtility: models the opponent from what it has seen (fictitious
  play) and best-responds. It exploits predictable opponents but can itself
  be exploited while its model is wrong.
- DepthLimited(d): the Equilibrium encounter policy with Battle Mode depth d,
  so comparing DepthLimited(1/3/5) isolates the effect of lookahead.

Every agent sees the encounter as the ROW player (EncounterState.game).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from resource_war.agents.base import Agent, EncounterState, sample_action
from resource_war.config import AgentsConfig, ConfigError, validate_depth
from resource_war.encounter.matrix import ATTACK, EXPAND
from resource_war.equilibrium.selection import select_equilibrium
from resource_war.equilibrium.solve import solve_game


class _Base:
    name = "agent"

    def __init__(self, depth: int) -> None:
        self.battle_depth = depth

    def observe(self, state: EncounterState, my_action: int, opp_action: int, payoff: Fraction) -> None:
        return None  # non-learning agents ignore outcomes

    def __repr__(self) -> str:
        return f"{self.name}(depth={self.battle_depth})"


class PureHeuristicAgent(_Base):
    name = "pure_heuristic"

    def __init__(self, threshold: Fraction, depth: int) -> None:
        super().__init__(depth)
        self.threshold = threshold

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        return ATTACK if state.value > self.threshold else EXPAND


class FixedMixedAgent(_Base):
    name = "fixed_mixed"

    def __init__(self, probabilities: tuple[Fraction, Fraction, Fraction], depth: int) -> None:
        super().__init__(depth)
        self.probabilities = probabilities

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        return sample_action(self.probabilities, rng)


class EquilibriumAgent(_Base):
    name = "equilibrium"

    def strategy(self, state: EncounterState) -> tuple[Fraction, ...]:
        """The mixed strategy this agent plays in ``state`` (selection rule: equilibrium/selection.py)."""
        return select_equilibrium(solve_game(state.game).equilibria).row

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        return sample_action(self.strategy(state), rng)


class DepthLimitedAgent(EquilibriumAgent):
    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.name = f"depth_limited_{depth}"


class ExpectedUtilityAgent(_Base):
    """Fictitious play: best response to the opponent's empirical action frequencies.

    The model starts from Dirichlet pseudo-counts (1, 1, 1), so before any
    data the opponent is assumed to play uniformly. Counts are kept
    separately for each cell value V, because an opponent's behaviour can
    depend on V (PureHeuristic attacks only when V > 5). A new agent is
    created for every game, so games stay independent.
    """

    name = "expected_utility"

    def __init__(self, prior: tuple[Fraction, Fraction, Fraction], per_value: bool, depth: int) -> None:
        super().__init__(depth)
        self.prior = prior
        self.per_value = per_value
        self.counts: dict[Fraction | None, list[Fraction]] = {}

    def _key(self, state: EncounterState) -> Fraction | None:
        return state.value if self.per_value else None

    def belief(self, state: EncounterState) -> tuple[Fraction, ...]:
        counts = self.counts.get(self._key(state), list(self.prior))
        total = sum(counts)
        return tuple(c / total for c in counts)

    def expected_values(self, state: EncounterState) -> list[Fraction]:
        q = self.belief(state)
        g = state.game
        return [sum((g.row_payoffs[i][j] * q[j] for j in range(3)), Fraction(0)) for i in range(3)]

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        values = self.expected_values(state)
        return values.index(max(values))  # first maximum = lexicographic tie-break (A, D, E)

    def observe(self, state: EncounterState, my_action: int, opp_action: int, payoff: Fraction) -> None:
        if opp_action not in (0, 1, 2):
            raise ValueError(f"opp_action must be 0, 1 or 2, got {opp_action!r}")
        key = self._key(state)
        if key not in self.counts:
            self.counts[key] = list(self.prior)
        self.counts[key][opp_action] += 1


AGENT_TYPES = ("pure_heuristic", "fixed_mixed", "equilibrium", "expected_utility", "depth_limited")


def make_agent(kind: str, cfg: AgentsConfig, depth: int, allowed_depths: tuple[int, ...] = (1, 3, 5)) -> Agent:
    """Build a fresh agent; raises ConfigError with a clear message for bad input."""
    depth = validate_depth(depth, allowed_depths, f"battle depth for {kind}")
    if kind == "pure_heuristic":
        return PureHeuristicAgent(cfg.heuristic_threshold, depth)
    if kind == "fixed_mixed":
        return FixedMixedAgent(cfg.fixed_mixed, depth)
    if kind == "equilibrium":
        return EquilibriumAgent(depth)
    if kind == "expected_utility":
        return ExpectedUtilityAgent(cfg.dirichlet_prior, cfg.counts_per_value, depth)
    if kind == "depth_limited":
        return DepthLimitedAgent(depth)
    raise ConfigError(f"unknown agent type {kind!r}; choose one of {list(AGENT_TYPES)}")
