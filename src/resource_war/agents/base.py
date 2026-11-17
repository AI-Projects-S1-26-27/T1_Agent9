"""The agent interface and the encounter information an agent receives.

Owner: Tushya Jain, Reviewer: Gursidak Singh Bassi

The proposal describes the agent loop as observe -> evaluate actions ->
choose -> receive payoff -> update. ``choose`` covers the middle steps and
``observe`` covers "receive payoff, update". Agents that do not learn simply
ignore ``observe``.

Agents only decide encounters (Attack / Defend / Expand) and carry a Battle
Mode search depth. Movement and collection are a shared, fixed policy in the
engine, so any difference in outcomes comes from these two choices alone
(the experimental control in DESIGN 3.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol, runtime_checkable

import numpy as np

from resource_war.encounter.matrix import BimatrixGame


@dataclass(frozen=True)
class EncounterState:
    """Everything an agent may use to decide one encounter.

    ``game`` is the encounter bimatrix from this agent's point of view: the
    agent is always the ROW player.
    """

    value: Fraction
    game: BimatrixGame
    turn: int
    cell: tuple[int, int]
    player: int
    my_depth: int
    opp_depth: int


@runtime_checkable
class Agent(Protocol):
    """What the engine needs from an agent."""

    name: str
    battle_depth: int

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        """Return an action index: 0 = Attack, 1 = Defend, 2 = Expand."""
        ...

    def observe(self, state: EncounterState, my_action: int, opp_action: int, payoff: Fraction) -> None:
        """Receive the outcome of an encounter this agent just played."""
        ...


def sample_action(probabilities: Sequence[Fraction], rng: np.random.Generator) -> int:
    """Draw an action index from exact probabilities using one uniform draw.

    One ``rng.random()`` call per decision, compared against cumulative sums,
    keeps the random stream identical across numpy versions' ``choice``
    implementations and makes replays byte-identical.
    """
    u = rng.random()
    cumulative = 0.0
    last_positive = max(i for i, p in enumerate(probabilities) if p > 0)
    for i, p in enumerate(probabilities):
        cumulative += float(p)
        if p > 0 and u < cumulative:
            return i
    return last_positive  # guards against float rounding when u is just below 1
