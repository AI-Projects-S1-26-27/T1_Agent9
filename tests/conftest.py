"""Shared fixtures and stub agents for the test suite.

Owner: Jainam Parekh Naresh, Reviewer: Vaibhav Malik

Stub agents are deliberately trivial (constant or uniformly random actions)
so engine tests do not depend on the real agent library.
"""

from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from resource_war.agents.base import EncounterState, sample_action
from resource_war.config import DEFAULT_CONFIG_DIR, Configs, load_all


class ConstantAgent:
    def __init__(self, action: int, depth: int = 1, name: str = "constant") -> None:
        self.action = action
        self.battle_depth = depth
        self.name = name
        self.observed: list[tuple[int, int, Fraction]] = []

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        return self.action

    def observe(self, state: EncounterState, my_action: int, opp_action: int, payoff: Fraction) -> None:
        self.observed.append((my_action, opp_action, payoff))


class UniformAgent(ConstantAgent):
    def __init__(self, depth: int = 1, name: str = "uniform") -> None:
        super().__init__(0, depth, name)

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        third = Fraction(1, 3)
        return sample_action((third, third, third), rng)


def coin_battle(rng: np.random.Generator, depths: tuple[int, int], compare_minimax: bool):
    """Battle stub for engine tests: a seeded coin decides the winner."""
    winner = int(rng.integers(2))
    return winner, {"stub": True, "winner": winner}


def tie_battle(rng: np.random.Generator, depths: tuple[int, int], compare_minimax: bool):
    return None, {"stub": True, "winner": None}


@pytest.fixture(scope="session")
def configs() -> Configs:
    return load_all()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A writable copy of the real config directory for corruption tests."""
    target = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, target)
    return target
