"""Game state, unit actions and legality checks.

Owner: Vaibhav Malik, Reviewer: Harshit Agrawal

Illegal actions raise ``IllegalMoveError`` with a message meant for the user
(the demo prints it as-is). The shared policy never produces an illegal
action; these checks exist for hand-entered moves and as a guard on the
policy itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction

import numpy as np

from resource_war.config import GameConfig
from resource_war.engine.board import Coord, in_bounds, manhattan, place_resources


class IllegalMoveError(ValueError):
    """An action that the rules do not allow in the current state."""


class ActionKind(Enum):
    MOVE = "Move"
    COLLECT = "Collect"
    EXPAND = "Expand"
    ATTACK = "Attack"
    DEFEND = "Defend"
    PASS = "Pass"


@dataclass(frozen=True)
class UnitAction:
    kind: ActionKind
    to: Coord | None = None  # destination for MOVE

    def describe(self) -> str:
        return f"{self.kind.value}->{self.to}" if self.kind is ActionKind.MOVE else self.kind.value


@dataclass
class Unit:
    uid: int
    player: int
    pos: Coord


@dataclass
class GameState:
    """Mutable state of one game. Score = resource_points + w * cells owned."""

    size: int
    turn: int
    units: list[Unit]
    resources: dict[Coord, Fraction]  # uncollected resource cells -> value
    owner: dict[Coord, int]  # owned (territory) cells -> player
    depleted: dict[Coord, tuple[Fraction, int]] = field(default_factory=dict)  # cell -> (value, turn)
    resource_points: list[Fraction] = field(default_factory=lambda: [Fraction(0), Fraction(0)])
    gross_collected: list[Fraction] = field(default_factory=lambda: [Fraction(0), Fraction(0)])
    costs: list[Fraction] = field(default_factory=lambda: [Fraction(0), Fraction(0)])

    def territory(self, player: int) -> int:
        return sum(1 for p in self.owner.values() if p == player)

    def score(self, player: int, territory_weight: Fraction) -> Fraction:
        return self.resource_points[player] + territory_weight * self.territory(player)

    def units_of(self, player: int) -> list[Unit]:
        return [u for u in self.units if u.player == player]

    def occupants(self, cell: Coord) -> set[int]:
        """Players with at least one unit on ``cell``."""
        return {u.player for u in self.units if u.pos == cell}


def initial_state(cfg: GameConfig, rng: np.random.Generator) -> GameState:
    """Units start on their base; each base is owned territory from turn 0."""
    resources = place_resources(cfg, rng)
    units = [
        Unit(uid=p * cfg.units_per_player + k, player=p, pos=cfg.bases[p])
        for p in range(2)
        for k in range(cfg.units_per_player)
    ]
    owner = {cfg.bases[0]: 0, cfg.bases[1]: 1}
    return GameState(cfg.board_size, 0, units, resources, owner)


def check_unit_action(state: GameState, unit: Unit, action: UnitAction) -> None:
    """Raise IllegalMoveError if ``unit`` may not take ``action`` outside an encounter."""
    kind = action.kind
    if kind in (ActionKind.ATTACK, ActionKind.DEFEND):
        raise IllegalMoveError(
            f"unit {unit.uid}: {kind.value} is only legal inside an encounter (a resource cell "
            f"occupied by both players)"
        )
    if kind is ActionKind.MOVE:
        if action.to is None or not in_bounds(action.to, state.size):
            raise IllegalMoveError(f"unit {unit.uid}: move target {action.to} is off the board")
        if manhattan(unit.pos, action.to) != 1:
            raise IllegalMoveError(f"unit {unit.uid}: can only move one step up/down/left/right, not {unit.pos}->{action.to}")
    elif kind is ActionKind.COLLECT:
        if unit.pos not in state.resources:
            raise IllegalMoveError(f"unit {unit.uid}: nothing to collect at {unit.pos}")
    elif kind is ActionKind.EXPAND:
        if unit.pos in state.resources:
            raise IllegalMoveError(f"unit {unit.uid}: cannot expand onto an uncollected resource at {unit.pos}; collect it first")
        if unit.pos in state.owner:
            raise IllegalMoveError(f"unit {unit.uid}: cell {unit.pos} is already owned")
