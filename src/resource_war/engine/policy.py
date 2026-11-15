"""The shared, deterministic non-encounter policy used by EVERY agent.

Owner: Vaibhav Malik, Reviewer: Harshit Agrawal

Experimental control (DESIGN 3.2): outside encounters all agents move and
collect identically, so agents differ only in their encounter policy and
battle depth. Any outcome difference is then attributable to the decision
technique, not to better pathfinding.

Per player, units are processed in uid order:
1. a unit standing on an uncollected resource collects it (and claims it);
2. otherwise it steps toward its target resource; targets are assigned so
   that two units of the same player do not chase the same cell while
   another one is free;
3. once no resources remain, units claim unowned cells as territory.

Ties between equally good targets are broken by a fixed cell order. With
``player_relative`` tie-breaking, player 1 uses the order of the board
rotated by 180 degrees, so the two players' policies are exact mirror
images. With ``absolute`` tie-breaking both use the same row-major order,
which sends both players to the same cell when two candidates tie.
"""

from __future__ import annotations

from fractions import Fraction

from resource_war.config import GameConfig
from resource_war.engine.board import Coord, manhattan, mirror
from resource_war.engine.state import ActionKind, GameState, Unit, UnitAction


def _tie_key(cell: Coord, player: int, cfg: GameConfig) -> Coord:
    if cfg.tie_break == "player_relative" and player == 1:
        return mirror(cell, cfg.board_size)
    return cell


def _target_key(unit: Unit, cell: Coord, value: Fraction | None, cfg: GameConfig) -> tuple:
    dist = manhattan(unit.pos, cell)
    tie = _tie_key(cell, unit.player, cfg)
    if cfg.targeting == "value_per_distance" and value is not None:
        return (-value / (dist + 1), dist, tie)
    return (dist, tie)


def step_toward(pos: Coord, target: Coord) -> Coord:
    """One step that closes the row gap first, then the column gap."""
    r, c = pos
    if r != target[0]:
        return (r + (1 if target[0] > r else -1), c)
    if c != target[1]:
        return (r, c + (1 if target[1] > c else -1))
    return pos


def _choose(unit: Unit, options: dict[Coord, Fraction | None], claimed: set[Coord], cfg: GameConfig) -> Coord | None:
    free = {cell: v for cell, v in options.items() if cell not in claimed}
    pool = free or options
    if not pool:
        return None
    return min(pool, key=lambda cell: _target_key(unit, cell, pool[cell], cfg))


def plan_unit_actions(state: GameState, cfg: GameConfig) -> dict[int, UnitAction]:
    """Return the shared policy's action for every unit (keyed by uid)."""
    actions: dict[int, UnitAction] = {}
    for player in (0, 1):
        units = state.units_of(player)
        claimed: set[Coord] = set()
        # Units already standing on a resource collect it and claim it first.
        for unit in units:
            if unit.pos in state.resources:
                actions[unit.uid] = UnitAction(ActionKind.COLLECT)
                claimed.add(unit.pos)
        for unit in units:
            if unit.uid in actions:
                continue
            if state.resources:
                target = _choose(unit, dict(state.resources), claimed, cfg)
            else:
                if unit.pos not in state.owner:
                    actions[unit.uid] = UnitAction(ActionKind.EXPAND)
                    claimed.add(unit.pos)
                    continue
                unowned = {
                    (r, c): None
                    for r in range(state.size)
                    for c in range(state.size)
                    if (r, c) not in state.owner
                }
                target = _choose(unit, unowned, claimed, cfg)
            if target is None:
                actions[unit.uid] = UnitAction(ActionKind.PASS)
                continue
            claimed.add(target)
            actions[unit.uid] = UnitAction(ActionKind.MOVE, step_toward(unit.pos, target))
    return actions
