"""Battle Mode state and the rules: move generation, move application, terminal test.

Owner: Harshit Agrawal, Reviewer: Raunak Sarawgi

The battle is a two-rank mini-board (DESIGN 4): each side has a few units,
each in the Front rank, the Back rank, or Withdrawn (left alive). Only
Front units can strike, and only at Front enemies.

Design choices that matter for search:
- States are immutable NamedTuples. Applying a move builds a new state, so
  a search can never corrupt a sibling's state, and states are hashable.
- ``legal_moves`` returns moves in ONE fixed order (units in index order;
  per unit STRIKE(target 0), STRIKE(target 1), DEFEND, RETREAT, ADVANCE).
  Minimax and alpha-beta both rely on this order, so their root tie-breaks
  agree and they return the same move.
- One action per ply deals at most one point of damage, so two units can
  never die on the same ply and both sides can never be eliminated at once.
  A property test checks this rather than special-casing it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from resource_war.config import BattleConfig

FRONT, BACK, WITHDRAWN = 0, 1, 2
POSITION_NAMES = ("Front", "Back", "Withdrawn")
STRIKE, DEFEND, RETREAT, ADVANCE = "STRIKE", "DEFEND", "RETREAT", "ADVANCE"


class BUnit(NamedTuple):
    hp: int
    pos: int
    defending: bool = False

    @property
    def in_battle(self) -> bool:
        return self.hp > 0 and self.pos != WITHDRAWN


class BattleState(NamedTuple):
    sides: tuple[tuple[BUnit, ...], tuple[BUnit, ...]]
    to_move: int
    ply: int

    def describe(self) -> str:
        def side(s: int) -> str:
            return " ".join(
                f"[{'dead' if u.hp == 0 else POSITION_NAMES[u.pos][0]}{u.hp}{'*' if u.defending else ''}]"
                for u in self.sides[s]
            )

        return f"ply {self.ply}, side {self.to_move} to move | side0 {side(0)} | side1 {side(1)}"


class Move(NamedTuple):
    unit: int
    action: str
    target: int | None = None

    def describe(self) -> str:
        return f"u{self.unit} {self.action}" + (f"->e{self.target}" if self.target is not None else "")


def random_start(rng: np.random.Generator, cfg: BattleConfig, first_mover: int) -> BattleState:
    """Seeded start: each unit's HP is uniform in hp_range and its rank uniform in start_positions."""
    lo, hi = cfg.hp_range
    sides = []
    for _ in range(2):
        units = []
        for _ in range(cfg.units_per_side):
            hp = int(rng.integers(lo, hi + 1))
            pos = FRONT if cfg.start_positions[int(rng.integers(len(cfg.start_positions)))] == "Front" else BACK
            units.append(BUnit(hp, pos))
        sides.append(tuple(units))
    return BattleState((sides[0], sides[1]), first_mover, 0)


def hp_in_battle(state: BattleState, side: int) -> int:
    return sum(u.hp for u in state.sides[side] if u.in_battle)


def front_count(state: BattleState, side: int) -> int:
    return sum(1 for u in state.sides[side] if u.in_battle and u.pos == FRONT)


def is_terminal(state: BattleState, cfg: BattleConfig) -> bool:
    return (
        not any(u.in_battle for u in state.sides[0])
        or not any(u.in_battle for u in state.sides[1])
        or state.ply >= cfg.ply_cap
    )


def winner(state: BattleState, cfg: BattleConfig) -> int | None:
    """Winner of a terminal state: 0, 1, or None for an exact tie at the ply cap.

    A side with no units left in the battle (all dead or withdrawn) loses.
    Otherwise, at the cap, higher total HP among units still fighting wins.
    """
    if not is_terminal(state, cfg):
        raise ValueError("winner() called on a non-terminal battle state")
    alive = [any(u.in_battle for u in state.sides[s]) for s in (0, 1)]
    if alive[0] != alive[1]:
        return 0 if alive[0] else 1
    hp0, hp1 = hp_in_battle(state, 0), hp_in_battle(state, 1)
    if hp0 == hp1:
        return None
    return 0 if hp0 > hp1 else 1


def legal_moves(state: BattleState) -> list[Move]:
    me, enemy = state.sides[state.to_move], state.sides[1 - state.to_move]
    moves: list[Move] = []
    for i, unit in enumerate(me):
        if not unit.in_battle:
            continue
        if unit.pos == FRONT:
            for t, target in enumerate(enemy):
                if target.in_battle and target.pos == FRONT:
                    moves.append(Move(i, STRIKE, t))
        moves.append(Move(i, DEFEND))
        moves.append(Move(i, RETREAT))
        if unit.pos == BACK:
            moves.append(Move(i, ADVANCE))
    return moves


def apply_move(state: BattleState, move: Move, cfg: BattleConfig) -> BattleState:
    """Return the state after ``move``. The mover's guards from its previous ply drop first."""
    side = state.to_move
    me = [u._replace(defending=False) for u in state.sides[side]]
    enemy = list(state.sides[1 - side])
    unit = me[move.unit]
    if not unit.in_battle:
        raise ValueError(f"illegal battle move {move.describe()}: unit {move.unit} is not in the battle")
    if move.action == STRIKE:
        target = enemy[move.target] if move.target is not None and 0 <= move.target < len(enemy) else None
        if unit.pos != FRONT or target is None or not target.in_battle or target.pos != FRONT:
            raise ValueError(f"illegal battle move {move.describe()}: strikes go from Front to a Front enemy")
        damage = 0 if target.defending else cfg.strike_damage
        enemy[move.target] = target._replace(hp=max(0, target.hp - damage))
    elif move.action == DEFEND:
        me[move.unit] = unit._replace(defending=True)
    elif move.action == RETREAT:
        me[move.unit] = unit._replace(pos=BACK if unit.pos == FRONT else WITHDRAWN)
    elif move.action == ADVANCE:
        if unit.pos != BACK:
            raise ValueError(f"illegal battle move {move.describe()}: only Back units can advance")
        me[move.unit] = unit._replace(pos=FRONT)
    else:
        raise ValueError(f"unknown battle action {move.action!r}")
    sides = (tuple(me), tuple(enemy)) if side == 0 else (tuple(enemy), tuple(me))
    return BattleState(sides, 1 - side, state.ply + 1)
