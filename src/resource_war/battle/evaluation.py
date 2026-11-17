"""Evaluation function and terminal utility, shared by minimax and alpha-beta.

Owner: Harshit Agrawal, Reviewer: Raunak Sarawgi

Both are written from the point of view of the side doing the search (the
"root" side).

Terminal states get the exact result (DESIGN 4.5):
    win_utility * outcome + (my HP - their HP),  outcome in {+1, 0, -1}.
A proven win (1000 + ...) always outranks any heuristic score, because the
largest possible |evaluation| is 64 at the default config. On a tie the HP
difference is 0 by definition of a tie, so a tie scores exactly 0.

Non-terminal states at the depth cutoff get the heuristic:
    hp_weight * (my HP - their HP) + front_weight * (my Front units - their Front units)
- The HP term measures material, the actual win condition, so it dominates
  (weight 10).
- The Front term is positional: only Front units can strike, so a Front
  unit is potential damage on the next ply (weight 2: a tie-breaker between
  equal-material positions, never worth a hit point).
HP counts only units still in the battle, consistent with the win rule (a
withdrawn unit's HP does not help win the cell).
"""

from __future__ import annotations

from resource_war.battle.state import BattleState, front_count, hp_in_battle, winner
from resource_war.config import BattleConfig


def evaluate(state: BattleState, side: int, cfg: BattleConfig) -> int:
    other = 1 - side
    return cfg.hp_weight * (hp_in_battle(state, side) - hp_in_battle(state, other)) + cfg.front_weight * (
        front_count(state, side) - front_count(state, other)
    )


def terminal_utility(state: BattleState, side: int, cfg: BattleConfig) -> int:
    w = winner(state, cfg)
    outcome = 0 if w is None else (1 if w == side else -1)
    return cfg.win_utility * outcome + (hp_in_battle(state, side) - hp_in_battle(state, 1 - side))
