"""Plain depth-limited minimax for Battle Mode.

Owner: Harshit Agrawal, Reviewer: Raunak Sarawgi

The side to move at the root is MAX; the opponent is MIN. Every node's
value is from the root side's point of view (DESIGN 4.6):
- terminal state -> exact terminal utility;
- depth exhausted -> heuristic evaluation;
- otherwise -> max (root side to move) or min (opponent to move) over ALL
  children. Plain minimax never skips a child; that is the baseline
  alpha-beta is measured against.

Root tie-breaking: children are visited in the fixed legal-move order and
the best move changes only on a STRICT improvement, so the first move that
reaches the maximum is chosen. Alpha-beta uses the identical rule (see
alphabeta.py for why the two then agree on the move).
"""

from __future__ import annotations

import math
from time import perf_counter

from resource_war.battle.evaluation import evaluate, terminal_utility
from resource_war.battle.instrument import SearchResult, SearchStats
from resource_war.battle.state import BattleState, apply_move, is_terminal, legal_moves
from resource_war.config import BattleConfig


def _minimax(state: BattleState, depth: int, root: int, cfg: BattleConfig, stats: SearchStats) -> int:
    stats.nodes += 1
    if is_terminal(state, cfg):
        stats.leaves += 1
        return terminal_utility(state, root, cfg)
    if depth == 0:
        stats.leaves += 1
        return evaluate(state, root, cfg)
    values = [_minimax(apply_move(state, m, cfg), depth - 1, root, cfg, stats) for m in legal_moves(state)]
    return max(values) if state.to_move == root else min(values)


def minimax_search(state: BattleState, depth: int, cfg: BattleConfig) -> SearchResult:
    """Choose a move for the side to move, searching ``depth`` plies."""
    if depth < 1:
        raise ValueError(f"search depth must be >= 1, got {depth}")
    if is_terminal(state, cfg):
        raise ValueError("no move to choose: the battle is already over")
    stats = SearchStats(nodes=1)
    start = perf_counter()
    root = state.to_move
    best_move, best_value = None, -math.inf
    for move in legal_moves(state):
        value = _minimax(apply_move(state, move, cfg), depth - 1, root, cfg, stats)
        if value > best_value:  # strict: keep the FIRST maximum
            best_move, best_value = move, value
    stats.seconds = perf_counter() - start
    return SearchResult("minimax", depth, best_move, int(best_value), stats)  # type: ignore[arg-type]
