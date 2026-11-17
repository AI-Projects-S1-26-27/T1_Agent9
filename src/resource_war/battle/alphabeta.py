"""Minimax with alpha-beta pruning for Battle Mode.

Owner: Raunak Sarawgi, Reviewer: Harshit Agrawal

alpha = the best value MAX is already guaranteed on the current path;
beta  = the best value MIN is already guaranteed on the current path.
When alpha >= beta at a node, the remaining children cannot change the
decision above, so they are skipped (pruned) and counted in stats.pruned.
This is the fail-soft form: a node may return a value outside (alpha, beta),
which is then only a bound on its true value.

Why alpha-beta returns the SAME MOVE as minimax, not just the same value
(H1a; DESIGN 4.6). At the root:
1. alpha always equals the exact best value found so far, because every
   child that raised alpha was searched with an open window above alpha and
   therefore returned its exact value.
2. A later child whose true value is <= alpha may be pruned below or fail
   low, but then it returns a bound that is <= alpha. Since the root only
   updates on a STRICT ">", such a child can never displace the incumbent.
3. A child whose true value is > alpha is searched with window
   (alpha, +inf) and returns its exact value, so it is chosen exactly when
   minimax would choose it.
Therefore both pick the first child (in the shared fixed order) that
attains the maximum. This relies on both using the same legal_moves order
and the same strict-">" root rule, which the code guarantees and
tests/test_battle.py checks on 500+ positions at depths 1-5.
"""

from __future__ import annotations

import math
from time import perf_counter

from resource_war.battle.evaluation import evaluate, terminal_utility
from resource_war.battle.instrument import SearchResult, SearchStats
from resource_war.battle.state import BattleState, Move, apply_move, is_terminal, legal_moves
from resource_war.config import BattleConfig


def _alphabeta(state: BattleState, depth: int, alpha: float, beta: float, root: int, cfg: BattleConfig, stats: SearchStats) -> float:
    stats.nodes += 1
    if is_terminal(state, cfg):
        stats.leaves += 1
        return terminal_utility(state, root, cfg)
    if depth == 0:
        stats.leaves += 1
        return evaluate(state, root, cfg)
    moves = legal_moves(state)
    if state.to_move == root:  # MAX node
        value = -math.inf
        for i, move in enumerate(moves):
            value = max(value, _alphabeta(apply_move(state, move, cfg), depth - 1, alpha, beta, root, cfg, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                stats.pruned += len(moves) - i - 1
                break
        return value
    value = math.inf  # MIN node
    for i, move in enumerate(moves):
        value = min(value, _alphabeta(apply_move(state, move, cfg), depth - 1, alpha, beta, root, cfg, stats))
        beta = min(beta, value)
        if alpha >= beta:
            stats.pruned += len(moves) - i - 1
            break
    return value


def rank_root_moves(state: BattleState, depth: int, cfg: BattleConfig) -> list[tuple[Move, int]]:
    """Exact value of EVERY root move (from the mover's point of view), in legal-move order.

    Used by the web demo's "engine lines" panel, like a chess analysis board. Each
    child is searched with a full (-inf, +inf) window, so every value is exact rather
    than a cut-off bound. The best value equals alphabeta_search's value, and the
    first move that attains it is the move alphabeta_search chooses (tested).
    """
    if depth < 1:
        raise ValueError(f"search depth must be >= 1, got {depth}")
    if is_terminal(state, cfg):
        raise ValueError("no move to choose: the battle is already over")
    root = state.to_move
    stats = SearchStats()
    return [
        (move, int(_alphabeta(apply_move(state, move, cfg), depth - 1, -math.inf, math.inf, root, cfg, stats)))
        for move in legal_moves(state)
    ]


def alphabeta_search(state: BattleState, depth: int, cfg: BattleConfig) -> SearchResult:
    """Choose a move for the side to move, searching ``depth`` plies with pruning."""
    if depth < 1:
        raise ValueError(f"search depth must be >= 1, got {depth}")
    if is_terminal(state, cfg):
        raise ValueError("no move to choose: the battle is already over")
    stats = SearchStats(nodes=1)
    start = perf_counter()
    root = state.to_move
    best_move, best_value = None, -math.inf
    alpha = -math.inf
    for move in legal_moves(state):
        value = _alphabeta(apply_move(state, move, cfg), depth - 1, alpha, math.inf, root, cfg, stats)
        if value > best_value:  # strict: keep the FIRST maximum, exactly as minimax does
            best_move, best_value = move, value
        alpha = max(alpha, best_value)
    stats.seconds = perf_counter() - start
    return SearchResult("alphabeta", depth, best_move, int(best_value), stats)  # type: ignore[arg-type]
