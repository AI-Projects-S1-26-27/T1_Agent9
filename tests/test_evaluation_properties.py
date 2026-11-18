"""Properties of the battle evaluation function.

Owner: Tushya Jain, Reviewer: Harshit Agrawal
"""

from __future__ import annotations

import numpy as np
import pytest

from resource_war.battle.evaluation import evaluate, terminal_utility
from resource_war.battle.state import apply_move, is_terminal, legal_moves, random_start


@pytest.fixture(scope="module")
def bcfg(configs):
    return configs.battle


def positions(cfg, n=60, seed=11):
    """Random starts plus a few plies of play, so mid-battle states are covered too."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        s = random_start(rng, cfg, i % 2)
        out.append(s)
        for _ in range(int(rng.integers(1, 6))):
            if is_terminal(s, cfg):
                break
            moves = legal_moves(s)
            s = apply_move(s, moves[int(rng.integers(len(moves)))], cfg)
            out.append(s)
    return out


def test_evaluate_is_zero_sum(bcfg) -> None:
    for s in positions(bcfg):
        assert evaluate(s, 0, bcfg) == -evaluate(s, 1, bcfg)


def test_terminal_utility_is_zero_sum(bcfg) -> None:
    terminals = [s for s in positions(bcfg, n=200) if is_terminal(s, bcfg)]
    assert terminals, "the sample should reach at least one finished battle"
    for s in terminals:
        assert terminal_utility(s, 0, bcfg) == -terminal_utility(s, 1, bcfg)


def test_a_heuristic_score_never_outranks_a_proven_result(bcfg) -> None:
    """The cutoff heuristic must stay below the smallest possible terminal win."""
    biggest_heuristic = max(abs(evaluate(s, side, bcfg)) for s in positions(bcfg) for side in (0, 1))
    assert biggest_heuristic < bcfg.win_utility
