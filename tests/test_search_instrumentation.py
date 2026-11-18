"""Search instrumentation: the counters mean what the docs say they mean.

Owner: Tushya Jain, Reviewer: Raunak Sarawgi
"""

from __future__ import annotations

import numpy as np
import pytest

from resource_war.battle.alphabeta import alphabeta_search
from resource_war.battle.minimax import minimax_search
from resource_war.battle.state import random_start


@pytest.fixture(scope="module")
def starts(configs):
    rng = np.random.default_rng(3)
    return [random_start(rng, configs.battle, i % 2) for i in range(40)]


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_counters_are_consistent(configs, starts, depth) -> None:
    for s in starts:
        mm = minimax_search(s, depth, configs.battle)
        ab = alphabeta_search(s, depth, configs.battle)
        assert mm.stats.pruned == 0  # plain minimax never prunes
        for r in (mm, ab):
            assert 0 < r.stats.leaves <= r.stats.nodes
            assert r.stats.seconds >= 0
        assert ab.stats.nodes <= mm.stats.nodes  # pruning can only remove work
        assert (ab.value, ab.move) == (mm.value, mm.move)


def test_summary_leaves_out_wall_time(configs, starts) -> None:
    """Wall time differs between runs, so it must not appear in the replayable summary."""
    summary = alphabeta_search(starts[0], 3, configs.battle).summary()
    assert "seconds" not in summary
    assert set(summary) == {"algorithm", "depth", "move", "value", "nodes", "leaves", "pruned"}


def test_node_counts_do_not_depend_on_the_run(configs, starts) -> None:
    a = alphabeta_search(starts[1], 4, configs.battle).summary()
    b = alphabeta_search(starts[1], 4, configs.battle).summary()
    assert a == b
