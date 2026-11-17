"""Engine: fair map, determinism, scoring invariant, rule edge cases.

Owner: Jainam Parekh Naresh, Reviewer: Vaibhav Malik
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from conftest import ConstantAgent, UniformAgent, coin_battle, tie_battle
from resource_war.encounter.matrix import ATTACK, DEFEND, EXPAND
from resource_war.engine.board import candidate_cells, mirror, place_resources
from resource_war.engine.game import Game
from resource_war.engine.state import ActionKind, IllegalMoveError, UnitAction


def _game(configs, agents, seed=1, **kw):
    kw.setdefault("battle_resolver", coin_battle)
    return Game(configs, agents, seed, **kw)


@pytest.mark.parametrize("placement", ["point_symmetric", "band"])
def test_resources_are_point_symmetric_with_equal_values(configs, placement) -> None:
    cfg = replace(configs.game, placement=placement)
    for seed in range(50):
        res = place_resources(cfg, np.random.default_rng(seed))
        assert len(res) == cfg.resource_count
        for cell, value in res.items():
            assert res[mirror(cell, cfg.board_size)] == value
            assert cell not in cfg.bases
            if placement == "band":
                assert abs(cell[0] + cell[1] - 7) <= cfg.band_width


def test_candidate_cells_never_contain_a_mirror_pair(configs) -> None:
    cells = set(candidate_cells(configs.game))
    assert not any(mirror(c, 8) in cells for c in cells)


def test_same_seed_gives_byte_identical_log(configs) -> None:
    a = _game(configs, (UniformAgent(), UniformAgent()), seed=123).run()
    b = _game(configs, (UniformAgent(), UniformAgent()), seed=123).run()
    c = _game(configs, (UniformAgent(), UniformAgent()), seed=124).run()
    assert a.log_bytes() == b.log_bytes()
    assert a.log_bytes() != c.log_bytes()


def test_score_is_resource_points_plus_territory(configs) -> None:
    result = _game(configs, (UniformAgent(), UniformAgent()), seed=5).run()
    w = configs.game.territory_weight
    for p in (0, 1):
        assert result.scores[p] == result.resource_points[p] + w * result.territory[p]
        assert result.resource_points[p] == result.gross_collected[p] - result.costs[p]
    assert result.turns == configs.game.turn_limit


def test_encounter_payoffs_match_matrix_entries(configs) -> None:
    """The payoff-to-score invariant: every non-battle encounter pays its matrix entry."""
    seen_pairs = set()
    for seed in range(20):
        result = _game(configs, (UniformAgent(), UniformAgent()), seed=seed).run()
        for e in result.encounters:
            seen_pairs.add(e.actions)
            if e.actions == (ATTACK, ATTACK):
                v, c = e.value, configs.payoffs.attack_cost
                assert sorted(e.payoffs) in ([-c, v - c], [-c, -c])
            else:
                assert e.payoffs == e.expected
    assert len(seen_pairs) == 9, "every action pair should occur at least once over 20 random games"


def test_defend_vs_defend_keeps_the_cell_contested(configs) -> None:
    game = _game(configs, (ConstantAgent(DEFEND), ConstantAgent(DEFEND)), seed=3)
    result = game.run()
    by_cell: dict = {}
    for e in result.encounters:
        by_cell.setdefault(e.cell, []).append(e.turn)
        assert e.payoffs == (-1, -1)
    # nobody ever obtains the cell, so the same cell is fought over on consecutive turns
    assert any(len(turns) > 1 and turns[1] == turns[0] + 1 for turns in by_cell.values())


def test_attack_vs_attack_battle_outcomes(configs) -> None:
    c = configs.payoffs.attack_cost
    win = _game(configs, (ConstantAgent(ATTACK), ConstantAgent(ATTACK)), seed=2).run()
    assert win.encounters and all(e.battle and e.battle["stub"] for e in win.encounters)
    for e in win.encounters:
        assert sorted(e.payoffs) == [-c, e.value - c]
    tie = _game(configs, (ConstantAgent(ATTACK), ConstantAgent(ATTACK)), seed=2, battle_resolver=tie_battle).run()
    assert all(e.payoffs == (-c, -c) for e in tie.encounters)


def test_agents_observe_every_encounter(configs) -> None:
    a, b = ConstantAgent(EXPAND), ConstantAgent(ATTACK)
    result = _game(configs, (a, b), seed=4).run()
    assert len(a.observed) == len(b.observed) == len(result.encounters)
    assert all(obs[:2] == (EXPAND, ATTACK) for obs in a.observed)


def test_mirror_policy_keeps_positions_symmetric(configs) -> None:
    """With mirrored tie-breaking and Expand-only agents the whole game is a mirror image."""
    game = _game(configs, (ConstantAgent(EXPAND), ConstantAgent(EXPAND)), seed=9)
    k = configs.game.units_per_player
    while not game.finished:
        game.step()
        units = game.state.units
        for i in range(k):
            assert units[k + i].pos == mirror(units[i].pos, 8)
    assert game.scores()[0] == game.scores()[1]


def test_turns_without_contested_cells_are_quiet(configs) -> None:
    game = _game(configs, (UniformAgent(), UniformAgent()), seed=11)
    quiet = 0
    while not game.finished:
        events = game.step()
        if not any(e["event"] == "encounter" for e in events):
            quiet += 1
    assert quiet > 0


@pytest.mark.parametrize(
    ("uid", "action", "message"),
    [
        (0, UnitAction(ActionKind.ATTACK), "only legal inside an encounter"),
        (0, UnitAction(ActionKind.DEFEND), "only legal inside an encounter"),
        (0, UnitAction(ActionKind.MOVE, (-1, 0)), "off the board"),
        (0, UnitAction(ActionKind.MOVE, (2, 0)), "one step"),
        (0, UnitAction(ActionKind.COLLECT), "nothing to collect"),
        (0, UnitAction(ActionKind.EXPAND), "already owned"),
    ],
)
def test_illegal_moves_are_rejected(configs, uid, action, message) -> None:
    game = _game(configs, (ConstantAgent(EXPAND), ConstantAgent(EXPAND)))
    with pytest.raises(IllegalMoveError, match=message):
        game.step({uid: action})


def test_invalid_agent_action_and_finished_game(configs) -> None:
    game = _game(configs, (ConstantAgent(7), ConstantAgent(EXPAND)), seed=0)
    with pytest.raises(IllegalMoveError, match="invalid encounter action"):
        game.run()
    done = _game(configs, (ConstantAgent(EXPAND), ConstantAgent(EXPAND)))
    done.run()
    with pytest.raises(IllegalMoveError, match="game is over"):
        done.step()


def test_invalid_depth_is_rejected(configs) -> None:
    with pytest.raises(ValueError, match="battle depth"):
        Game(configs, (ConstantAgent(EXPAND, depth=2), ConstantAgent(EXPAND)), 0, battle_resolver=coin_battle)


def test_regrowth_brings_cells_back(configs) -> None:
    result = _game(configs, (ConstantAgent(EXPAND), ConstantAgent(EXPAND)), seed=6).run()
    assert any('"event":"regrow"' in line for line in result.log_lines)
    no_regrow = replace(configs, game=replace(configs.game, regrow_turns=None))
    quiet = _game(no_regrow, (ConstantAgent(EXPAND), ConstantAgent(EXPAND)), seed=6).run()
    assert not any('"event":"regrow"' in line for line in quiet.log_lines)
    assert all(isinstance(v, Fraction) for v in quiet.scores)
