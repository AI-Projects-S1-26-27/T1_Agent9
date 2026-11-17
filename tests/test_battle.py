"""Battle Mode: rules, minimax correctness, alpha-beta equivalence, determinism.

Owner: Jainam Parekh Naresh, Reviewer: Harshit Agrawal
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import ConstantAgent
from resource_war.battle.alphabeta import alphabeta_search
from resource_war.battle.benchmark import BENCHMARK_FILE, generate_positions, load_positions, save_positions, state_from_json, state_to_json
from resource_war.battle.evaluation import evaluate, terminal_utility
from resource_war.battle.minimax import minimax_search
from resource_war.battle.play import play_battle, resolve_battle
from resource_war.battle.state import (
    ADVANCE,
    BACK,
    DEFEND,
    FRONT,
    RETREAT,
    STRIKE,
    WITHDRAWN,
    BattleState,
    BUnit,
    Move,
    apply_move,
    is_terminal,
    legal_moves,
    random_start,
    winner,
)
from resource_war.encounter.matrix import ATTACK
from resource_war.engine.game import Game


@pytest.fixture(scope="module")
def bcfg(configs):
    return configs.battle


def S(side0, side1, to_move=0, ply=0) -> BattleState:
    return BattleState((tuple(BUnit(*u) for u in side0), tuple(BUnit(*u) for u in side1)), to_move, ply)


# ------------------------------------------------------------------ rules
def test_move_order_is_fixed(bcfg) -> None:
    s = S([(2, FRONT), (3, BACK)], [(2, FRONT), (2, FRONT)])
    assert legal_moves(s) == [
        Move(0, STRIKE, 0), Move(0, STRIKE, 1), Move(0, DEFEND), Move(0, RETREAT),
        Move(1, DEFEND), Move(1, RETREAT), Move(1, ADVANCE),
    ]
    assert len(legal_moves(S([(2, FRONT), (2, FRONT)], [(2, FRONT), (2, FRONT)]))) == 8  # branching <= 8


def test_defend_blocks_one_strike_and_drops_on_own_next_ply(bcfg) -> None:
    s = S([(2, FRONT), (2, FRONT)], [(2, FRONT), (2, FRONT)])
    s = apply_move(s, Move(0, DEFEND), bcfg)  # side 0 guards unit 0
    s = apply_move(s, Move(0, STRIKE, 0), bcfg)  # side 1 strikes it: blocked
    assert s.sides[0][0].hp == 2 and s.sides[0][0].defending
    s = apply_move(s, Move(1, DEFEND), bcfg)  # side 0's next ply: old guard drops
    assert not s.sides[0][0].defending and s.sides[0][1].defending
    s = apply_move(s, Move(0, STRIKE, 0), bcfg)
    assert s.sides[0][0].hp == 1


def test_retreat_withdraw_and_withdrawn_hp_does_not_count(bcfg) -> None:
    s = S([(3, BACK), (1, FRONT)], [(2, FRONT), (2, FRONT)])
    s = apply_move(s, Move(0, RETREAT), bcfg)
    assert s.sides[0][0].pos == WITHDRAWN and not s.sides[0][0].in_battle
    assert evaluate(s, 0, bcfg) == 10 * (1 - 4) + 2 * (1 - 2)


def test_side_with_no_units_left_loses_even_with_more_hp(bcfg) -> None:
    s = S([(3, WITHDRAWN), (3, WITHDRAWN)], [(1, FRONT), (0, FRONT)], to_move=1, ply=4)
    assert is_terminal(s, bcfg) and winner(s, bcfg) == 1
    assert terminal_utility(s, 1, bcfg) == bcfg.win_utility + 1


def test_tie_at_ply_cap(bcfg) -> None:
    s = S([(2, FRONT), (1, BACK)], [(3, BACK), (0, FRONT)], ply=bcfg.ply_cap)
    assert winner(s, bcfg) is None and terminal_utility(s, 0, bcfg) == 0


def test_illegal_battle_moves_raise(bcfg) -> None:
    s = S([(2, BACK), (2, FRONT)], [(2, BACK), (2, FRONT)])
    with pytest.raises(ValueError, match="Front"):
        apply_move(s, Move(0, STRIKE, 1), bcfg)  # Back unit cannot strike
    with pytest.raises(ValueError, match="Front"):
        apply_move(s, Move(1, STRIKE, 0), bcfg)  # cannot strike a Back enemy
    with pytest.raises(ValueError, match="advance"):
        apply_move(s, Move(1, ADVANCE), bcfg)


def test_search_rejects_terminal_states_and_bad_depth(bcfg) -> None:
    done = S([(1, FRONT)], [(0, FRONT)], ply=3)
    with pytest.raises(ValueError, match="already over"):
        minimax_search(done, 3, bcfg)
    with pytest.raises(ValueError, match="depth"):
        alphabeta_search(S([(1, FRONT)], [(1, FRONT)]), 0, bcfg)


# ------------------------------------------------------------------ property tests
@st.composite
def playouts(draw):
    seed = draw(st.integers(0, 2**32 - 1))
    choices = draw(st.lists(st.integers(0, 7), min_size=0, max_size=12))
    return seed, choices


@settings(max_examples=300, deadline=None)
@given(playouts())
def test_random_playouts_respect_the_rules(data) -> None:
    from resource_war.config import load_all

    cfg = load_all().battle
    seed, choices = data
    rng = np.random.default_rng(seed)
    s = random_start(rng, cfg, int(rng.integers(2)))
    for c in choices:
        if is_terminal(s, cfg):
            break
        moves = legal_moves(s)
        assert moves, "a non-terminal state always has a legal move (DEFEND)"
        for m in moves:  # every generated move is legal
            apply_move(s, m, cfg)
        nxt = apply_move(s, moves[c % len(moves)], cfg)
        hp_changes = [(a.hp, b.hp) for side in (0, 1) for a, b in zip(s.sides[side], nxt.sides[side]) if a.hp != b.hp]
        assert len(hp_changes) <= 1, "one action per ply hits at most one unit"
        assert all(b >= 0 and a - b == 1 for a, b in hp_changes)
        deaths = sum(1 for a, b in hp_changes if b == 0)
        assert deaths <= 1, "two units can never die on the same ply"
        assert any(u.in_battle for u in nxt.sides[0]) or any(u.in_battle for u in nxt.sides[1])
        s = nxt
    assert s.ply <= cfg.ply_cap


def test_every_battle_terminates_within_the_ply_cap(bcfg) -> None:
    rng = np.random.default_rng(7)
    for _ in range(30):
        rec = play_battle(random_start(rng, bcfg, int(rng.integers(2))), (3, 3), bcfg)
        assert len(rec.plies) <= bcfg.ply_cap and is_terminal(rec.final, bcfg)


# ------------------------------------------------------------------ search correctness
def brute_force_value(state: BattleState, root: int, cfg) -> int:
    """Exact game value by full enumeration to the end (no depth limit, no evaluation function).

    Written independently of minimax.py on purpose: only the rules are shared.
    """
    if is_terminal(state, cfg):
        return terminal_utility(state, root, cfg)
    vals = [brute_force_value(apply_move(state, m, cfg), root, cfg) for m in legal_moves(state)]
    return max(vals) if state.to_move == root else min(vals)


def test_minimax_matches_brute_force_on_small_endgames(bcfg) -> None:
    rng = np.random.default_rng(11)
    checked = 0
    for state in generate_positions(400, 11, bcfg):
        remaining = bcfg.ply_cap - state.ply
        if remaining > 4:
            state = state._replace(ply=bcfg.ply_cap - int(rng.integers(1, 5)))
            remaining = bcfg.ply_cap - state.ply
        depth = remaining  # deep enough that the evaluation function is never used
        result = minimax_search(state, depth, bcfg)
        assert result.value == brute_force_value(state, state.to_move, bcfg)
        checked += 1
    assert checked == 400


@pytest.fixture(scope="module")
def equivalence_positions(bcfg):
    return generate_positions(500, 2027, bcfg)


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
def test_alphabeta_returns_same_value_and_move_as_minimax(bcfg, equivalence_positions, depth) -> None:
    """H1a evidence on 500 seeded positions per depth (separate from the frozen E3 set)."""
    saved = 0
    for state in equivalence_positions:
        mm = minimax_search(state, depth, bcfg)
        ab = alphabeta_search(state, depth, bcfg)
        assert (ab.value, ab.move) == (mm.value, mm.move), state.describe()
        assert ab.stats.nodes <= mm.stats.nodes
        saved += mm.stats.nodes - ab.stats.nodes
    if depth >= 3:
        assert saved > 0


def test_tied_evaluations_pick_the_first_move(bcfg) -> None:
    """Two identical Back units: ADVANCE by unit 0 and by unit 1 tie for best (+2).

    Both searches must return the FIRST of the tied moves (unit 0), which is
    the strict-'>' root rule that makes minimax and alpha-beta agree on moves.
    """
    s = S([(2, BACK), (2, BACK)], [(2, BACK), (2, BACK)])
    children = [evaluate(apply_move(s, m, bcfg), 0, bcfg) for m in legal_moves(s)]
    assert children.count(max(children)) == 2  # a genuine tie at depth 1
    for search in (minimax_search, alphabeta_search):
        result = search(s, 1, bcfg)
        assert result.move == Move(0, ADVANCE) and result.value == 2


def test_win_utility_dominates_any_evaluation(bcfg) -> None:
    assert bcfg.win_utility > bcfg.max_abs_eval
    assert math.isfinite(bcfg.win_utility)


# ------------------------------------------------------------------ benchmark file and engine integration
def test_benchmark_roundtrip_and_refuses_overwrite(tmp_path, bcfg) -> None:
    positions = generate_positions(5, 3, bcfg)
    path = tmp_path / "positions.json"
    save_positions(positions, 3, path)
    assert load_positions(path) == positions
    with pytest.raises(FileExistsError, match="frozen"):
        save_positions(positions, 3, path)
    assert all(state_from_json(state_to_json(p)) == p for p in positions)


def test_frozen_benchmark_set_is_present_and_unchanged() -> None:
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    assert data["count"] == len(data["positions"]) == 500
    from resource_war.config import load_all

    assert data["positions"] == [state_to_json(p) for p in generate_positions(500, data["seed"], load_all().battle)]


def test_resolve_battle_compare_mode_and_determinism(bcfg) -> None:
    a = resolve_battle(np.random.default_rng(5), bcfg, (1, 5), compare_minimax=True)
    b = resolve_battle(np.random.default_rng(5), bcfg, (1, 5), compare_minimax=True)
    strip = lambda s: {k: v for k, v in s.items() if k != "timing"}  # noqa: E731
    assert a[0] == b[0] and strip(a[1]) == strip(b[1])
    assert a[1]["minimax_nodes"] >= a[1]["alphabeta_nodes"]


def test_game_with_real_battles_replays_byte_identically(configs) -> None:
    agents = lambda: (ConstantAgent(ATTACK, depth=3), ConstantAgent(ATTACK, depth=5))  # noqa: E731
    first = Game(configs, agents(), 77).run()
    second = Game(configs, agents(), 77).run()
    assert any(e.battle for e in first.encounters)
    assert first.log_bytes() == second.log_bytes()
    assert b"timing" not in first.log_bytes()
