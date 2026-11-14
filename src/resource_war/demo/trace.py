"""Explainable, step-by-step traces of real games for the web demo (watch and play modes).

Owner: Rohit Gangil, Reviewer: Atharv Singhal

The web demo never fakes a game: every frame comes from the real engine
(engine/game.py), the real agents and the real searches.

How a trace is built:
- The engine's ``observer`` hook snapshots the board after every logged event.
- Agents are wrapped in ``ExplainedAgent``, which records WHY each encounter
  decision was made (probabilities, beliefs, expected values) before the
  agent draws its action. Explaining consumes no randomness, so the game is
  identical to an unwrapped one (tested).
- Battles are resolved by ``TracingBattle``, which consumes the random stream
  exactly like battle/play.py's resolver, and additionally records every
  battle state, the ranked value of every legal move (rank_root_moves) and
  the minimax vs alpha-beta node counts.

Play mode ("you" are Blue, player 0) is STATELESS. The browser sends the
seed and the list of decisions made so far; the server replays the game from
the start, using those decisions, until the next decision is needed
(``NeedDecision``). Because the game is deterministic given the seed and
your decisions, rewinding is just "replay with fewer decisions", and any
past choice can be changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any

import numpy as np

from resource_war.agents.base import EncounterState
from resource_war.agents.library import (
    AGENT_TYPES,
    EquilibriumAgent,
    ExpectedUtilityAgent,
    FixedMixedAgent,
    PureHeuristicAgent,
    make_agent,
)
from resource_war.battle.alphabeta import alphabeta_search, rank_root_moves
from resource_war.battle.minimax import minimax_search
from resource_war.battle.state import (
    ADVANCE,
    DEFEND,
    FRONT,
    RETREAT,
    STRIKE,
    WITHDRAWN,
    BattleState,
    Move,
    apply_move,
    front_count,
    hp_in_battle,
    is_terminal,
    random_start,
    winner,
)
from resource_war.config import BattleConfig, ConfigError, Configs, validate_depth, validate_seed
from resource_war.demo.render import structure_banner
from resource_war.encounter.analysis import analyze
from resource_war.encounter.matrix import ACTION_LONG_NAMES, ACTION_NAMES, BimatrixGame, build_encounter_game
from resource_war.engine.game import Game
from resource_war.engine.state import GameState
from resource_war.equilibrium.selection import select_equilibrium
from resource_war.equilibrium.solve import solve_game

PLAYERS = ("Blue", "Orange")
MAX_DECISIONS = 600

AGENT_INFO: dict[str, dict[str, str]] = {
    "pure_heuristic": {"label": "Heuristic", "blurb": "Attacks any crystal worth more than 5, otherwise expands. No game theory, no learning."},
    "fixed_mixed": {"label": "Fixed odds", "blurb": "Rolls the same odds every time (attack 40%, defend 35%, expand 25%), whatever the crystal is worth."},
    "equilibrium": {"label": "Equilibrium", "blurb": "Plays its Nash equilibrium mix for each standoff. It cannot be exploited, but it never exploits you either."},
    "expected_utility": {"label": "Learner", "blurb": "Watches what you do at each crystal value and plays the best reply to those odds (fictitious play)."},
    "depth_limited": {"label": "Depth-limited", "blurb": "Plays standoffs like Equilibrium; only how far it looks ahead in battles differs."},
    "human": {"label": "You", "blurb": "You choose every standoff and every battle move."},
}


class NeedDecision(Exception):
    """Raised mid-game in play mode when the next human decision is not in the script."""

    def __init__(self, kind: str, payload: dict[str, Any]) -> None:
        super().__init__(kind)
        self.kind = kind
        self.payload = payload


def f(x: Fraction | int | float) -> float:
    return float(x)


def fx(x: Fraction) -> str:
    """Exact, readable number: '3', '17/9'."""
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


# ---------------------------------------------------------------------------- cached searches
# Pure functions of (state, depth, config): caching makes play-mode replays fast.
@lru_cache(maxsize=200_000)
def _ranked(state: BattleState, depth: int, cfg: BattleConfig) -> tuple[tuple[Move, int], ...]:
    return tuple(rank_root_moves(state, depth, cfg))


@lru_cache(maxsize=200_000)
def _ab(state: BattleState, depth: int, cfg: BattleConfig):
    return alphabeta_search(state, depth, cfg)


@lru_cache(maxsize=200_000)
def _mm(state: BattleState, depth: int, cfg: BattleConfig):
    return minimax_search(state, depth, cfg)


# ---------------------------------------------------------------------------- explanations
def _pure(i: int) -> list[float]:
    return [1.0 if k == i else 0.0 for k in range(3)]


def explain(agent: Any, state: EncounterState) -> dict[str, Any]:
    """Why ``agent`` will act as it does in ``state``. Uses no randomness."""
    v = state.value
    if isinstance(agent, PureHeuristicAgent):
        attack = v > agent.threshold
        return {
            "rule": f"The crystal is worth {fx(v)}, which is {'above' if attack else 'not above'} my threshold of {fx(agent.threshold)}, so I {'attack' if attack else 'expand'}.",
            "probs": _pure(0 if attack else 2),
        }
    if isinstance(agent, FixedMixedAgent):
        return {"rule": "I ignore the crystal and roll my fixed odds.", "probs": [f(p) for p in agent.probabilities]}
    if isinstance(agent, EquilibriumAgent):  # also DepthLimited
        mix = agent.strategy(state)
        pure = [i for i, p in enumerate(mix) if p == 1]
        if pure:
            rule = f"Game theory says {ACTION_LONG_NAMES[pure[0]]} is the only sensible move here (the equilibrium is pure), so I play it."
        else:
            rule = "No single move is safe here, so I mix: these are my Nash equilibrium odds, which leave you nothing to exploit. Then I draw one."
        return {"rule": rule, "probs": [f(p) for p in mix], "exact": [fx(p) for p in mix]}
    if isinstance(agent, ExpectedUtilityAgent):
        belief = agent.belief(state)
        values = agent.expected_values(state)
        best = values.index(max(values))
        seen = sum(agent.counts.get(agent._key(state), agent.prior)) - sum(agent.prior)
        return {
            "rule": (
                f"I have seen my opponent {int(seen)} time{'s' if seen != 1 else ''} at crystals worth {fx(v)}. "
                f"Against the odds I believe they play, {ACTION_LONG_NAMES[best]} earns the most."
            ),
            "belief": [f(b) for b in belief],
            "values": [f(x) for x in values],
            "probs": _pure(best),
        }
    return {"rule": "", "probs": [0.0, 0.0, 0.0]}


class ExplainedAgent:
    """Wraps a real agent and records the reasoning behind each encounter decision."""

    def __init__(self, inner: Any, kind: str) -> None:
        self.inner = inner
        self.kind = kind
        self.name = inner.name
        self.battle_depth = inner.battle_depth
        self.records: list[dict[str, Any]] = []

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        why = explain(self.inner, state)
        action = self.inner.choose(state, rng)
        self.records.append({"why": why, "action": action, "decision_index": None, "cell": list(state.cell), "value": f(state.value), "turn": state.turn})
        return action

    def observe(self, state: EncounterState, my_action: int, opp_action: int, payoff: Fraction) -> None:
        self.inner.observe(state, my_action, opp_action, payoff)


@dataclass
class DecisionScript:
    """The human's decisions so far, replayed in order."""

    decisions: list[dict[str, Any]]
    pos: int = 0

    def take(self, kind: str, pending: Any) -> tuple[Any, int]:
        if self.pos < len(self.decisions):
            d = self.decisions[self.pos]
            if d.get("kind") != kind:
                raise ValueError(f"decision {self.pos + 1} is a {d.get('kind')!r} decision, but the game needs a {kind!r} decision here")
            index = self.pos
            self.pos += 1
            return d.get("choice"), index
        raise NeedDecision(kind, pending())


class HumanAgent:
    name = "human"
    kind = "human"

    def __init__(self, depth: int, script: DecisionScript) -> None:
        self.battle_depth = depth
        self.script = script
        self.records: list[dict[str, Any]] = []

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        choice, index = self.script.take("encounter", lambda: {"state": state})
        if choice not in ACTION_NAMES:
            raise ValueError(f"decision {index + 1}: a standoff choice must be one of {list(ACTION_NAMES)}, got {choice!r}")
        action = ACTION_NAMES.index(choice)
        self.records.append({"why": {"rule": "Your choice.", "probs": _pure(action)}, "action": action, "decision_index": index, "cell": list(state.cell), "value": f(state.value), "turn": state.turn})
        return action

    def observe(self, state: EncounterState, my_action: int, opp_action: int, payoff: Fraction) -> None:
        return None


# ---------------------------------------------------------------------------- battle tracing
def battle_state_json(s: BattleState) -> dict[str, Any]:
    return {
        "sides": [[{"hp": u.hp, "pos": ("front", "back", "withdrawn")[u.pos], "defending": u.defending, "in_battle": u.in_battle} for u in side] for side in s.sides],
        "to_move": s.to_move,
        "ply": s.ply,
        "hp": [hp_in_battle(s, 0), hp_in_battle(s, 1)],
        "front": [front_count(s, 0), front_count(s, 1)],
    }


def move_label(s: BattleState, m: Move) -> str:
    """What a move would do, phrased for a button or a ranked list."""
    side = s.to_move
    foe = PLAYERS[1 - side]
    if m.action == STRIKE:
        return f"Unit {m.unit + 1} strikes {foe} unit {m.target + 1}"
    if m.action == DEFEND:
        return f"Unit {m.unit + 1} raises its guard"
    if m.action == RETREAT:
        return f"Unit {m.unit + 1} {'falls back to the rear' if s.sides[side][m.unit].pos == FRONT else 'withdraws from the battle'}"
    return f"Unit {m.unit + 1} advances to the front"


def move_story(before: BattleState, m: Move, after: BattleState) -> str:
    side = before.to_move
    me, foe = PLAYERS[side], PLAYERS[1 - side]
    who = f"{me} unit {m.unit + 1}"
    if m.action == STRIKE:
        hp0 = before.sides[1 - side][m.target].hp
        hp1 = after.sides[1 - side][m.target].hp
        target = f"{foe} unit {m.target + 1}"
        if hp1 == hp0:
            return f"{who} strikes {target}, but it was guarding: no damage."
        if hp1 == 0:
            return f"{who} strikes {target} and knocks it out."
        return f"{who} strikes {target}: −1 HP (now {hp1})."
    if m.action == DEFEND:
        return f"{who} raises its guard: the next strike on it does nothing."
    if m.action == RETREAT:
        if after.sides[side][m.unit].pos == WITHDRAWN:
            return f"{who} withdraws from the battle (it survives, but its HP no longer counts)."
        return f"{who} falls back to the rear rank, out of reach of strikes."
    return f"{who} advances to the front rank, where it can strike and be struck."


def value_story(v: int, cfg: BattleConfig) -> str:
    if v >= cfg.win_utility - cfg.max_abs_eval:
        return "a proven win within the search horizon"
    if v <= -(cfg.win_utility - cfg.max_abs_eval):
        return "a proven loss within the search horizon"
    return f"evaluation {v:+d} (10 per HP of lead, 2 per extra unit in the front rank)"


@dataclass
class TracingBattle:
    """Battle resolver for the engine that records everything the demo shows.

    Consumes the random stream exactly like battle.play.resolve_battle (a coin
    for the first mover, then random_start), so a traced game is the same game.
    """

    cfg: BattleConfig
    script: DecisionScript | None = None
    human_side: int | None = None
    records: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, rng: np.random.Generator, depths: tuple[int, int], compare_minimax: bool) -> tuple[int | None, dict[str, Any]]:
        cfg = self.cfg
        first = int(rng.integers(2))
        state = random_start(rng, cfg, first)
        rec: dict[str, Any] = {"depths": list(depths), "first_mover": first, "start": battle_state_json(state), "plies": [], "winner": None, "final": None}
        self.records.append(rec)
        while not is_terminal(state, cfg):
            side = state.to_move
            ranked = _ranked(state, depths[side], cfg)
            moves = [m for m, _ in ranked]
            human = self.human_side == side and self.script is not None
            mm = ab = None
            decision_index = None
            if human:
                snapshot_state = state

                def pending(s=snapshot_state, r=ranked, d=depths[side]):
                    return {"battle": rec, "state": s, "ranked": r, "depth": d}

                choice, decision_index = self.script.take("battle", pending)  # type: ignore[union-attr]
                if isinstance(choice, bool) or not isinstance(choice, int) or not 0 <= choice < len(moves):
                    raise ValueError(f"decision {decision_index + 1}: battle move must be an index from 0 to {len(moves) - 1}, got {choice!r}")
                move = moves[choice]
            else:
                ab = _ab(state, depths[side], cfg)
                mm = _mm(state, depths[side], cfg)
                if (mm.move, mm.value) != (ab.move, ab.value):
                    raise RuntimeError("alpha-beta and minimax disagree; this is a bug")
                move = ab.move
            after = apply_move(state, move, cfg)
            value = dict(ranked)[move]
            ordered = sorted(enumerate(ranked), key=lambda t: (-t[1][1], t[0]))
            rec["plies"].append(
                {
                    "side": side,
                    "human": human,
                    "decision_index": decision_index,
                    "depth": depths[side],
                    "before": battle_state_json(state),
                    "after": battle_state_json(after),
                    "move": {"unit": move.unit, "action": move.action, "target": move.target},
                    "story": move_story(state, move, after),
                    "value": value,
                    "value_story": value_story(value, cfg),
                    "best_value": max(v for _, v in ranked),
                    "ranked": [
                        {"index": i, "label": move_label(state, m), "value": v, "chosen": m == move, "value_story": value_story(v, cfg)}
                        for i, (m, v) in ordered
                    ],
                    "alphabeta": None if ab is None else {"nodes": ab.stats.nodes, "pruned": ab.stats.pruned, "ms": ab.stats.seconds * 1000},
                    "minimax": None if mm is None else {"nodes": mm.stats.nodes, "ms": mm.stats.seconds * 1000},
                }
            )
            state = after
        w = winner(state, cfg)
        rec["winner"] = w
        rec["final"] = battle_state_json(state)
        return w, {"winner": w, "first_mover": first, "plies": len(rec["plies"]), "battle_index": len(self.records) - 1}


# ---------------------------------------------------------------------------- board snapshots
def snapshot(state: GameState, weight: Fraction) -> dict[str, Any]:
    return {
        "turn": state.turn,
        "units": [[u.uid, u.player, u.pos[0], u.pos[1]] for u in state.units],
        "resources": [[r, c, f(v)] for (r, c), v in sorted(state.resources.items())],
        "owner": [[r, c, p] for (r, c), p in sorted(state.owner.items())],
        "scores": [f(state.score(p, weight)) for p in (0, 1)],
        "collected": [f(x) for x in state.gross_collected],
        "points": [f(x) for x in state.resource_points],
        "territory": [state.territory(0), state.territory(1)],
    }


# ---------------------------------------------------------------------------- encounter payloads
def matrix_json(game: BimatrixGame) -> dict[str, Any]:
    return {
        "labels": list(ACTION_NAMES),
        "cells": [[[fx(game.row_payoffs[i][j]), fx(game.col_payoffs[i][j])] for j in range(3)] for i in range(3)],
    }


def analysis_json(game: BimatrixGame) -> dict[str, Any]:
    a = analyze(game)
    strict = [ACTION_LONG_NAMES[d.dominated] + f" (by {ACTION_LONG_NAMES[d.dominator]})" for d in a.strict_dominated[0]]
    strict_idx = {d.dominated for d in a.strict_dominated[0]}
    weak = [ACTION_LONG_NAMES[d.dominated] + f" (by {ACTION_LONG_NAMES[d.dominator]})" for d in a.weak_dominated[0] if d.dominated not in strict_idx]
    sel = a.selected_row
    eq_text = (
        "Equilibrium: both play " + ", ".join(f"{ACTION_LONG_NAMES[i]} {fx(p)}" for i, p in enumerate(sel.row) if p > 0)
        + f", each expecting {fx(sel.row_payoff)} ({f(sel.row_payoff):.2f})."
        if sel.row == a.selected_col_strategy
        else "Equilibrium (selected): Blue " + ", ".join(f"{ACTION_NAMES[i]} {fx(p)}" for i, p in enumerate(sel.row) if p > 0)
        + "; Orange " + ", ".join(f"{ACTION_NAMES[i]} {fx(p)}" for i, p in enumerate(a.selected_col_strategy) if p > 0) + "."
    )
    pareto = ""
    if a.pareto_dominated_equilibria:
        e, cells = a.pareto_dominated_equilibria[0]
        i, j = cells[0]
        pareto = (
            f"Both would do better at ({ACTION_LONG_NAMES[i]}, {ACTION_LONG_NAMES[j]}) = ({fx(game.row_payoffs[i][j])}, {fx(game.col_payoffs[i][j])}) "
            "than at the equilibrium, but it is not stable: each side would be tempted to deviate."
        )
    return {
        "banner": structure_banner(a),
        "strictly_dominated": strict,
        "weakly_dominated": weak,
        "pure_ne": [[ACTION_NAMES[i], ACTION_NAMES[j]] for i, j in a.pure_ne],
        "equilibrium_text": eq_text,
        "equilibrium_row": [f(p) for p in sel.row],
        "equilibrium_col": [f(p) for p in a.selected_col_strategy],
        "degenerate": a.equilibria.degenerate,
        "pareto_text": pareto,
    }


OUTCOME = {
    ("E", "E"): "They both build, so they split it: {v2} each.",
    ("A", "E"): "{a} seizes the crystal from the builder: {a} +{vmc} (the attack costs {c}), {b} gets nothing.",
    ("A", "D"): "{b}'s guard holds and {b} keeps the crystal: {b} +{vmd} (guarding costs {d}); {a} pays the attack cost of {c}.",
    ("D", "D"): "Both dig in and nobody builds: each pays {d}, and the crystal stays contested.",
    ("D", "E"): "Nobody attacks the builder, so {b} takes the crystal (+{v}); {a} wasted {d} guarding.",
    ("A", "A"): "Both attack: Battle Mode decides who takes the crystal.",
}


def outcome_text(actions: tuple[int, int], value: Fraction, cfg: Configs) -> str:
    names = (ACTION_NAMES[actions[0]], ACTION_NAMES[actions[1]])
    order = {"A": 0, "D": 1, "E": 2}
    a, b = PLAYERS
    key = names
    if order[names[0]] > order[names[1]]:
        key = (names[1], names[0])
        a, b = b, a
    c, d = cfg.payoffs.attack_cost, cfg.payoffs.defend_cost
    return OUTCOME[key].format(a=a, b=b, v=fx(value), v2=fx(value / 2), vmc=fx(value - c), vmd=fx(value - d), c=fx(c), d=fx(d))


# ---------------------------------------------------------------------------- frames
@dataclass
class _Recorder:
    weight: Fraction
    events: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)

    def __call__(self, event: dict[str, Any], state: GameState) -> None:
        self.events.append((dict(event), snapshot(state, self.weight)))


def _player_payload(agent: Any, kind: str, depth: int, record: dict[str, Any], payoff: Fraction | None) -> dict[str, Any]:
    info = AGENT_INFO[kind]
    return {
        "kind": kind,
        "label": info["label"],
        "depth": depth,
        "action": ACTION_NAMES[record["action"]],
        "action_name": ACTION_LONG_NAMES[record["action"]],
        "why": record["why"],
        "payoff": None if payoff is None else fx(payoff),
        "decision_index": record.get("decision_index"),
    }


def _turn_story(events: list[dict[str, Any]], include_moves: bool) -> list[str]:
    lines = []
    for e in events:
        if e["event"] == "regrow":
            lines.append(f"The crystal at {tuple(e['cell'])} grows back (worth {e['value']}).")
        elif e["event"] == "collect":
            lines.append(f"{PLAYERS[e['player']]} collects a {e['value']}-crystal at {tuple(e['cell'])} and claims the tile.")
        elif e["event"] == "expand":
            lines.append(f"{PLAYERS[e['player']]} claims the empty tile {tuple(e['cell'])}.")
        elif e["event"] == "expand_blocked":
            lines.append(f"Both sides tried to claim {tuple(e['cell'])} at once: neither gets it.")
    moves = [e for e in events if e["event"] == "move"]
    if include_moves and moves:
        lines.insert(0, "Every unit steps toward the most valuable crystal for its distance (the shared movement rule, identical for every agent).")
    if not lines:
        lines.append("Units keep moving; nothing is collected this turn.")
    return lines


def build_frames(
    configs: Configs,
    kinds: tuple[str, str],
    depths: tuple[int, int],
    agents: tuple[Any, Any],
    recorder: _Recorder,
    battles: TracingBattle,
    seed: int,
    pending: NeedDecision | None = None,
    pending_state: GameState | None = None,
) -> list[dict[str, Any]]:
    w = configs.game.territory_weight
    events = recorder.events
    setup_event, setup_snap = events[0]
    frames: list[dict[str, Any]] = [
        {
            "kind": "setup",
            "turn": 0,
            "title": "The map",
            "text": [
                f"{PLAYERS[0]} ({AGENT_INFO[kinds[0]]['label']}, battle depth {depths[0]}) starts top-left; "
                f"{PLAYERS[1]} ({AGENT_INFO[kinds[1]]['label']}, battle depth {depths[1]}) starts bottom-right.",
                "Twelve crystals lie in mirrored pairs, so the map is fair. A used crystal grows back 8 turns later.",
                "When both sides reach the same crystal, they each secretly choose Attack, Defend or Expand.",
                f"Seed {seed}: the same seed always replays the same game.",
            ],
            "board": setup_snap,
            "highlight": {"cells": [], "moves": []},
        }
    ]
    enc_index = 0
    battle_index = 0
    i = 1
    n = len(events)
    while i < n:
        event, snap = events[i]
        if event["event"] != "turn":
            i += 1
            continue
        turn = event["turn"]
        j = i + 1
        group: list[tuple[dict[str, Any], dict[str, Any]]] = []
        while j < n and events[j][0]["event"] not in ("turn",):
            group.append(events[j])
            j += 1
        moves = [(e, s) for e, s in group if e["event"] == "move"]
        move_arrows = [[*e["from"], *e["to"], e["player"]] for e, _ in moves]
        encounters = [(e, s) for e, s in group if e["event"] == "encounter"]
        before_enc = [(e, s) for e, s in group if e["event"] in ("regrow", "move")]
        after_enc = [(e, s) for e, s in group if e["event"] in ("collect", "expand", "expand_blocked")]
        end = [(e, s) for e, s in group if e["event"] == "end"]
        last_snap = group[-1][1] if group else snap
        in_progress = pending is not None and j >= n
        if not encounters and not in_progress:
            frames.append(
                {"kind": "turn", "turn": turn, "title": f"Turn {turn}", "text": _turn_story([e for e, _ in group], True), "board": last_snap,
                 "highlight": {"cells": [e["cell"] for e, _ in after_enc if "cell" in e], "moves": move_arrows}}
            )
        else:
            move_snap = (before_enc[-1][1] if before_enc else snap)
            frames.append(
                {"kind": "turn", "turn": turn, "title": f"Turn {turn}: units move", "text": _turn_story([e for e, _ in before_enc], True), "board": move_snap,
                 "highlight": {"cells": [], "moves": move_arrows}}
            )
            prev_snap = move_snap
            for e, s in encounters:
                frames += _encounter_frames(configs, kinds, depths, agents, battles, e, prev_snap, s, enc_index, battle_index)
                if e.get("battle"):
                    battle_index += 1
                enc_index += 1
                prev_snap = s
            if after_enc and not in_progress:
                frames.append(
                    {"kind": "turn", "turn": turn, "title": f"Turn {turn}: collection", "text": _turn_story([e for e, _ in after_enc], False), "board": after_enc[-1][1],
                     "highlight": {"cells": [e["cell"] for e, _ in after_enc if "cell" in e], "moves": []}}
                )
            if in_progress and pending is not None and pending_state is not None:
                frames += _pending_frames(configs, kinds, depths, agents, battles, pending, snapshot(pending_state, w), enc_index, battle_index)
        for e, s in end:
            winner_ = e["winner"]
            sc = s["scores"]
            frames.append(
                {
                    "kind": "end",
                    "turn": turn,
                    "title": "Game over" if winner_ is None else f"{PLAYERS[winner_]} wins",
                    "text": [
                        f"Final score: {PLAYERS[0]} {sc[0]:g}, {PLAYERS[1]} {sc[1]:g}" + (" (a draw)." if winner_ is None else "."),
                        f"Crystals collected: {s['collected'][0]:g} vs {s['collected'][1]:g}. Tiles held: {s['territory'][0]} vs {s['territory'][1]}.",
                        f"Standoffs: {enc_index}. Battles: {battle_index}.",
                    ],
                    "board": s,
                    "winner": winner_,
                    "highlight": {"cells": [], "moves": []},
                }
            )
        i = j
    return frames


def _encounter_core(configs, kinds, depths, agents, cell, value, k):
    p0, p1 = configs.payoffs.p_row, configs.payoffs.p_col
    game = build_encounter_game(value, configs.payoffs, p0, p1)
    recs = (agents[0].records[k], agents[1].records[k])
    return game, recs


def _encounter_frames(configs, kinds, depths, agents, battles, event, before, after, k, b):
    cell = event["cell"]
    value = Fraction(event["value"])
    game, recs = _encounter_core(configs, kinds, depths, agents, cell, value, k)
    actions = (recs[0]["action"], recs[1]["action"])
    payoffs = tuple(Fraction(x) for x in event["payoffs"])
    base = {
        "cell": cell,
        "value": f(value),
        "matrix": matrix_json(game),
        "chosen": list(actions),
        "analysis": analysis_json(game),
        "players": [_player_payload(agents[p], kinds[p], depths[p], recs[p], payoffs[p]) for p in (0, 1)],
    }
    head = f"Standoff at {tuple(cell)} over a {fx(value)}-crystal: {PLAYERS[0]} chooses {ACTION_LONG_NAMES[actions[0]]}, {PLAYERS[1]} chooses {ACTION_LONG_NAMES[actions[1]]}."
    frames = []
    if event.get("battle"):
        battle = battles.records[b]
        pre = dict(base, players=[dict(p, payoff=None) for p in base["players"]], battle_index=b)
        frames.append({"kind": "encounter", "turn": event["turn"], "title": f"Standoff at {tuple(cell)}", "text": [head, outcome_text(actions, value, configs)], "board": before,
                       "highlight": {"cells": [cell], "moves": []}, "encounter": pre, "decision_index": recs[0].get("decision_index")})
        frames += _battle_frames(battle, event["turn"], before, cell)
        w = event["battle"]["winner"]
        result = (
            f"{PLAYERS[w]} wins the battle and takes the crystal: {PLAYERS[w]} {fx(payoffs[w]):s}, {PLAYERS[1 - w]} {fx(payoffs[1 - w])} (both paid the attack cost)."
            if w is not None
            else f"The battle is a draw: nobody gets the crystal, and both pay the attack cost ({fx(payoffs[0])} each)."
        )
        frames.append({"kind": "encounter_result", "turn": event["turn"], "title": "Battle result", "text": [result], "board": after,
                       "highlight": {"cells": [cell], "moves": []}, "encounter": dict(base, battle_index=b)})
    else:
        frames.append({"kind": "encounter", "turn": event["turn"], "title": f"Standoff at {tuple(cell)}", "text": [head, outcome_text(actions, value, configs)], "board": after,
                       "highlight": {"cells": [cell], "moves": []}, "encounter": base, "decision_index": recs[0].get("decision_index")})
    return frames


def _battle_frames(battle: dict[str, Any], turn: int, board: dict[str, Any], cell) -> list[dict[str, Any]]:
    first = PLAYERS[battle["first_mover"]]
    frames = [
        {
            "kind": "battle_start",
            "turn": turn,
            "title": "Battle begins",
            "text": [
                f"Two units each, in a front and a rear rank. {first} moves first (a coin toss). Up to 10 moves in total.",
                f"Each side searches ahead with minimax: {PLAYERS[0]} looks {battle['depths'][0]} moves deep, {PLAYERS[1]} {battle['depths'][1]}.",
                "Whoever has more HP left in the fight at the end wins the crystal. A side with no units left in the fight loses.",
            ],
            "board": board,
            "highlight": {"cells": [cell], "moves": []},
            "battle": {"state": battle["start"], "depths": battle["depths"], "ply": None},
        }
    ]
    for k, ply in enumerate(battle["plies"]):
        mover = PLAYERS[ply["side"]]
        text = [ply["story"]]
        if ply["human"]:
            text.append(f"Your choice. The engine (depth {ply['depth']}) rates it {ply['value_story']}; its best was {ply['best_value']:+d}.")
        else:
            saved = 1 - ply["alphabeta"]["nodes"] / ply["minimax"]["nodes"]
            text.append(f"{mover} searched {ply['depth']} moves ahead and judged this {ply['value_story']}.")
            text.append(
                f"Minimax examined {ply['minimax']['nodes']:,} positions; alpha-beta found the same move after {ply['alphabeta']['nodes']:,} ({saved:.0%} fewer)."
            )
        frames.append(
            {"kind": "battle_ply", "turn": turn, "title": f"Battle move {k + 1}: {mover}", "text": text, "board": board,
             "highlight": {"cells": [cell], "moves": []}, "battle": {"state": ply["after"], "depths": battle["depths"], "ply": ply, "index": k},
             "decision_index": ply["decision_index"]}
        )
    return frames


def _pending_frames(configs, kinds, depths, agents, battles, pending: NeedDecision, board, k, b) -> list[dict[str, Any]]:
    if pending.kind == "encounter":
        st: EncounterState = pending.payload["state"]
        game = st.game
        opp = kinds[1]
        sel = select_equilibrium(solve_game(game.swap_players()).equilibria)  # opponent's equilibrium strategy
        opp_eq = sel.row
        ev = [sum(game.row_payoffs[i][j] * opp_eq[j] for j in range(3)) for i in range(3)]
        return [
            {
                "kind": "pending",
                "turn": st.turn,
                "title": "Your move: standoff",
                "text": [
                    f"Your unit and the {AGENT_INFO[opp]['label']} meet on a {fx(st.value)}-crystal at {tuple(st.cell)}.",
                    "Choose secretly: Attack, Defend or Expand. Your opponent is choosing at the same moment.",
                ],
                "board": board,
                "highlight": {"cells": [list(st.cell)], "moves": []},
                "pending": {
                    "kind": "encounter",
                    "cell": list(st.cell),
                    "value": f(st.value),
                    "matrix": matrix_json(game),
                    "analysis": analysis_json(game),
                    "opponent": {"kind": opp, **AGENT_INFO[opp]},
                    "ev_vs_equilibrium": [f(x) for x in ev],
                    "ev_text": "If your opponent played its equilibrium odds, each choice would earn you: "
                    + ", ".join(f"{ACTION_LONG_NAMES[i]} {f(ev[i]):.2f}" for i in range(3)) + ".",
                },
            }
        ]
    # battle pending: the encounter decision frame and the plies so far, then the choice
    battle = pending.payload["battle"]
    frames = []
    st0 = agents[0].records[k]
    st1 = agents[1].records[k]
    cell = st0["cell"]
    frames.append(
        {"kind": "encounter", "turn": board["turn"], "title": f"Standoff at {tuple(cell)}", "text": [f"You both attacked the {st0['value']:g}-crystal: Battle Mode decides who takes it."], "board": board,
         "highlight": {"cells": [cell], "moves": []}, "decision_index": st0["decision_index"],
         "encounter": {"players": [_player_payload(agents[p], kinds[p], depths[p], (st0, st1)[p], None) for p in (0, 1)], "chosen": [st0["action"], st1["action"]], "battle_index": b}}
    )
    frames += _battle_frames(battle, board["turn"], board, cell)
    state: BattleState = pending.payload["state"]
    ranked = pending.payload["ranked"]
    cfg = configs.battle
    frames.append(
        {
            "kind": "pending",
            "turn": board["turn"],
            "title": "Your move: battle",
            "text": [f"Choose one move for one of your units. You look {pending.payload['depth']} moves ahead; the advice shows what that search thinks of each move."],
            "board": board,
            "highlight": {"cells": [cell], "moves": []},
            "battle": {"state": battle_state_json(state), "depths": battle["depths"], "ply": None},
            "pending": {
                "kind": "battle",
                "moves": [{"index": i, "label": move_label(state, m), "value": v, "value_story": value_story(v, cfg)} for i, (m, v) in enumerate(ranked)],
                "best_value": max(v for _, v in ranked),
            },
        }
    )
    return frames


# ---------------------------------------------------------------------------- public API
def _validate_kind(kind: Any, what: str) -> str:
    if kind not in AGENT_TYPES:
        raise ConfigError(f"{what} must be one of {list(AGENT_TYPES)}, got {kind!r}")
    return kind


def watch(configs: Configs, kinds: tuple[str, str], depths: tuple[int, int], seed: int) -> dict[str, Any]:
    """Full AI-vs-AI trace."""
    kinds = (_validate_kind(kinds[0], "Blue's agent"), _validate_kind(kinds[1], "Orange's agent"))
    depths = tuple(validate_depth(d, configs.battle.allowed_depths, "battle depth") for d in depths)  # type: ignore[assignment]
    seed = validate_seed(seed)
    agents = tuple(ExplainedAgent(make_agent(k, configs.agents, d, configs.battle.allowed_depths), k) for k, d in zip(kinds, depths))
    recorder = _Recorder(configs.game.territory_weight)
    battles = TracingBattle(configs.battle)
    game = Game(configs, agents, seed, battle_resolver=battles, observer=recorder)  # type: ignore[arg-type]
    result = game.run()
    frames = build_frames(configs, kinds, depths, agents, recorder, battles, seed)
    return {
        "mode": "watch",
        "seed": seed,
        "kinds": list(kinds),
        "depths": list(depths),
        "agents": [{"kind": k, **AGENT_INFO[k]} for k in kinds],
        "frames": frames,
        "finished": True,
        "winner": result.winner,
        "scores": [f(s) for s in result.scores],
        "pending": None,
    }


def play(configs: Configs, opponent: str, depths: tuple[int, int], seed: int, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Human (Blue) vs an AI (Orange), replayed from ``decisions`` up to the next decision."""
    opponent = _validate_kind(opponent, "opponent")
    depths = tuple(validate_depth(d, configs.battle.allowed_depths, "battle depth") for d in depths)  # type: ignore[assignment]
    seed = validate_seed(seed)
    if not isinstance(decisions, list) or len(decisions) > MAX_DECISIONS:
        raise ConfigError(f"decisions must be a list of at most {MAX_DECISIONS} items")
    for d in decisions:
        if not isinstance(d, dict) or d.get("kind") not in ("encounter", "battle"):
            raise ConfigError("each decision must look like {\"kind\": \"encounter\"|\"battle\", \"choice\": ...}")
    script = DecisionScript(list(decisions))
    human = HumanAgent(depths[0], script)
    ai = ExplainedAgent(make_agent(opponent, configs.agents, depths[1], configs.battle.allowed_depths), opponent)
    agents = (human, ai)
    kinds = ("human", opponent)
    recorder = _Recorder(configs.game.territory_weight)
    battles = TracingBattle(configs.battle, script, human_side=0)
    game = Game(configs, agents, seed, battle_resolver=battles, observer=recorder)  # type: ignore[arg-type]
    pending = None
    try:
        game.run()
    except NeedDecision as need:
        pending = need
    if pending is None and script.pos < len(script.decisions):
        raise ConfigError(f"the game ended after {script.pos} decisions, but {len(script.decisions)} were given")
    frames = build_frames(configs, kinds, depths, agents, recorder, battles, seed, pending, game.state if pending else None)
    finished = pending is None
    return {
        "mode": "play",
        "seed": seed,
        "kinds": list(kinds),
        "depths": list(depths),
        "agents": [{"kind": k, **AGENT_INFO[k]} for k in kinds],
        "frames": frames,
        "finished": finished,
        "winner": frames[-1].get("winner") if finished else None,
        "scores": frames[-1]["board"]["scores"],
        "pending": frames[-1]["pending"] if pending else None,
        "decisions": decisions,
    }


def explore(configs: Configs, value: Any, c: Any, d: Any, p_row: Any, p_col: Any) -> dict[str, Any]:
    """Configuration explorer: the encounter game for any V, c, d and battle-win probabilities."""
    from resource_war.config import to_fraction

    payoffs = configs.payoffs.with_overrides(
        attack_cost=to_fraction(c, "attack cost"), defend_cost=to_fraction(d, "defend cost"),
        p_row=to_fraction(p_row, "Blue's battle-win probability"), p_col=to_fraction(p_col, "Orange's battle-win probability"),
    )
    game = build_encounter_game(to_fraction(value, "crystal value"), payoffs)
    a = analysis_json(game)
    two_c = 2 * payoffs.attack_cost
    v = to_fraction(value, "crystal value")
    regime = "below" if v < two_c else "exactly at" if v == two_c else "above"
    return {"matrix": matrix_json(game), "analysis": a, "regime": regime, "two_c": f(two_c)}
