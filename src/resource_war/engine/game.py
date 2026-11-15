"""The seeded, simultaneous turn loop and encounter resolution.

Owner: Vaibhav Malik, Reviewer: Harshit Agrawal

Determinism (DESIGN 3.4): one ``numpy.random.Generator`` created from the seed
is the only source of randomness: map placement, mixed-strategy draws,
battle start positions and first movers all consume it in a fixed order.
Every event is written to a canonical JSON-lines log (sorted keys, exact
fractions as strings), so the same seed and config produce a byte-identical
log on every machine. A test checks this.

Payoff-to-score invariant (DESIGN 3.3): for every encounter that is not
Attack-vs-Attack, the score change applied here is asserted to equal the
payoff-matrix entry exactly. For Attack-vs-Attack the matrix holds an
expectation (p*V - c), and the realized battle outcome (V - c or -c) is
applied instead. This is what makes the game-theoretic analysis predictive
of tournament scores.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np

from resource_war.agents.base import Agent, EncounterState
from resource_war.config import Configs, PayoffConfig, validate_seed
from resource_war.encounter.matrix import (
    ACTION_NAMES,
    ATTACK,
    build_encounter_game,
    realized_gains_and_costs,
)
from resource_war.engine.board import Coord
from resource_war.engine.policy import plan_unit_actions
from resource_war.engine.state import (
    ActionKind,
    GameState,
    IllegalMoveError,
    UnitAction,
    check_unit_action,
    initial_state,
)

# (rng, (depth of player 0, depth of player 1), compare_minimax) -> (winner or None, json-safe summary)
BattleResolver = Callable[[np.random.Generator, tuple[int, int], bool], tuple[int | None, dict[str, Any]]]
# (depth of player 0, depth of player 1) -> (P(player 0 wins), P(player 1 wins))
BattleProbs = Callable[[int, int], tuple[Fraction, Fraction]]


def to_jsonable(obj: Any) -> Any:
    """Canonical JSON form: Fractions as 'p/q' strings, tuples as lists."""
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class EncounterRecord:
    turn: int
    cell: Coord
    value: Fraction
    actions: tuple[int, int]
    payoffs: tuple[Fraction, Fraction]  # realized score change per player
    expected: tuple[Fraction, Fraction]  # matrix entry (an expectation for A-vs-A)
    battle: dict[str, Any] | None = None


@dataclass
class GameResult:
    seed: int
    agent_names: tuple[str, str]
    depths: tuple[int, int]
    turns: int
    scores: tuple[Fraction, Fraction]
    resource_points: tuple[Fraction, Fraction]
    gross_collected: tuple[Fraction, Fraction]
    costs: tuple[Fraction, Fraction]
    territory: tuple[int, int]
    winner: int | None
    encounters: list[EncounterRecord] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)

    def log_bytes(self) -> bytes:
        return ("\n".join(self.log_lines) + "\n").encode("utf-8")


def default_battle_resolver(configs: Configs) -> BattleResolver:
    """Resolve Attack-vs-Attack by actually playing Battle Mode (alpha-beta search)."""
    from resource_war.battle.play import resolve_battle  # imported lazily: battle is optional for engine tests

    def resolver(rng: np.random.Generator, depths: tuple[int, int], compare_minimax: bool):
        return resolve_battle(rng, configs.battle, depths, compare_minimax=compare_minimax)

    return resolver


class Game:
    """One game between two agents. Use ``run()``, or ``step()`` turn by turn (demo)."""

    def __init__(
        self,
        configs: Configs,
        agents: tuple[Agent, Agent],
        seed: int,
        *,
        payoffs: PayoffConfig | None = None,
        battle_probs: BattleProbs | None = None,
        battle_resolver: BattleResolver | None = None,
        compare_minimax: bool = False,
        observer: Callable[[dict[str, Any], GameState], None] | None = None,
    ) -> None:
        """``observer`` (optional) is called after every logged event with the event and the
        current state. The web demo uses it to snapshot the board after each step; it must
        not modify the state."""
        self.observer = observer
        self.seed = validate_seed(seed)
        self.configs = configs
        self.cfg = configs.game
        self.payoffs = payoffs or configs.payoffs
        self.agents = agents
        self.depths = (agents[0].battle_depth, agents[1].battle_depth)
        for d in self.depths:
            if d not in configs.battle.allowed_depths:
                raise ValueError(f"battle depth {d} is not one of {list(configs.battle.allowed_depths)}")
        self.battle_probs = battle_probs or (lambda d0, d1: (self.payoffs.p_row, self.payoffs.p_col))
        self.resolve_battle = battle_resolver or default_battle_resolver(configs)
        self.compare_minimax = compare_minimax
        self.rng = np.random.default_rng(self.seed)
        self.state: GameState = initial_state(self.cfg, self.rng)
        self.encounters: list[EncounterRecord] = []
        self.log_lines: list[str] = []
        self._log(
            {
                "event": "setup",
                "seed": self.seed,
                "agents": [a.name for a in agents],
                "depths": list(self.depths),
                "bases": [list(b) for b in self.cfg.bases],
                "resources": [[list(c), v] for c, v in self.state.resources.items()],
            }
        )

    # ------------------------------------------------------------------ helpers
    def _log(self, event: dict[str, Any]) -> dict[str, Any]:
        self.log_lines.append(canonical_json(event))
        if self.observer is not None:
            self.observer(event, self.state)
        return event

    @property
    def finished(self) -> bool:
        return self.state.turn >= self.cfg.turn_limit

    def scores(self) -> tuple[Fraction, Fraction]:
        w = self.cfg.territory_weight
        return self.state.score(0, w), self.state.score(1, w)

    # ------------------------------------------------------------------ turn
    def step(self, unit_actions: dict[int, UnitAction] | None = None) -> list[dict[str, Any]]:
        """Play one turn. ``unit_actions`` overrides the shared policy (validated)."""
        if self.finished:
            raise IllegalMoveError("the game is over; start a new game or replay the seed")
        st = self.state
        st.turn += 1
        events = [self._log({"event": "turn", "turn": st.turn})]
        events += self._regrow()

        planned = plan_unit_actions(st, self.cfg)
        if unit_actions:
            planned.update(unit_actions)
        units = {u.uid: u for u in st.units}
        for uid in sorted(planned):
            check_unit_action(st, units[uid], planned[uid])

        # 1. simultaneous movement
        for uid in sorted(planned):
            action, unit = planned[uid], units[uid]
            if action.kind is ActionKind.MOVE:
                events.append(self._log({"event": "move", "uid": uid, "player": unit.player, "from": list(unit.pos), "to": list(action.to)}))
                unit.pos = action.to  # type: ignore[assignment]

        # 2. encounters on contested resource cells, in row-major order
        contested = sorted(cell for cell in st.resources if st.occupants(cell) == {0, 1})
        for cell in contested:
            events.append(self._encounter(cell))

        # 3. uncontested collection and expansion
        events += self._collect_and_expand(planned, set(contested))
        if self.finished:
            s0, s1 = self.scores()
            winner = 0 if s0 > s1 else 1 if s1 > s0 else None
            events.append(self._log({"event": "end", "scores": [s0, s1], "winner": winner, "territory": [st.territory(0), st.territory(1)]}))
        return events

    def _regrow(self) -> list[dict[str, Any]]:
        regrow = self.cfg.regrow_turns
        st = self.state
        if regrow is None:
            return []
        events = []
        for cell in sorted(st.depleted):
            value, when = st.depleted[cell]
            if st.turn - when >= regrow:
                st.resources[cell] = value
                del st.depleted[cell]
                events.append(self._log({"event": "regrow", "cell": list(cell), "value": value}))
        st.resources = dict(sorted(st.resources.items()))
        return events

    def _encounter(self, cell: Coord) -> dict[str, Any]:
        st = self.state
        value = st.resources[cell]
        p0, p1 = self.battle_probs(*self.depths)
        games = (
            build_encounter_game(value, self.payoffs, p0, p1),
            build_encounter_game(value, self.payoffs, p1, p0),
        )
        states = tuple(
            EncounterState(value, games[i], st.turn, cell, i, self.depths[i], self.depths[1 - i]) for i in (0, 1)
        )
        actions = []
        for i in (0, 1):
            a = self.agents[i].choose(states[i], self.rng)
            if not isinstance(a, (int, np.integer)) or isinstance(a, bool) or not 0 <= int(a) <= 2:
                raise IllegalMoveError(f"agent {self.agents[i].name} returned an invalid encounter action {a!r}; expected 0 (A), 1 (D) or 2 (E)")
            actions.append(int(a))
        a0, a1 = actions
        battle_summary = None
        winner = None
        if a0 == ATTACK and a1 == ATTACK:
            winner, battle_summary = self.resolve_battle(self.rng, self.depths, self.compare_minimax)
        gains, costs = realized_gains_and_costs(a0, a1, value, self.payoffs, winner)
        payoffs = (gains[0] - costs[0], gains[1] - costs[1])
        expected = games[0].payoff(a0, a1)
        if battle_summary is None and payoffs != expected:
            raise RuntimeError(f"payoff-to-score invariant broken at {cell}: {payoffs} != {expected}")
        for i in (0, 1):
            st.resource_points[i] += payoffs[i]
            st.gross_collected[i] += gains[i]
            st.costs[i] += costs[i]
        if gains[0] > 0 or gains[1] > 0:
            # Somebody obtained the cell's value: it is used up. Encounters never
            # change ownership, so no territory point is added on top of the payoff.
            del st.resources[cell]
            st.depleted[cell] = (value, st.turn)
        record = EncounterRecord(st.turn, cell, value, (a0, a1), payoffs, expected, battle_summary)
        self.encounters.append(record)
        for i in (0, 1):
            self.agents[i].observe(states[i], actions[i], actions[1 - i], payoffs[i])
        return self._log(
            {
                "event": "encounter",
                "turn": st.turn,
                "cell": list(cell),
                "value": value,
                "actions": [ACTION_NAMES[a0], ACTION_NAMES[a1]],
                "payoffs": list(payoffs),
                "expected": list(expected),
                # wall times vary between runs; keep them out of the byte-identical log
                "battle": {k: v for k, v in battle_summary.items() if k != "timing"} if battle_summary else None,
            }
        )

    def _collect_and_expand(self, planned: dict[int, UnitAction], contested: set[Coord]) -> list[dict[str, Any]]:
        st = self.state
        events = []
        units = {u.uid: u for u in st.units}
        for uid in sorted(planned):
            unit = units[uid]
            if planned[uid].kind is not ActionKind.COLLECT:
                continue
            cell = unit.pos
            if cell in contested or cell not in st.resources:
                continue  # contested this turn, or collected by a teammate earlier this turn
            value = st.resources.pop(cell)
            st.resource_points[unit.player] += value
            st.gross_collected[unit.player] += value
            st.owner[cell] = unit.player
            st.depleted[cell] = (value, st.turn)
            events.append(self._log({"event": "collect", "uid": uid, "player": unit.player, "cell": list(cell), "value": value}))
        claims: dict[Coord, set[int]] = {}
        for uid in sorted(planned):
            if planned[uid].kind is ActionKind.EXPAND:
                claims.setdefault(units[uid].pos, set()).add(units[uid].player)
        for cell in sorted(claims):
            players = claims[cell]
            if len(players) == 1 and cell not in st.owner:
                (player,) = players
                st.owner[cell] = player
                events.append(self._log({"event": "expand", "player": player, "cell": list(cell)}))
            elif len(players) == 2:
                events.append(self._log({"event": "expand_blocked", "cell": list(cell)}))
        return events

    # ------------------------------------------------------------------ full game
    def run(self) -> GameResult:
        while not self.finished:
            self.step()
        return self.result()

    def result(self) -> GameResult:
        st = self.state
        s0, s1 = self.scores()
        winner = 0 if s0 > s1 else 1 if s1 > s0 else None
        return GameResult(
            seed=self.seed,
            agent_names=(self.agents[0].name, self.agents[1].name),
            depths=self.depths,
            turns=st.turn,
            scores=(s0, s1),
            resource_points=(st.resource_points[0], st.resource_points[1]),
            gross_collected=(st.gross_collected[0], st.gross_collected[1]),
            costs=(st.costs[0], st.costs[1]),
            territory=(st.territory(0), st.territory(1)),
            winner=winner if self.finished else None,
            encounters=list(self.encounters),
            log_lines=list(self.log_lines),
        )
