"""Playing a whole battle, and the resolver the engine calls for Attack-vs-Attack.

Owner: Harshit Agrawal, Reviewer: Raunak Sarawgi

Each ply, the side to move searches with its OWN depth (its agent's battle
depth); this is where bounded rationality enters the game. Battles are
played with alpha-beta, which is safe because it returns exactly
minimax's move (alphabeta.py). With ``compare_minimax`` every decision is
also searched by plain minimax, the two results are asserted equal, and
both node counts are recorded: this powers the demo's side-by-side view.

Determinism: the summary that goes into the game log contains only
deterministic fields. Wall times live under the "timing" key, which the
engine strips before logging, so replays stay byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from resource_war.battle.alphabeta import alphabeta_search
from resource_war.battle.minimax import minimax_search
from resource_war.battle.state import BattleState, apply_move, hp_in_battle, is_terminal, random_start, winner
from resource_war.config import BattleConfig


@dataclass
class PlyRecord:
    side: int
    state: str
    alphabeta: dict[str, Any]
    minimax: dict[str, Any] | None
    ab_seconds: float
    mm_seconds: float | None


@dataclass
class BattleRecord:
    start: BattleState
    depths: tuple[int, int]
    winner: int | None
    final: BattleState
    plies: list[PlyRecord] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        ab_nodes = sum(p.alphabeta["nodes"] for p in self.plies)
        out: dict[str, Any] = {
            "first_mover": self.start.to_move,
            "start": self.start.describe(),
            "depths": list(self.depths),
            "winner": self.winner,
            "plies": len(self.plies),
            "final_hp": [hp_in_battle(self.final, 0), hp_in_battle(self.final, 1)],
            "moves": [f"s{p.side}:{p.alphabeta['move']}" for p in self.plies],
            "alphabeta_nodes": ab_nodes,
            "decisions": [
                {
                    "side": p.side,
                    "move": p.alphabeta["move"],
                    "value": p.alphabeta["value"],
                    "alphabeta_nodes": p.alphabeta["nodes"],
                    "alphabeta_pruned": p.alphabeta["pruned"],
                    **({"minimax_nodes": p.minimax["nodes"]} if p.minimax else {}),
                }
                for p in self.plies
            ],
        }
        if self.plies and self.plies[0].minimax is not None:
            out["minimax_nodes"] = sum(p.minimax["nodes"] for p in self.plies if p.minimax)
        out["timing"] = {
            "alphabeta_seconds": sum(p.ab_seconds for p in self.plies),
            "minimax_seconds": sum(p.mm_seconds or 0.0 for p in self.plies),
            "per_ply": [[p.ab_seconds, p.mm_seconds] for p in self.plies],
        }
        return out


def play_battle(start: BattleState, depths: tuple[int, int], cfg: BattleConfig, compare_minimax: bool = False) -> BattleRecord:
    state = start
    record = BattleRecord(start, depths, None, start)
    while not is_terminal(state, cfg):
        side = state.to_move
        ab = alphabeta_search(state, depths[side], cfg)
        mm = None
        if compare_minimax:
            mm = minimax_search(state, depths[side], cfg)
            if (mm.move, mm.value) != (ab.move, ab.value):
                raise RuntimeError(f"alpha-beta disagrees with minimax at {state.describe()}: {ab.summary()} vs {mm.summary()}")
        record.plies.append(
            PlyRecord(side, state.describe(), ab.summary(), mm.summary() if mm else None, ab.stats.seconds, mm.stats.seconds if mm else None)
        )
        state = apply_move(state, ab.move, cfg)
    record.final = state
    record.winner = winner(state, cfg)
    return record


def resolve_battle(
    rng: np.random.Generator, cfg: BattleConfig, depths: tuple[int, int], compare_minimax: bool = False
) -> tuple[int | None, dict[str, Any]]:
    """Engine hook: coin for the first mover, seeded start, full battle. Side 0 = player 0."""
    first_mover = int(rng.integers(2))
    start = random_start(rng, cfg, first_mover)
    record = play_battle(start, depths, cfg, compare_minimax)
    return record.winner, record.summary()
