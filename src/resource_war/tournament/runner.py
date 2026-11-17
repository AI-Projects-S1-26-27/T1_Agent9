"""Seeded, side-swapped, multi-process tournament runner.

Owner: Manas Tripathi, Reviewer: Jainam Parekh Naresh

Common random numbers: every seed is played TWICE, once with each agent as
player 0. Both games share the same map, so map luck and any first-mover or
position effect cancel within the pair. The pair is the unit the bootstrap
resamples (stats.py).

Every result row is a plain dict of JSON-safe values (exact fractions as
'p/q' strings plus float copies), so rows can cross process boundaries and
be written straight to CSV. Each game is fully determined by (agent kinds,
depths, seed, side order, config), so any row can be replayed in the demo.

Before a long run, the runner times a few games and prints an estimated
runtime, so nobody starts a 40-minute run by accident before a demo.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from resource_war.agents.library import make_agent
from resource_war.config import Configs, load_all
from resource_war.encounter.matrix import ACTION_NAMES
from resource_war.engine.game import Game

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MatchSpec:
    """One game: agent kinds and depths for (player 0, player 1), a seed, and which side is 'first'."""

    experiment: str
    pairing: str
    kinds: tuple[str, str]
    depths: tuple[int, int]
    seed: int
    swapped: bool  # True when the pairing's SECOND agent is player 0
    battle_probs: tuple[tuple[str, str], ...] | None = None  # ((p0, p1),) as strings, or None for config p


def side_swapped_specs(experiment: str, kind_a: str, depth_a: int, kind_b: str, depth_b: int, seeds: Iterable[int], label: str | None = None) -> list[MatchSpec]:
    label = label or f"{kind_a}@{depth_a} vs {kind_b}@{depth_b}"
    specs = []
    for s in seeds:
        specs.append(MatchSpec(experiment, label, (kind_a, kind_b), (depth_a, depth_b), s, False))
        specs.append(MatchSpec(experiment, label, (kind_b, kind_a), (depth_b, depth_a), s, True))
    return specs


_WORKER_CONFIGS: Configs | None = None


def _init_worker(config_dir: str | None) -> None:
    global _WORKER_CONFIGS
    _WORKER_CONFIGS = load_all(config_dir)


def play_match(spec: MatchSpec, configs: Configs | None = None) -> dict[str, Any]:
    """Play one game and return a game row plus its encounter rows (all JSON-safe)."""
    cfgs = configs or _WORKER_CONFIGS or load_all()
    agents = tuple(make_agent(k, cfgs.agents, d, cfgs.battle.allowed_depths) for k, d in zip(spec.kinds, spec.depths))
    probs = None
    if spec.battle_probs:
        p0, p1 = (Fraction(x) for x in spec.battle_probs[0])
        probs = lambda d0, d1: (p0, p1)  # noqa: E731
    start = time.perf_counter()
    result = Game(cfgs, agents, spec.seed, battle_probs=probs).run()  # type: ignore[arg-type]
    seconds = time.perf_counter() - start
    # "first" = the pairing's first-named agent, whichever side it played
    first = 1 if spec.swapped else 0
    second = 1 - first
    winner = result.winner
    row = {
        "experiment": spec.experiment,
        "pairing": spec.pairing,
        "seed": spec.seed,
        "swapped": spec.swapped,
        "first_kind": spec.kinds[first],
        "second_kind": spec.kinds[second],
        "first_depth": spec.depths[first],
        "second_depth": spec.depths[second],
        "first_score": float(result.scores[first]),
        "second_score": float(result.scores[second]),
        "score_diff": float(result.scores[first] - result.scores[second]),
        "first_win": int(winner == first),
        "second_win": int(winner == second),
        "tie": int(winner is None),
        "first_resources": float(result.gross_collected[first]),
        "second_resources": float(result.gross_collected[second]),
        "first_territory": result.territory[first],
        "second_territory": result.territory[second],
        "first_net_points": float(result.resource_points[first]),
        "second_net_points": float(result.resource_points[second]),
        "encounters": len(result.encounters),
        "battles": sum(1 for e in result.encounters if e.battle),
        "seconds": seconds,
    }
    encounters = [
        {
            "experiment": spec.experiment,
            "pairing": spec.pairing,
            "seed": spec.seed,
            "swapped": spec.swapped,
            "turn": e.turn,
            "value": float(e.value),
            "first_action": ACTION_NAMES[e.actions[first]],
            "second_action": ACTION_NAMES[e.actions[second]],
            "first_payoff": float(e.payoffs[first]),
            "second_payoff": float(e.payoffs[second]),
            "battle_winner": (None if not e.battle else ("first" if e.battle["winner"] == first else "second" if e.battle["winner"] == second else "tie")),
        }
        for e in result.encounters
    ]
    return {"game": row, "encounters": encounters}


def estimate_runtime(specs: list[MatchSpec], configs: Configs, processes: int, sample: int = 4) -> float:
    """Time a few games sequentially and extrapolate (wall time, given ``processes`` workers)."""
    if not specs:
        return 0.0
    step = max(1, len(specs) // sample)
    probe = specs[::step][:sample]
    start = time.perf_counter()
    for spec in probe:
        play_match(spec, configs)
    per_game = (time.perf_counter() - start) / len(probe)
    return per_game * len(specs) / max(1, processes)


def run_matches(
    specs: list[MatchSpec],
    configs: Configs | None = None,
    processes: int | None = None,
    config_dir: str | None = None,
    progress: Callable[[str], None] | None = print,
) -> list[dict[str, Any]]:
    """Run all matches (in parallel when processes > 1), preserving input order."""
    configs = configs or load_all(config_dir)
    processes = processes or max(1, (os.cpu_count() or 2) - 1)
    if progress:
        eta = estimate_runtime(specs, configs, processes)
        progress(f"  {len(specs)} games on {processes} process(es); estimated {eta:.0f} s")
    if processes == 1 or len(specs) < 8:
        return [play_match(s, configs) for s in specs]
    with ProcessPoolExecutor(max_workers=processes, initializer=_init_worker, initargs=(config_dir,)) as pool:
        return list(pool.map(play_match, specs, chunksize=max(1, len(specs) // (processes * 8))))


def git_commit() -> str:
    """Commit hash for result metadata, or a clear marker when there is none."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return "no-commit-yet"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("+uncommitted-changes" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "git-unavailable"


def with_payoffs(configs: Configs, **changes: Any) -> Configs:
    return replace(configs, payoffs=configs.payoffs.with_overrides(**changes))
