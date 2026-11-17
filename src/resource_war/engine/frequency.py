"""P1 gate: how often do encounters happen under each map/policy variant?

Owner: Vaibhav Malik, Reviewer: Harshit Agrawal

Agents differ only at encounters (DESIGN 3.2). If encounters are rare, all
agents tie and the demo shows no measurable difference, which is what the
instructor's first-round feedback warned about. This module plays every seed
with both players always choosing Expand (so no battle is needed and the
measurement is independent of any agent), and reports encounters per game
for each variant of placement, targeting, tie-breaking and regrowth.

Run:  python -m resource_war.engine.frequency [--seeds 200]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np

from resource_war.agents.base import EncounterState
from resource_war.config import Configs, load_all
from resource_war.encounter.matrix import EXPAND
from resource_war.engine.game import Game

RESULTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results"


class _AlwaysExpand:
    """Measurement stub: always Expand, so encounters never trigger a battle."""

    def __init__(self, depth: int) -> None:
        self.name = "always_expand"
        self.battle_depth = depth

    def choose(self, state: EncounterState, rng: np.random.Generator) -> int:
        return EXPAND

    def observe(self, state: EncounterState, my_action: int, opp_action: int, payoff: Fraction) -> None:
        return None


def _no_battle(*_args):  # pragma: no cover - never called with always-Expand agents
    raise RuntimeError("always-Expand agents never fight")


def measure(configs: Configs, seeds: range) -> dict[str, float]:
    depth = configs.battle.allowed_depths[0]
    counts, values, first_turns = [], [], []
    for seed in seeds:
        game = Game(configs, (_AlwaysExpand(depth), _AlwaysExpand(depth)), seed, battle_resolver=_no_battle)
        result = game.run()
        counts.append(len(result.encounters))
        values += [float(e.value) for e in result.encounters]
        if result.encounters:
            first_turns.append(result.encounters[0].turn)
    return {
        "mean_encounters": statistics.fmean(counts),
        "median_encounters": statistics.median(counts),
        "share_games_with_encounter": sum(1 for c in counts if c) / len(counts),
        "mean_value_at_encounter": statistics.fmean(values) if values else float("nan"),
        "distinct_values_seen": len(set(values)),
        "mean_first_encounter_turn": statistics.fmean(first_turns) if first_turns else float("nan"),
    }


VARIANTS = {
    "placement": ["point_symmetric", "band"],
    "targeting": ["nearest", "value_per_distance"],
    "tie_break": ["player_relative", "absolute"],
    "regrow_turns": [None, 8],
}


def variant_rows(base: Configs, seeds: int, verbose: bool = True) -> list[dict]:
    """Measure every combination in VARIANTS; one row per variant."""
    rows = []
    for placement, targeting, tie_break, regrow in itertools.product(*VARIANTS.values()):
        game_cfg = replace(base.game, placement=placement, targeting=targeting, tie_break=tie_break, regrow_turns=regrow)
        stats = measure(replace(base, game=game_cfg), range(seeds))
        rows.append({"placement": placement, "targeting": targeting, "tie_break": tie_break, "regrow_turns": regrow, "seeds": seeds, **stats})
        if verbose:
            print(
                f"{placement:15s} {targeting:18s} {tie_break:15s} regrow={str(regrow):4s}  "
                f"mean={stats['mean_encounters']:.2f}  games_with_any={stats['share_games_with_encounter']:.2f}  "
                f"mean_V={stats['mean_value_at_encounter']:.2f}"
            )
    return rows


def write_rows(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "P1_encounter_frequency.csv")
    args = parser.parse_args(argv)
    rows = variant_rows(load_all(), args.seeds)
    write_rows(rows, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
