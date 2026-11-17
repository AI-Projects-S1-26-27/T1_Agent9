"""E3: the frozen benchmark set, alpha-beta vs minimax, and the depth-5 gate.

Owner: Raunak Sarawgi, Reviewer: Harshit Agrawal

The benchmark positions are generated ONCE from a fixed seed and saved to
experiments/benchmark/positions.json; ``generate`` refuses to overwrite an
existing file. Results therefore always refer to the same positions, and
nobody can (even accidentally) regenerate the set until a result looks
better.

Positions are realistic mid-battle states: a seeded random start followed
by 0-5 random legal moves, so the set covers wounded units, defended units,
units in the Back rank and withdrawn units, not just fresh starts.

Usage:
    python -m resource_war.battle.benchmark generate      # once, ever
    python -m resource_war.battle.benchmark run           # E3 + depth-5 gate
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

from resource_war.battle.alphabeta import alphabeta_search
from resource_war.battle.minimax import minimax_search
from resource_war.battle.state import BattleState, BUnit, apply_move, is_terminal, legal_moves, random_start
from resource_war.config import BattleConfig, load_all

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_FILE = ROOT / "experiments" / "benchmark" / "positions.json"
RESULTS_FILE = ROOT / "experiments" / "results" / "E3_alphabeta_vs_minimax.csv"
GATE_FILE = ROOT / "experiments" / "results" / "E3_depth5_gate.json"


def generate_positions(count: int, seed: int, cfg: BattleConfig) -> list[BattleState]:
    rng = np.random.default_rng(seed)
    positions: list[BattleState] = []
    while len(positions) < count:
        state = random_start(rng, cfg, int(rng.integers(2)))
        for _ in range(int(rng.integers(0, 6))):
            moves = legal_moves(state)
            nxt = apply_move(state, moves[int(rng.integers(len(moves)))], cfg)
            if is_terminal(nxt, cfg):
                break
            state = nxt
        if not is_terminal(state, cfg):
            positions.append(state)
    return positions


def state_to_json(state: BattleState) -> dict[str, Any]:
    return {"sides": [[list(u) for u in side] for side in state.sides], "to_move": state.to_move, "ply": state.ply}


def state_from_json(data: dict[str, Any]) -> BattleState:
    sides = tuple(tuple(BUnit(int(h), int(p), bool(d)) for h, p, d in side) for side in data["sides"])
    return BattleState((sides[0], sides[1]), int(data["to_move"]), int(data["ply"]))


def save_positions(positions: list[BattleState], seed: int, path: Path = BENCHMARK_FILE, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; the benchmark set is frozen and must never be regenerated")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "count": len(positions), "positions": [state_to_json(s) for s in positions]}
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_positions(path: Path = BENCHMARK_FILE) -> list[BattleState]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m resource_war.battle.benchmark generate` once")
    return [state_from_json(p) for p in json.loads(path.read_text(encoding="utf-8"))["positions"]]


def run_benchmark(positions: list[BattleState], depths: list[int], cfg: BattleConfig) -> list[dict[str, Any]]:
    rows = []
    for index, state in enumerate(positions):
        for depth in depths:
            mm = minimax_search(state, depth, cfg)
            ab = alphabeta_search(state, depth, cfg)
            rows.append(
                {
                    "position": index,
                    "depth": depth,
                    "same_value": mm.value == ab.value,
                    "same_move": mm.move == ab.move,
                    "minimax_nodes": mm.stats.nodes,
                    "alphabeta_nodes": ab.stats.nodes,
                    "alphabeta_pruned": ab.stats.pruned,
                    "node_ratio": ab.stats.nodes / mm.stats.nodes,
                    "minimax_seconds": mm.stats.seconds,
                    "alphabeta_seconds": ab.stats.seconds,
                }
            )
    return rows


def depth5_gate(rows: list[dict[str, Any]], cfg: BattleConfig) -> dict[str, Any]:
    d5 = [r for r in rows if r["depth"] == 5]
    times = [r["minimax_seconds"] for r in d5]
    nodes = [r["minimax_nodes"] for r in d5]
    return {
        "positions": len(d5),
        "gate_seconds": cfg.depth5_gate_seconds,
        "minimax_mean_seconds": statistics.fmean(times),
        "minimax_max_seconds": max(times),
        "minimax_p95_seconds": float(np.percentile(times, 95)),
        "minimax_mean_nodes": statistics.fmean(nodes),
        "minimax_max_nodes": max(nodes),
        "estimate_full_tree_nodes": sum(8**k for k in range(6)),
        "estimate_knuth_moore_best_case_leaves": 8**3 + 8**2 - 1,
        "passed": max(times) <= cfg.depth5_gate_seconds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E3 benchmark: alpha-beta vs minimax")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate", help="create the frozen benchmark set (refuses to overwrite)")
    run = sub.add_parser("run", help="run E3 on the frozen set and check the depth-5 gate")
    run.add_argument("--limit", type=int, default=None, help="use only the first N positions (quick check)")
    args = parser.parse_args(argv)
    configs = load_all()
    e3 = configs.experiments.section("E3_alphabeta_benchmark")
    if args.cmd == "generate":
        seed = configs.experiments.base_seed
        positions = generate_positions(int(e3["positions"]), seed, configs.battle)
        try:
            save_positions(positions, seed)
        except FileExistsError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"wrote {len(positions)} positions to {BENCHMARK_FILE}")
        return 0
    positions = load_positions()[: args.limit]
    rows = run_benchmark(positions, list(e3["depths"]), configs.battle)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gate = depth5_gate(rows, configs.battle)
    GATE_FILE.write_text(json.dumps(gate, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
