"""CLI fallback for the live demo (rich), running the same engine as the Streamlit app.

Owner: Rohit Gangil, Reviewer: Atharv Singhal

    python -m resource_war.demo.cli play --agents equilibrium fixed_mixed --depths 3 3 --seed 7
    python -m resource_war.demo.cli analyze --value 6 --attack-cost 2 --defend-cost 1
    python -m resource_war.demo.cli analyze --matrix my_game.yaml
    python -m resource_war.demo.cli battle --depths 1 5 --seed 3
    python -m resource_war.demo.cli mini --agents expected_utility fixed_mixed --games 20
    python -m resource_war.demo.cli results
    python -m resource_war.demo.cli edge-cases

Every invalid input ends in a one-line red message and exit code 2, never a
stack trace (pass --debug to see the trace while developing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from resource_war.agents.library import AGENT_TYPES, make_agent
from resource_war.config import ConfigError, load_all, parse_payoffs, read_yaml, to_fraction, validate_depth, validate_seed
from resource_war.demo import render
from resource_war.encounter.analysis import analyze
from resource_war.encounter.matrix import build_encounter_game, make_game
from resource_war.engine.game import Game
from resource_war.engine.state import IllegalMoveError

console = Console()
ROOT = Path(__file__).resolve().parents[3]


def _battle_table(summary) -> Table:
    table = Table(title=render.battle_outcome_text(summary), title_style="bold")
    rows = render.battle_rows(summary)
    for col in (rows[0] if rows else {"ply": 0}):
        table.add_column(col, justify="right" if col not in ("side", "move") else "left")
    for row in rows:
        table.add_row(*(str(v) for v in row.values()))
    return table


def cmd_play(args, configs) -> None:
    seed = validate_seed(args.seed)
    if args.payoffs:
        configs = configs.__class__(parse_payoffs(read_yaml(args.payoffs), Path(args.payoffs).name), configs.game, configs.battle, configs.agents, configs.experiments)
    depths = [validate_depth(d, configs.battle.allowed_depths) for d in args.depths]
    agents = tuple(make_agent(k, configs.agents, d, configs.battle.allowed_depths) for k, d in zip(args.agents, depths))
    game = Game(configs, agents, seed, compare_minimax=True)  # type: ignore[arg-type]
    console.print(Panel(f"Player 0: {render.AGENT_LABELS[args.agents[0]]}, depth {depths[0]}\nPlayer 1: {render.AGENT_LABELS[args.agents[1]]}, depth {depths[1]}\nSeed {seed} (replay with the same seed to get the identical game)", title="Setup"))
    shown_cards: set = set()
    while not game.finished:
        events = game.step()
        encounters = [e for e in events if e["event"] == "encounter"]
        if args.every_turn or encounters:
            console.rule(f"Turn {game.state.turn}  |  score {float(game.scores()[0]):.1f} : {float(game.scores()[1]):.1f}")
            console.print(render.board_text(game.state, configs.game.bases))
        for e in encounters:
            record = game.encounters[-len(encounters) + encounters.index(e)]
            console.print(f"[bold]Encounter[/bold] at {tuple(e['cell'])}, V = {e['value']}: player 0 plays {e['actions'][0]}, player 1 plays {e['actions'][1]} -> payoffs {e['payoffs'][0]}, {e['payoffs'][1]}")
            key = record.value
            if args.cards == "all" or (args.cards == "first" and key not in shown_cards):
                shown_cards.add(key)
                g0 = build_encounter_game(record.value, game.payoffs, *game.battle_probs(*game.depths))
                console.print(Panel(analyze(g0).to_text(), title=f"Game Analysis (player 0's view), V = {record.value}"))
            if record.battle:
                console.print(_battle_table(record.battle))
    result = game.result()
    console.rule("Game over")
    console.print(f"Scores: player 0 {float(result.scores[0]):.2f}, player 1 {float(result.scores[1]):.2f}; winner: {'tie' if result.winner is None else 'player ' + str(result.winner)}")
    console.print(f"Resources collected {[float(x) for x in result.gross_collected]}, territory {list(result.territory)}, encounters {len(result.encounters)}")


def cmd_analyze(args, configs) -> None:
    if args.matrix:
        data = read_yaml(args.matrix)
        game = make_game(data.get("row"), data.get("col"), data.get("row_labels"), data.get("col_labels"), Path(args.matrix).name)
    else:
        payoffs = configs.payoffs.with_overrides(
            attack_cost=to_fraction(args.attack_cost, "--attack-cost") if args.attack_cost is not None else configs.payoffs.attack_cost,
            defend_cost=to_fraction(args.defend_cost, "--defend-cost") if args.defend_cost is not None else configs.payoffs.defend_cost,
            p_row=to_fraction(args.p_row, "--p-row") if args.p_row is not None else configs.payoffs.p_row,
            p_col=to_fraction(args.p_col, "--p-col") if args.p_col is not None else configs.payoffs.p_col,
        )
        game = build_encounter_game(args.value, payoffs)
    analysis = analyze(game)
    console.print(Panel(render.structure_banner(analysis), title="In one sentence"))
    console.print(analysis.to_text())


def cmd_battle(args, configs) -> None:
    import numpy as np

    from resource_war.battle.play import resolve_battle

    seed = validate_seed(args.seed)
    depths = tuple(validate_depth(d, configs.battle.allowed_depths) for d in args.depths)
    _, summary = resolve_battle(np.random.default_rng(seed), configs.battle, depths, compare_minimax=True)  # type: ignore[arg-type]
    console.print(f"Start: {summary['start']}")
    console.print(_battle_table(summary))


def cmd_mini(args, configs) -> None:
    from resource_war.tournament.runner import run_matches, side_swapped_specs
    from resource_war.tournament.stats import wilson_interval

    if not 1 <= args.games <= 400:
        raise ConfigError("--games must be between 1 and 400 for a live mini-tournament")
    depth = validate_depth(args.depth, configs.battle.allowed_depths)
    seeds = range(validate_seed(args.seed), args.seed + (args.games + 1) // 2)
    specs = side_swapped_specs("mini", args.agents[0], depth, args.agents[1], depth, seeds)
    rows = [r["game"] for r in run_matches(specs, configs, processes=1, progress=None)]
    n = len(rows)
    table = Table(title=f"Live mini-tournament: {n} games, seeds {seeds.start}-{seeds.stop - 1}, sides swapped")
    for col in ("agent", "wins", "win rate [95% Wilson CI]", "mean score"):
        table.add_column(col)
    for who, key, score in (("first", "first_win", "first_score"), ("second", "second_win", "second_score")):
        wins = sum(r[key] for r in rows)
        table.add_row(args.agents[0] if who == "first" else args.agents[1], str(wins), wilson_interval(wins, n).fmt(), f"{sum(r[score] for r in rows) / n:.2f}")
    table.add_row("ties", str(sum(r["tie"] for r in rows)), wilson_interval(sum(r["tie"] for r in rows), n).fmt(), "")
    console.print(table)


def cmd_results(args, configs) -> None:
    gen = ROOT / "report" / "generated"
    if not (gen / "verdicts.md").exists():
        raise ConfigError("no precomputed results yet: run `python -m resource_war.reproduce` first")
    for name in ("verdicts.md", "e1_pairings.md", "h4_per_value.md", "e3_benchmark.md", "h2_battle_depth.md", "e4_coupling.md"):
        console.rule(name)
        console.print((gen / name).read_text(encoding="utf-8"))


def cmd_edge_cases(args, configs) -> None:
    from resource_war.demo.edge_cases import main as edge_main

    edge_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resource_war.demo.cli", description="Agent9 resource war: live demo (CLI fallback)")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    play = sub.add_parser("play", help="play one game turn by turn")
    play.add_argument("--agents", nargs=2, choices=AGENT_TYPES, default=["equilibrium", "fixed_mixed"])
    play.add_argument("--depths", nargs=2, default=[3, 3])
    play.add_argument("--seed", default=7)
    play.add_argument("--payoffs", help="alternative payoffs YAML file")
    play.add_argument("--cards", choices=["first", "all", "none"], default="first", help="Game Analysis cards: first per cell value (default), all, none")
    play.add_argument("--every-turn", action="store_true", help="draw the board every turn, not only on encounters")
    an = sub.add_parser("analyze", help="Game Analysis card for an encounter or a custom matrix")
    an.add_argument("--value", default=6)
    an.add_argument("--attack-cost")
    an.add_argument("--defend-cost")
    an.add_argument("--p-row")
    an.add_argument("--p-col")
    an.add_argument("--matrix", help="YAML with schema_version: 1, row: [[...]], col: [[...]], optional row_labels/col_labels")
    bt = sub.add_parser("battle", help="one Battle Mode fight with minimax and alpha-beta side by side")
    bt.add_argument("--depths", nargs=2, default=[1, 5])
    bt.add_argument("--seed", default=3)
    mini = sub.add_parser("mini", help="live mini-tournament (default 20 games)")
    mini.add_argument("--agents", nargs=2, choices=AGENT_TYPES, default=["expected_utility", "fixed_mixed"])
    mini.add_argument("--games", type=int, default=20)
    mini.add_argument("--depth", default=3)
    mini.add_argument("--seed", type=int, default=1)
    sub.add_parser("results", help="precomputed tournament results (tables)")
    sub.add_parser("edge-cases", help="show how invalid inputs are rejected")
    return parser


COMMANDS = {"play": cmd_play, "analyze": cmd_analyze, "battle": cmd_battle, "mini": cmd_mini, "results": cmd_results, "edge-cases": cmd_edge_cases}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configs = load_all()
        COMMANDS[args.cmd](args, configs)
    except (ConfigError, IllegalMoveError, ValueError, FileNotFoundError) as exc:
        if args.debug:
            raise
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
