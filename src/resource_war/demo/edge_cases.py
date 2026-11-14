"""Demo robustness script: every bad input gets a clear message, never a stack trace.

Owner: Rohit Gangil, Reviewer: Atharv Singhal

Run:  python -m resource_war.demo.edge_cases

Each case feeds one invalid input to the real code path the demo uses and
prints the message a user would see. tests/test_demo.py runs the same list
and checks that every case is rejected with a ConfigError, IllegalMoveError
or ValueError carrying a readable message.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from resource_war.config import DEFAULT_CONFIG_DIR, ConfigError, load_all, parse_payoffs, parse_yaml_text, validate_depth, validate_seed
from resource_war.encounter.matrix import make_game
from resource_war.engine.state import ActionKind, IllegalMoveError, UnitAction

EXPECTED_ERRORS = (ConfigError, IllegalMoveError, ValueError)


def _corrupt_config(file: str, old: str, new: str) -> None:
    tmp = Path(tempfile.mkdtemp())
    try:
        shutil.copytree(DEFAULT_CONFIG_DIR, tmp / "config")
        target = tmp / "config" / file
        target.write_text(target.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
        load_all(tmp / "config")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _missing_config() -> None:
    load_all(Path(tempfile.gettempdir()) / "definitely-not-a-config-dir")


def _illegal_move() -> None:
    from resource_war.agents.library import make_agent
    from resource_war.engine.game import Game

    cfg = load_all()
    game = Game(cfg, (make_agent("equilibrium", cfg.agents, 1), make_agent("equilibrium", cfg.agents, 1)), 0)
    game.step({0: UnitAction(ActionKind.ATTACK)})


def _move_two_steps() -> None:
    from resource_war.agents.library import make_agent
    from resource_war.engine.game import Game

    cfg = load_all()
    game = Game(cfg, (make_agent("equilibrium", cfg.agents, 1), make_agent("equilibrium", cfg.agents, 1)), 0)
    game.step({0: UnitAction(ActionKind.MOVE, (2, 0))})


CASES: list[tuple[str, Callable[[], object]]] = [
    ("Illegal move: Attack outside an encounter", _illegal_move),
    ("Illegal move: a unit jumps two cells", _move_two_steps),
    ("Missing config directory", _missing_config),
    ("Malformed YAML (unclosed bracket)", lambda: parse_yaml_text("attack_cost: [2\n", "uploaded payoffs.yaml")),
    ("YAML that is a list, not a mapping", lambda: parse_yaml_text("- 1\n- 2\n", "uploaded payoffs.yaml")),
    ("NaN payoff in payoffs.yaml", lambda: _corrupt_config("payoffs.yaml", "attack_cost: 2", "attack_cost: .nan")),
    ("Non-numeric payoff in payoffs.yaml", lambda: _corrupt_config("payoffs.yaml", "defend_cost: 1", "defend_cost: one")),
    ("Battle win probabilities summing above 1", lambda: _corrupt_config("payoffs.yaml", "  col: 0.5", "  col: 0.7")),
    ("Uploaded payoff file missing a key", lambda: parse_payoffs(parse_yaml_text("schema_version: 1\nactions: [A, D, E]\n", "upload.yaml"), "upload.yaml")),
    ("Custom matrix with wrong shape", lambda: make_game([[1, 2], [3, 4]], [[1, 2, 3], [4, 5, 6]])),
    ("Custom matrix with a ragged row", lambda: make_game([[1, 2], [3]], [[1, 2], [3, 4]])),
    ("Custom matrix with text in a cell", lambda: make_game([[1, "x"], [3, 4]], [[1, 2], [3, 4]])),
    ("Custom matrix with NaN", lambda: make_game([[1, float("nan")], [3, 4]], [[1, 2], [3, 4]])),
    ("Out-of-range depth (4)", lambda: validate_depth(4, (1, 3, 5))),
    ("Out-of-range depth (0)", lambda: validate_depth(0, (1, 3, 5))),
    ("Negative seed", lambda: validate_seed(-5)),
    ("Seed too large", lambda: validate_seed(2**40)),
    ("Seed that is not a number", lambda: validate_seed("abc")),
]


def run_cases() -> list[tuple[str, str]]:
    """Run every case; return (name, message). A case that does NOT raise is a bug."""
    out = []
    for name, case in CASES:
        try:
            case()
        except EXPECTED_ERRORS as exc:
            out.append((name, f"{type(exc).__name__}: {exc}"))
        else:
            out.append((name, "NOT REJECTED (bug)"))
    return out


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Robustness: invalid inputs and the message the user sees", show_lines=True)
    table.add_column("Invalid input", style="bold")
    table.add_column("Message shown (no stack trace)")
    for name, message in run_cases():
        table.add_row(name, message)
    Console().print(table)


if __name__ == "__main__":
    main()
