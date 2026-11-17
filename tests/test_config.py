"""Config loading: exact numbers and clear rejection of bad input.

Owner: Jainam Parekh Naresh, Reviewer: Vaibhav Malik
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from resource_war.config import (
    ConfigError,
    load_all,
    parse_yaml_text,
    to_fraction,
    validate_depth,
    validate_seed,
)


def test_default_configs_load_with_exact_numbers(configs) -> None:
    assert configs.payoffs.attack_cost == 2
    assert configs.payoffs.defend_cost == 1
    assert configs.payoffs.p_row == Fraction(1, 2)
    assert configs.agents.fixed_mixed == (Fraction(2, 5), Fraction(7, 20), Fraction(1, 4))
    assert configs.payoffs.resource_values == (2, 4, 6, 8)
    assert configs.battle.win_utility > configs.battle.max_abs_eval == 64


def _replace_in(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"test setup: '{old}' not found in {path.name}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize(
    ("file", "old", "new", "message"),
    [
        ("payoffs.yaml", "attack_cost: 2", "attack_cost: .nan", "finite number"),
        ("payoffs.yaml", "attack_cost: 2", "attack_cost: .inf", "finite number"),
        ("payoffs.yaml", "attack_cost: 2", "attack_cost: yes", "expected a number"),
        ("payoffs.yaml", "attack_cost: 2", "attack_cost: two", "expected a number"),
        ("payoffs.yaml", "attack_cost: 2", "attack_cost: -1", ">= 0"),
        ("payoffs.yaml", "  col: 0.5", "  col: 0.6", "<= 1"),
        ("payoffs.yaml", "  row: 0.5", "  row: 1.5", "[0, 1]"),
        ("payoffs.yaml", "actions: [A, D, E]", "actions: [A, E]", "actions must be exactly"),
        ("payoffs.yaml", "resource_values: [2, 4, 6, 8]", "resource_values: [2, 0]", "> 0"),
        ("payoffs.yaml", "attack_cost: 2", "attack_kost: 2", "missing required key 'attack_cost'"),
        ("payoffs.yaml", "schema_version: 1", "schema_version: 2", "schema_version must be 1"),
        ("game.yaml", "count: 12", "count: 11", "must be even"),
        ("game.yaml", "bases: [[0, 0], [7, 7]]", "bases: [[0, 0], [7, 6]]", "point-symmetric"),
        ("game.yaml", "bases: [[0, 0], [7, 7]]", "bases: [[0, 0], [9, 9]]", "<= 7"),
        ("game.yaml", "turn_limit: 40", "turn_limit: 2.5", "expected an integer"),
        ("game.yaml", "seed: 0", "seed: -3", ">= 0"),
        ("game.yaml", "placement: band", "placement: spiral", "placement must be one of"),
        ("battle.yaml", "hp_range: [2, 3]", "hp_range: [3, 2]", "low <= high"),
        ("battle.yaml", "win_utility: 1000", "win_utility: 50", "must exceed"),
        ("battle.yaml", "start_positions: [Front, Back]", "start_positions: [Front, Middle]", "start_positions"),
        ("agents.yaml", "{A: 0.40, D: 0.35, E: 0.25}", "{A: 0.40, D: 0.35, E: 0.30}", "sum to exactly 1"),
        ("agents.yaml", "depths: [1, 3, 5]", "depths: [1, 3, 7]", "not in battle.yaml allowed_depths"),
        ("experiments.yaml", "confidence: 0.95", "confidence: 1.5", "(0, 1)"),
    ],
)
def test_bad_config_is_rejected_with_a_clear_message(config_dir: Path, file: str, old: str, new: str, message: str) -> None:
    _replace_in(config_dir / file, old, new)
    with pytest.raises(ConfigError) as err:
        load_all(config_dir)
    assert message in str(err.value)
    assert str(err.value).startswith(file), "the message must name the offending file first"


def test_missing_file_and_invalid_yaml(config_dir: Path) -> None:
    (config_dir / "battle.yaml").unlink()
    with pytest.raises(ConfigError, match="not found"):
        load_all(config_dir)
    (config_dir / "battle.yaml").write_text("units_per_side: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_all(config_dir)
    (config_dir / "battle.yaml").write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_all(config_dir)


def test_uploaded_yaml_text_errors_are_friendly() -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        parse_yaml_text("a: [1, 2", "upload.yaml")
    with pytest.raises(ConfigError, match="schema_version"):
        parse_yaml_text("a: 1", "upload.yaml")


def test_decimal_strings_become_exact_fractions() -> None:
    assert to_fraction(0.35, "x") == Fraction(7, 20)
    assert to_fraction("3/4", "x") == Fraction(3, 4)
    with pytest.raises(ConfigError):
        to_fraction(float("nan"), "x")
    with pytest.raises(ConfigError):
        to_fraction(True, "x")


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, "abc", None, True])
def test_seed_validation(bad) -> None:
    with pytest.raises(ConfigError):
        validate_seed(bad)


def test_seed_and_depth_accept_valid_values() -> None:
    assert validate_seed(0) == 0
    assert validate_seed("42") == 42
    assert validate_depth(5, (1, 3, 5)) == 5
    for bad in (0, 2, 7, "x"):
        with pytest.raises(ConfigError):
            validate_depth(bad, (1, 3, 5))
