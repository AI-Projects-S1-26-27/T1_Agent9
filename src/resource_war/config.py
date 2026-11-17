"""Load and validate the YAML configuration files.

Owner: Jainam Parekh Naresh, Reviewer: Vaibhav Malik

Why this module exists:
- Every analysis, game and experiment reads its numbers from config/*.yaml.
  Bad input must be rejected here, with a message naming the file and the
  key, so the engine and the demo never see a malformed value or crash with
  a stack trace.
- Payoff numbers become exact ``Fraction`` objects via ``Fraction(str(x))``.
  YAML parses ``0.35`` as the float 0.35000000000000003; going through
  ``str`` recovers the decimal the human wrote, so 0.35 means exactly 7/20
  and equilibria computed from it are exact.
- Booleans are rejected where numbers are expected: in Python ``True == 1``,
  so a typo such as ``attack_cost: yes`` would otherwise silently become 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
MAX_SEED = 2**32 - 1
ACTIONS = ("A", "D", "E")
POSITIONS = ("Front", "Back")
PLACEMENTS = ("point_symmetric", "band")
TARGETING = ("nearest", "value_per_distance")
TIE_BREAKS = ("absolute", "player_relative")


class ConfigError(ValueError):
    """Raised for any invalid configuration; the message is shown to users as-is."""


# --------------------------------------------------------------------------
# Primitive validators. Each takes a ``ctx`` string ("file: key") so that the
# error message tells the user exactly where the problem is.
# --------------------------------------------------------------------------


def to_fraction(value: Any, ctx: str) -> Fraction:
    """Convert a YAML scalar to an exact Fraction, rejecting bools, NaN and inf."""
    if isinstance(value, bool) or value is None:
        raise ConfigError(f"{ctx}: expected a number, got {value!r}")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{ctx}: expected a finite number, got {value!r}")
        return Fraction(str(value))
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = Fraction(text)
        except (ValueError, ZeroDivisionError):
            raise ConfigError(f"{ctx}: expected a number, got {value!r}") from None
        return parsed
    raise ConfigError(f"{ctx}: expected a number, got {type(value).__name__} {value!r}")


def to_int(value: Any, ctx: str, lo: int | None = None, hi: int | None = None) -> int:
    """Convert to int, rejecting bools and non-integral numbers, and check the range."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ConfigError(f"{ctx}: expected an integer, got {value!r}")
    try:
        frac = to_fraction(value, ctx)
    except ConfigError:
        raise ConfigError(f"{ctx}: expected an integer, got {value!r}") from None
    if frac.denominator != 1:
        raise ConfigError(f"{ctx}: expected an integer, got {value!r}")
    result = int(frac)
    if lo is not None and result < lo:
        raise ConfigError(f"{ctx}: must be >= {lo}, got {result}")
    if hi is not None and result > hi:
        raise ConfigError(f"{ctx}: must be <= {hi}, got {result}")
    return result


def validate_seed(seed: Any, ctx: str = "seed") -> int:
    """Seeds are integers in [0, 2**32 - 1] so they are easy to type and replay."""
    return to_int(seed, ctx, 0, MAX_SEED)


def validate_depth(depth: Any, allowed: tuple[int, ...], ctx: str = "depth") -> int:
    """Battle depth must be one of the configured depths (1, 3 or 5 by default)."""
    value = to_int(depth, ctx, 1)
    if value not in allowed:
        raise ConfigError(f"{ctx}: must be one of {list(allowed)}, got {value}")
    return value


def _require(data: Mapping[str, Any], key: str, ctx: str) -> Any:
    if key not in data:
        raise ConfigError(f"{ctx}: missing required key '{key}'")
    return data[key]


def _mapping(value: Any, ctx: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{ctx}: expected a mapping, got {type(value).__name__}")
    return value


def _list(value: Any, ctx: str, min_len: int = 1) -> list[Any]:
    if not isinstance(value, list) or len(value) < min_len:
        raise ConfigError(f"{ctx}: expected a list with at least {min_len} item(s), got {value!r}")
    return value


def read_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML mapping with schema_version 1, with friendly errors."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"{path.name}: config file not found at {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: not valid YAML ({exc.__class__.__name__}: {exc})") from None
    return check_schema(data, path.name)


def parse_yaml_text(text: str, source: str) -> dict[str, Any]:
    """Parse YAML text (e.g. an uploaded file in the demo) with friendly errors."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: not valid YAML ({exc.__class__.__name__})") from None
    return check_schema(data, source)


def check_schema(data: Any, source: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: top level must be a mapping of keys to values")
    if data.get("schema_version") != 1:
        raise ConfigError(f"{source}: schema_version must be 1, got {data.get('schema_version')!r}")
    return data


# --------------------------------------------------------------------------
# Payoffs (encounter game)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PayoffConfig:
    """Parameters of the encounter game; see docs/DESIGN.md section 2.2."""

    attack_cost: Fraction
    defend_cost: Fraction
    p_row: Fraction
    p_col: Fraction
    resource_values: tuple[Fraction, ...]
    actions: tuple[str, ...] = ACTIONS

    def with_overrides(self, **changes: Any) -> PayoffConfig:
        """Return a validated copy with some fields replaced (used by sweeps and sliders)."""
        fields = {
            "attack_cost": self.attack_cost,
            "defend_cost": self.defend_cost,
            "p_row": self.p_row,
            "p_col": self.p_col,
            "resource_values": self.resource_values,
        }
        fields.update(changes)
        return make_payoff_config(**fields)


def make_payoff_config(
    attack_cost: Any,
    defend_cost: Any,
    p_row: Any,
    p_col: Any,
    resource_values: Any,
    source: str = "payoffs",
) -> PayoffConfig:
    c = to_fraction(attack_cost, f"{source}: attack_cost")
    d = to_fraction(defend_cost, f"{source}: defend_cost")
    if c < 0 or d < 0:
        raise ConfigError(f"{source}: attack_cost and defend_cost must be >= 0")
    pr = to_fraction(p_row, f"{source}: battle_win_prob.row")
    pc = to_fraction(p_col, f"{source}: battle_win_prob.col")
    for name, p in (("row", pr), ("col", pc)):
        if not 0 <= p <= 1:
            raise ConfigError(f"{source}: battle_win_prob.{name} must be in [0, 1], got {p}")
    if pr + pc > 1:
        raise ConfigError(
            f"{source}: battle_win_prob.row + battle_win_prob.col must be <= 1 "
            f"(the remainder is the tie probability), got {pr + pc}"
        )
    values = tuple(
        to_fraction(v, f"{source}: resource_values[{i}]")
        for i, v in enumerate(_list(list(resource_values), f"{source}: resource_values"))
    )
    if any(v <= 0 for v in values):
        raise ConfigError(f"{source}: every resource value must be > 0")
    return PayoffConfig(c, d, pr, pc, values)


def parse_payoffs(data: Mapping[str, Any], source: str = "payoffs.yaml") -> PayoffConfig:
    actions = _require(data, "actions", source)
    if not isinstance(actions, list) or actions != list(ACTIONS):
        raise ConfigError(f"{source}: actions must be exactly {list(ACTIONS)}, got {actions!r}")
    probs = _mapping(_require(data, "battle_win_prob", source), f"{source}: battle_win_prob")
    return make_payoff_config(
        _require(data, "attack_cost", source),
        _require(data, "defend_cost", source),
        _require(probs, "row", f"{source}: battle_win_prob"),
        _require(probs, "col", f"{source}: battle_win_prob"),
        _list(_require(data, "resource_values", source), f"{source}: resource_values"),
        source,
    )


# --------------------------------------------------------------------------
# Main game
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GameConfig:
    """Main-game environment; see docs/DESIGN.md section 3."""

    board_size: int
    bases: tuple[tuple[int, int], tuple[int, int]]
    units_per_player: int
    resource_count: int
    resource_values: tuple[Fraction, ...]
    placement: str
    band_width: int
    regrow_turns: int | None
    targeting: str
    tie_break: str
    turn_limit: int
    territory_weight: Fraction
    seed: int
    num_players: int = 2


def parse_game(data: Mapping[str, Any], source: str = "game.yaml") -> GameConfig:
    size = to_int(_require(data, "board_size", source), f"{source}: board_size", 4, 26)
    players = to_int(_require(data, "num_players", source), f"{source}: num_players")
    if players != 2:
        raise ConfigError(f"{source}: num_players must be 2 (the encounter game is two-player)")
    raw_bases = _list(_require(data, "bases", source), f"{source}: bases", 2)
    if len(raw_bases) != 2:
        raise ConfigError(f"{source}: bases must list exactly 2 cells")
    bases = []
    for i, b in enumerate(raw_bases):
        if not isinstance(b, list) or len(b) != 2:
            raise ConfigError(f"{source}: bases[{i}] must be [row, col], got {b!r}")
        r = to_int(b[0], f"{source}: bases[{i}][0]", 0, size - 1)
        c = to_int(b[1], f"{source}: bases[{i}][1]", 0, size - 1)
        bases.append((r, c))
    if bases[1] != (size - 1 - bases[0][0], size - 1 - bases[0][1]):
        raise ConfigError(
            f"{source}: bases must be point-symmetric about the board centre for fairness, "
            f"got {bases}"
        )
    units = to_int(_require(data, "units_per_player", source), f"{source}: units_per_player", 1, 8)
    res = _mapping(_require(data, "resources", source), f"{source}: resources")
    count = to_int(_require(res, "count", f"{source}: resources"), f"{source}: resources.count", 2)
    if count % 2:
        raise ConfigError(f"{source}: resources.count must be even (resources come in mirror pairs)")
    if count > size * size // 2:
        raise ConfigError(f"{source}: resources.count {count} is too large for a {size}x{size} board")
    values = tuple(
        to_fraction(v, f"{source}: resources.values[{i}]")
        for i, v in enumerate(_list(_require(res, "values", f"{source}: resources"), f"{source}: resources.values"))
    )
    if any(v <= 0 for v in values):
        raise ConfigError(f"{source}: every resource value must be > 0")
    placement = _require(res, "placement", f"{source}: resources")
    if placement not in PLACEMENTS:
        raise ConfigError(f"{source}: resources.placement must be one of {list(PLACEMENTS)}, got {placement!r}")
    band = to_int(res.get("band_width", 2), f"{source}: resources.band_width", 0, size)
    regrow_raw = res.get("regrow_turns")
    regrow = None if regrow_raw is None else to_int(regrow_raw, f"{source}: resources.regrow_turns", 1)
    policy = _mapping(data.get("policy", {}), f"{source}: policy")
    targeting = policy.get("targeting", "nearest")
    if targeting not in TARGETING:
        raise ConfigError(f"{source}: policy.targeting must be one of {list(TARGETING)}, got {targeting!r}")
    tie_break = policy.get("tie_break", "player_relative")
    if tie_break not in TIE_BREAKS:
        raise ConfigError(f"{source}: policy.tie_break must be one of {list(TIE_BREAKS)}, got {tie_break!r}")
    turn_limit = to_int(_require(data, "turn_limit", source), f"{source}: turn_limit", 1, 1000)
    weight = to_fraction(_require(data, "territory_weight", source), f"{source}: territory_weight")
    if weight < 0:
        raise ConfigError(f"{source}: territory_weight must be >= 0")
    seed = validate_seed(_require(data, "seed", source), f"{source}: seed")
    return GameConfig(
        size, (bases[0], bases[1]), units, count, values, placement, band, regrow,
        targeting, tie_break, turn_limit, weight, seed,
    )


# --------------------------------------------------------------------------
# Battle Mode
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BattleConfig:
    """Battle Mode parameters; see docs/DESIGN.md section 4."""

    units_per_side: int
    hp_range: tuple[int, int]
    start_positions: tuple[str, ...]
    strike_damage: int
    ply_cap: int
    allowed_depths: tuple[int, ...]
    hp_weight: int
    front_weight: int
    win_utility: int
    depth5_gate_seconds: float

    @property
    def max_abs_eval(self) -> int:
        """Largest possible |evaluation|; win_utility must exceed it (DESIGN 4.5)."""
        max_hp = self.units_per_side * self.hp_range[1]
        return self.hp_weight * max_hp + self.front_weight * self.units_per_side


def parse_battle(data: Mapping[str, Any], source: str = "battle.yaml") -> BattleConfig:
    units = to_int(_require(data, "units_per_side", source), f"{source}: units_per_side", 1, 3)
    hp = _list(_require(data, "hp_range", source), f"{source}: hp_range", 2)
    lo = to_int(hp[0], f"{source}: hp_range[0]", 1, 9)
    hi = to_int(hp[1], f"{source}: hp_range[1]", 1, 9)
    if lo > hi:
        raise ConfigError(f"{source}: hp_range must be [low, high] with low <= high, got {hp}")
    positions = tuple(_list(_require(data, "start_positions", source), f"{source}: start_positions"))
    if any(p not in POSITIONS for p in positions) or len(set(positions)) != len(positions):
        raise ConfigError(f"{source}: start_positions must be distinct values from {list(POSITIONS)}")
    damage = to_int(_require(data, "strike_damage", source), f"{source}: strike_damage", 1, 9)
    cap = to_int(_require(data, "ply_cap", source), f"{source}: ply_cap", 1, 40)
    depths = tuple(
        to_int(d, f"{source}: allowed_depths[{i}]", 1, 8)
        for i, d in enumerate(_list(_require(data, "allowed_depths", source), f"{source}: allowed_depths"))
    )
    ev = _mapping(_require(data, "evaluation", source), f"{source}: evaluation")
    hp_w = to_int(_require(ev, "hp_weight", f"{source}: evaluation"), f"{source}: evaluation.hp_weight", 0)
    fr_w = to_int(_require(ev, "front_weight", f"{source}: evaluation"), f"{source}: evaluation.front_weight", 0)
    term = _mapping(_require(data, "terminal", source), f"{source}: terminal")
    win = to_int(_require(term, "win_utility", f"{source}: terminal"), f"{source}: terminal.win_utility", 1)
    gate = float(to_fraction(_require(data, "depth5_gate_seconds", source), f"{source}: depth5_gate_seconds"))
    if gate <= 0:
        raise ConfigError(f"{source}: depth5_gate_seconds must be > 0")
    cfg = BattleConfig(units, (lo, hi), positions, damage, cap, depths, hp_w, fr_w, win, gate)
    if cfg.win_utility <= cfg.max_abs_eval:
        raise ConfigError(
            f"{source}: terminal.win_utility ({win}) must exceed the largest possible "
            f"|evaluation| ({cfg.max_abs_eval}) so a proven win always beats a heuristic score"
        )
    return cfg


# --------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentsConfig:
    """Agent parameters; see docs/DESIGN.md section 6."""

    heuristic_threshold: Fraction
    fixed_mixed: tuple[Fraction, Fraction, Fraction]
    selection: str
    dirichlet_prior: tuple[Fraction, Fraction, Fraction]
    counts_per_value: bool
    depths: tuple[int, ...]


def parse_agents(data: Mapping[str, Any], source: str = "agents.yaml") -> AgentsConfig:
    ph = _mapping(_require(data, "pure_heuristic", source), f"{source}: pure_heuristic")
    threshold = to_fraction(_require(ph, "threshold", f"{source}: pure_heuristic"), f"{source}: pure_heuristic.threshold")
    fm = _mapping(_require(data, "fixed_mixed", source), f"{source}: fixed_mixed")
    probs_raw = _mapping(_require(fm, "probabilities", f"{source}: fixed_mixed"), f"{source}: fixed_mixed.probabilities")
    if set(probs_raw) != set(ACTIONS):
        raise ConfigError(f"{source}: fixed_mixed.probabilities must have exactly the keys {list(ACTIONS)}")
    probs = tuple(to_fraction(probs_raw[a], f"{source}: fixed_mixed.probabilities.{a}") for a in ACTIONS)
    if any(p < 0 for p in probs) or sum(probs) != 1:
        raise ConfigError(f"{source}: fixed_mixed.probabilities must be >= 0 and sum to exactly 1, got sum {sum(probs)}")
    eq = _mapping(_require(data, "equilibrium", source), f"{source}: equilibrium")
    selection = _require(eq, "selection", f"{source}: equilibrium")
    if selection != "max_own_payoff_then_lexicographic":
        raise ConfigError(f"{source}: equilibrium.selection must be 'max_own_payoff_then_lexicographic'")
    eu = _mapping(_require(data, "expected_utility", source), f"{source}: expected_utility")
    prior_raw = _list(_require(eu, "dirichlet_prior", f"{source}: expected_utility"), f"{source}: expected_utility.dirichlet_prior", 3)
    if len(prior_raw) != 3:
        raise ConfigError(f"{source}: expected_utility.dirichlet_prior must have 3 entries (A, D, E)")
    prior = tuple(to_fraction(p, f"{source}: expected_utility.dirichlet_prior[{i}]") for i, p in enumerate(prior_raw))
    if any(p <= 0 for p in prior):
        raise ConfigError(f"{source}: expected_utility.dirichlet_prior entries must be > 0")
    per_value = eu.get("counts_per_value", True)
    if not isinstance(per_value, bool):
        raise ConfigError(f"{source}: expected_utility.counts_per_value must be true or false")
    if eu.get("tie_break", "lexicographic") != "lexicographic":
        raise ConfigError(f"{source}: expected_utility.tie_break must be 'lexicographic'")
    dl = _mapping(_require(data, "depth_limited", source), f"{source}: depth_limited")
    depths = tuple(
        to_int(d, f"{source}: depth_limited.depths[{i}]", 1, 8)
        for i, d in enumerate(_list(_require(dl, "depths", f"{source}: depth_limited"), f"{source}: depth_limited.depths"))
    )
    if dl.get("encounter_policy", "equilibrium") != "equilibrium":
        raise ConfigError(f"{source}: depth_limited.encounter_policy must be 'equilibrium'")
    return AgentsConfig(threshold, probs, selection, prior, per_value, depths)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentsConfig:
    """Experiment parameters. Section bodies are kept as plain mappings."""

    base_seed: int
    confidence: float
    bootstrap_resamples: int
    sections: Mapping[str, Mapping[str, Any]]

    def section(self, name: str) -> Mapping[str, Any]:
        if name not in self.sections:
            raise ConfigError(f"experiments.yaml: missing section '{name}'")
        return self.sections[name]


def parse_experiments(data: Mapping[str, Any], source: str = "experiments.yaml") -> ExperimentsConfig:
    seed = validate_seed(_require(data, "base_seed", source), f"{source}: base_seed")
    stats = _mapping(_require(data, "statistics", source), f"{source}: statistics")
    conf = float(to_fraction(_require(stats, "confidence", f"{source}: statistics"), f"{source}: statistics.confidence"))
    if not 0 < conf < 1:
        raise ConfigError(f"{source}: statistics.confidence must be in (0, 1)")
    resamples = to_int(_require(stats, "bootstrap_resamples", f"{source}: statistics"), f"{source}: statistics.bootstrap_resamples", 100)
    sections = {k: _mapping(v, f"{source}: {k}") for k, v in data.items() if k not in ("schema_version", "base_seed", "statistics")}
    return ExperimentsConfig(seed, conf, resamples, sections)


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Configs:
    payoffs: PayoffConfig
    game: GameConfig
    battle: BattleConfig
    agents: AgentsConfig
    experiments: ExperimentsConfig


def load_payoffs(path: Path | str | None = None) -> PayoffConfig:
    path = Path(path) if path else DEFAULT_CONFIG_DIR / "payoffs.yaml"
    return parse_payoffs(read_yaml(path), path.name)


def load_all(config_dir: Path | str | None = None) -> Configs:
    """Load and cross-validate every config file in ``config_dir``."""
    root = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    payoffs = parse_payoffs(read_yaml(root / "payoffs.yaml"))
    game = parse_game(read_yaml(root / "game.yaml"))
    battle = parse_battle(read_yaml(root / "battle.yaml"))
    agents = parse_agents(read_yaml(root / "agents.yaml"))
    experiments = parse_experiments(read_yaml(root / "experiments.yaml"))
    bad = [d for d in agents.depths if d not in battle.allowed_depths]
    if bad:
        raise ConfigError(
            f"agents.yaml: depth_limited.depths {bad} are not in battle.yaml allowed_depths "
            f"{list(battle.allowed_depths)}"
        )
    return Configs(payoffs, game, battle, agents, experiments)
