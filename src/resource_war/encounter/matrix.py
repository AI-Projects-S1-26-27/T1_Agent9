"""Exact bimatrix games and the encounter payoff-matrix builder.

Owner: Atharv Singhal, Reviewer: Gursidak Singh Bassi

Why exact arithmetic:
- Dominance, best responses and Pareto comparisons all hinge on equality
  tests (is A's payoff >= B's in every column?). With floats, 0.1 + 0.2 != 0.3
  can flip a weak-dominance verdict. With ``Fraction`` every comparison is
  exact, so the analysis card never shows a rounding artefact as a result.

Conventions used throughout the project:
- ``row_payoffs[i][j]`` is the row player's payoff when row plays i and the
  column plays j; ``col_payoffs[i][j]`` is the column player's payoff for the
  same cell. Indices follow the action order A, D, E.
- An agent always sees the encounter from its own point of view, as the ROW
  player. ``swap_players`` turns the game round for the other agent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from resource_war.config import ConfigError, PayoffConfig, to_fraction

ATTACK, DEFEND, EXPAND = 0, 1, 2
ACTION_NAMES = ("A", "D", "E")
ACTION_LONG_NAMES = ("Attack", "Defend", "Expand")

Matrix = tuple[tuple[Fraction, ...], ...]


@dataclass(frozen=True)
class BimatrixGame:
    """A finite two-player normal-form game with exact payoffs."""

    row_payoffs: Matrix
    col_payoffs: Matrix
    row_labels: tuple[str, ...]
    col_labels: tuple[str, ...]
    name: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.row_payoffs), len(self.row_payoffs[0])

    def payoff(self, i: int, j: int) -> tuple[Fraction, Fraction]:
        return self.row_payoffs[i][j], self.col_payoffs[i][j]

    def swap_players(self) -> BimatrixGame:
        """The same game seen by the other player (who becomes the row player)."""
        m, n = self.shape
        row = tuple(tuple(self.col_payoffs[i][j] for i in range(m)) for j in range(n))
        col = tuple(tuple(self.row_payoffs[i][j] for i in range(m)) for j in range(n))
        return BimatrixGame(row, col, self.col_labels, self.row_labels, self.name)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Float copies for nashpy; never used for our own exact analysis."""
        return (
            np.array([[float(v) for v in r] for r in self.row_payoffs]),
            np.array([[float(v) for v in r] for r in self.col_payoffs]),
        )

    def expected_payoffs(self, x: Sequence[Fraction], y: Sequence[Fraction]) -> tuple[Fraction, Fraction]:
        """Expected payoffs (row, column) when row mixes x and column mixes y."""
        m, n = self.shape
        u = sum((x[i] * self.row_payoffs[i][j] * y[j] for i in range(m) for j in range(n)), Fraction(0))
        v = sum((x[i] * self.col_payoffs[i][j] * y[j] for i in range(m) for j in range(n)), Fraction(0))
        return u, v


def make_game(
    row_payoffs: Sequence[Sequence[Any]],
    col_payoffs: Sequence[Sequence[Any]],
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    name: str = "custom",
) -> BimatrixGame:
    """Build a validated game from user-supplied lists (custom matrices in the demo).

    Rejects ragged or empty matrices, mismatched shapes, and non-numeric or
    non-finite entries with a ConfigError naming the offending cell.
    """

    def convert(matrix: Sequence[Sequence[Any]], who: str) -> Matrix:
        if not isinstance(matrix, (list, tuple)) or not matrix:
            raise ConfigError(f"{name}: {who} payoff matrix must be a non-empty list of rows")
        width = None
        rows = []
        for i, row in enumerate(matrix):
            if not isinstance(row, (list, tuple)) or not row:
                raise ConfigError(f"{name}: {who} payoff row {i} must be a non-empty list")
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ConfigError(f"{name}: {who} payoff matrix is ragged (row {i} has {len(row)} entries, expected {width})")
            rows.append(tuple(to_fraction(v, f"{name}: {who} payoff [{i}][{j}]") for j, v in enumerate(row)))
        return tuple(rows)

    a = convert(row_payoffs, "row")
    b = convert(col_payoffs, "column")
    if (len(a), len(a[0])) != (len(b), len(b[0])):
        raise ConfigError(f"{name}: row and column payoff matrices must have the same shape, got {len(a)}x{len(a[0])} and {len(b)}x{len(b[0])}")
    m, n = len(a), len(a[0])
    rl = tuple(row_labels) if row_labels else tuple(f"R{i + 1}" for i in range(m))
    cl = tuple(col_labels) if col_labels else tuple(f"C{j + 1}" for j in range(n))
    if len(rl) != m or len(cl) != n:
        raise ConfigError(f"{name}: need {m} row labels and {n} column labels")
    return BimatrixGame(a, b, rl, cl, name)


def encounter_row_matrix(value: Fraction, c: Fraction, d: Fraction, p_win: Fraction) -> Matrix:
    """Row player's payoffs (DESIGN 2.2). ``p_win`` is the row player's battle-win probability.

    Rows/columns: A, D, E.
        A: [p_win*V - c,  -c,   V - c]
        D: [V - d,        -d,   -d   ]
        E: [0,            V,    V/2  ]
    """
    v = Fraction(value)
    return (
        (p_win * v - c, -c, v - c),
        (v - d, -d, -d),
        (Fraction(0), v, v / 2),
    )


def build_encounter_game(
    value: Any,
    payoffs: PayoffConfig,
    p_row: Fraction | None = None,
    p_col: Fraction | None = None,
) -> BimatrixGame:
    """Build the encounter bimatrix for a cell worth ``value``.

    The column player's matrix is the row structure seen from the column's
    side, with the column's OWN battle-win probability in the A-vs-A cell.
    Because battles can tie, p_col is not 1 - p_row (DESIGN 2.2, decision 1).
    """
    v = to_fraction(value, "cell value")
    if v <= 0:
        raise ConfigError(f"cell value must be > 0, got {v}")
    pr = payoffs.p_row if p_row is None else Fraction(p_row)
    pc = payoffs.p_col if p_col is None else Fraction(p_col)
    if not (0 <= pr <= 1 and 0 <= pc <= 1 and pr + pc <= 1):
        raise ConfigError(f"battle win probabilities must lie in [0, 1] and sum to <= 1, got {pr} and {pc}")
    c, d = payoffs.attack_cost, payoffs.defend_cost
    row = encounter_row_matrix(v, c, d, pr)
    col_struct = encounter_row_matrix(v, c, d, pc)
    # Column payoff when row plays i and column plays j = the column's own
    # structure evaluated at (its action j, opponent action i).
    col = tuple(tuple(col_struct[j][i] for j in range(3)) for i in range(3))
    return BimatrixGame(row, col, ACTION_NAMES, ACTION_NAMES, f"encounter V={v}")


def realized_gains_and_costs(
    row_action: int, col_action: int, value: Fraction, payoffs: PayoffConfig, battle_winner: int | None = None
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    """What each player actually gains (share of V) and pays in one encounter.

    This is the engine's view of an encounter, written as "who obtains the
    cell" plus "who pays what". The engine asserts that gain - cost equals the
    matrix entry for every non-battle pair, which is the payoff-to-score
    invariant (DESIGN 3.3). For Attack-vs-Attack, ``battle_winner`` is 0 (row),
    1 (column) or None (tie).
    """
    v = Fraction(value)
    cost_of = {ATTACK: payoffs.attack_cost, DEFEND: payoffs.defend_cost, EXPAND: Fraction(0)}
    costs = (cost_of[row_action], cost_of[col_action])
    zero = Fraction(0)
    pair = (row_action, col_action)
    if pair == (ATTACK, ATTACK):
        gains = (v, zero) if battle_winner == 0 else (zero, v) if battle_winner == 1 else (zero, zero)
    elif pair == (EXPAND, EXPAND):
        gains = (v / 2, v / 2)
    elif pair == (DEFEND, DEFEND):
        gains = (zero, zero)
    elif ATTACK in pair and DEFEND in pair:
        gains = (v, zero) if row_action == DEFEND else (zero, v)
    elif ATTACK in pair and EXPAND in pair:
        gains = (v, zero) if row_action == ATTACK else (zero, v)
    else:  # DEFEND vs EXPAND: the builder takes the cell
        gains = (v, zero) if row_action == EXPAND else (zero, v)
    return gains, costs
