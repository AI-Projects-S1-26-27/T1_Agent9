"""The "Game Analysis" card: everything the course concepts say about one encounter.

Owner: Atharv Singhal, Reviewer: Gursidak Singh Bassi

For a bimatrix game the card reports, in order:
1. the payoff matrix;
2. strictly and weakly dominated strategies, strategies dominated by a
   mixture, and the steps of iterated elimination (strict, and weak with
   its caveat);
3. pure Nash equilibria;
4. mixed Nash equilibria (exact support enumeration, or vertex enumeration
   for degenerate games);
5. the equilibrium each Equilibrium agent would select, and whether the
   resulting play is itself an equilibrium (mis-coordination check);
6. Pareto-optimal outcomes, and whether each equilibrium is Pareto-dominated.

The same object renders as plain text (CLI, logs) and as a dict (Streamlit).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any

from resource_war.encounter.dominance import (
    Domination,
    EliminationResult,
    dominant_strategy,
    dominated_by_mixed,
    dominated_strategies,
    iterated_elimination,
)
from resource_war.encounter.matrix import BimatrixGame
from resource_war.encounter.pareto import pareto_dominating_outcomes, pareto_optimal_outcomes
from resource_war.encounter.pure_nash import pure_nash_equilibria
from resource_war.equilibrium.selection import select_equilibrium
from resource_war.equilibrium.solve import EquilibriumSet, solve_game
from resource_war.equilibrium.support_enum import Equilibrium
from resource_war.equilibrium.verify import is_nash


def fmt(value: Fraction, decimals: bool = True) -> str:
    """'17/9 (~1.889)' for non-integers, '3' for integers."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value} (~{float(value):.3f})" if decimals else str(value)


def fmt_mix(mix: tuple[Fraction, ...], labels: tuple[str, ...]) -> str:
    parts = [f"{lab}={fmt(p, False)}" for lab, p in zip(labels, mix) if p > 0]
    approx = ", ".join(f"{float(p):.3f}" for p in mix)
    return f"[{', '.join(parts)}] (~[{approx}])"


@dataclass(frozen=True)
class GameAnalysis:
    game: BimatrixGame
    strict_dominated: tuple[tuple[Domination, ...], tuple[Domination, ...]]
    weak_dominated: tuple[tuple[Domination, ...], tuple[Domination, ...]]
    mixed_dominated: tuple[tuple[int, ...], tuple[int, ...]]
    dominant: tuple[int | None, int | None]
    strict_elimination: EliminationResult
    weak_elimination: EliminationResult
    pure_ne: tuple[tuple[int, int], ...]
    equilibria: EquilibriumSet
    selected_row: Equilibrium
    selected_col_strategy: tuple[Fraction, ...]
    played_is_equilibrium: bool
    pareto_optimal: tuple[tuple[int, int], ...]
    pareto_dominated_equilibria: tuple[tuple[Equilibrium, tuple[tuple[int, int], ...]], ...]

    # ------------------------------------------------------------ rendering
    def _cell(self, i: int, j: int) -> str:
        return f"{self.game.row_labels[i]},{self.game.col_labels[j]}"

    def _pair(self, i: int, j: int) -> str:
        u, v = self.game.payoff(i, j)
        return f"({self._cell(i, j)}) = ({fmt(u, False)}, {fmt(v, False)})"

    def _dom_text(self, items: tuple[Domination, ...], player: int) -> str:
        labels = self.game.row_labels if player == 0 else self.game.col_labels
        if not items:
            return "none"
        return ", ".join(f"{labels[d.dominated]} (by {labels[d.dominator]})" for d in items)

    def _elim_text(self, result: EliminationResult) -> str:
        if not result.steps:
            return "nothing can be removed"
        parts = []
        for d in result.steps:
            labels = self.game.row_labels if d.player == 0 else self.game.col_labels
            who = "row" if d.player == 0 else "column"
            parts.append(f"{who} drops {labels[d.dominated]} (dominated by {labels[d.dominator]})")
        rows = ",".join(self.game.row_labels[i] for i in result.remaining_rows)
        cols = ",".join(self.game.col_labels[j] for j in result.remaining_cols)
        return "; ".join(parts) + f"  ->  remaining row {{{rows}}}, column {{{cols}}}"

    def to_text(self) -> str:
        g = self.game
        m, n = g.shape
        width = max(14, max(len(f"({fmt(u, False)}, {fmt(v, False)})") for r in range(m) for c in range(n) for u, v in [g.payoff(r, c)]) + 2)
        lines = [f"GAME ANALYSIS: {g.name}", "Payoffs (row player, column player):"]
        lines.append("      " + "".join(f"{lab:^{width}}" for lab in g.col_labels))
        for i in range(m):
            cells = "".join(f"{'(' + fmt(g.row_payoffs[i][j], False) + ', ' + fmt(g.col_payoffs[i][j], False) + ')':^{width}}" for j in range(n))
            lines.append(f"  {g.row_labels[i]:>3} " + cells)
        lines.append("Dominance (pure strategies):")
        for p, who in ((0, "Row"), (1, "Column")):
            labels = g.row_labels if p == 0 else g.col_labels
            mixed = ", ".join(labels[k] for k in self.mixed_dominated[p]) or "none"
            dom = self.dominant[p]
            lines.append(
                f"  {who}: strictly dominated: {self._dom_text(self.strict_dominated[p], p)}; "
                f"weakly dominated: {self._dom_text(self.weak_dominated[p], p)}; "
                f"dominated by a mixture: {mixed}; dominant strategy: {labels[dom] if dom is not None else 'none'}"
            )
        lines.append(f"  Iterated strict elimination: {self._elim_text(self.strict_elimination)}")
        lines.append(f"  Iterated weak elimination (order-dependent, may remove equilibria): {self._elim_text(self.weak_elimination)}")
        lines.append("Pure Nash equilibria: " + (", ".join(self._pair(i, j) for i, j in self.pure_ne) or "none"))
        lines.append(f"Nash equilibria ({self.equilibria.method}, exact):")
        for e in self.equilibria.equilibria:
            kind = "pure" if e.is_pure else "mixed"
            lines.append(
                f"  [{kind}] row {fmt_mix(e.row, g.row_labels)}, column {fmt_mix(e.col, g.col_labels)}, "
                f"payoffs ({fmt(e.row_payoff)}, {fmt(e.col_payoff)})"
            )
        for note in self.equilibria.notes():
            lines.append(f"  note: {note}")
        lines.append(
            "Selected by the Equilibrium agents (max own payoff, then lexicographic): "
            f"row plays {fmt_mix(self.selected_row.row, g.row_labels)}, "
            f"column plays {fmt_mix(self.selected_col_strategy, g.col_labels)}"
            + ("" if self.played_is_equilibrium else "  WARNING: this combination is NOT an equilibrium (mis-coordination)")
        )
        lines.append("Pareto-optimal outcomes: " + ", ".join(self._pair(i, j) for i, j in self.pareto_optimal))
        if self.pareto_dominated_equilibria:
            for e, cells in self.pareto_dominated_equilibria:
                lines.append(
                    f"Inefficiency: equilibrium payoffs ({fmt(e.row_payoff)}, {fmt(e.col_payoff)}) are "
                    f"Pareto-dominated by " + ", ".join(self._pair(i, j) for i, j in cells)
                )
        else:
            lines.append("Efficiency: no equilibrium is Pareto-dominated by a pure outcome.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        g = self.game
        return {
            "name": g.name,
            "row_labels": list(g.row_labels),
            "col_labels": list(g.col_labels),
            "row_payoffs": [[str(v) for v in r] for r in g.row_payoffs],
            "col_payoffs": [[str(v) for v in r] for r in g.col_payoffs],
            "strict_dominated": [[(g.row_labels if p == 0 else g.col_labels)[d.dominated] for d in self.strict_dominated[p]] for p in (0, 1)],
            "weak_dominated": [[(g.row_labels if p == 0 else g.col_labels)[d.dominated] for d in self.weak_dominated[p]] for p in (0, 1)],
            "pure_ne": [[g.row_labels[i], g.col_labels[j]] for i, j in self.pure_ne],
            "equilibria": [
                {"row": [str(v) for v in e.row], "col": [str(v) for v in e.col], "payoffs": [str(e.row_payoff), str(e.col_payoff)]}
                for e in self.equilibria.equilibria
            ],
            "degenerate": self.equilibria.degenerate,
            "pareto_optimal": [[g.row_labels[i], g.col_labels[j]] for i, j in self.pareto_optimal],
            "equilibrium_pareto_dominated": bool(self.pareto_dominated_equilibria),
        }


@lru_cache(maxsize=1024)
def analyze(game: BimatrixGame) -> GameAnalysis:
    eq_set = solve_game(game)
    selected_row = select_equilibrium(eq_set.equilibria)
    selected_col = select_equilibrium(solve_game(game.swap_players()).equilibria)
    col_strategy = selected_col.row  # the column player's own strategy, seen as the row of the swapped game
    dominated_eqs = []
    for e in eq_set.equilibria:
        cells = tuple(pareto_dominating_outcomes(game, (e.row_payoff, e.col_payoff)))
        if cells:
            dominated_eqs.append((e, cells))
    return GameAnalysis(
        game=game,
        strict_dominated=(tuple(dominated_strategies(game, 0, True)), tuple(dominated_strategies(game, 1, True))),
        weak_dominated=(tuple(dominated_strategies(game, 0, False)), tuple(dominated_strategies(game, 1, False))),
        mixed_dominated=(tuple(dominated_by_mixed(game, 0)), tuple(dominated_by_mixed(game, 1))),
        dominant=(dominant_strategy(game, 0), dominant_strategy(game, 1)),
        strict_elimination=iterated_elimination(game, strict=True),
        weak_elimination=iterated_elimination(game, strict=False),
        pure_ne=tuple(pure_nash_equilibria(game)),
        equilibria=eq_set,
        selected_row=selected_row,
        selected_col_strategy=col_strategy,
        played_is_equilibrium=is_nash(game, selected_row.row, col_strategy),
        pareto_optimal=tuple(pareto_optimal_outcomes(game)),
        pareto_dominated_equilibria=tuple(dominated_eqs),
    )

# Verified: Atharv (Game Analysis card)
