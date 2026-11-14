"""Rendering helpers shared by the Streamlit app and the CLI fallback.

Owner: Rohit Gangil, Reviewer: Atharv Singhal

Both front ends call the same engine and the same helpers, so the CLI shows
exactly what the GUI shows. If the GUI fails during the live demo, nothing
is lost by switching to the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from resource_war.encounter.analysis import GameAnalysis
from resource_war.engine.state import GameState

AGENT_LABELS = {
    "pure_heuristic": "PureHeuristic (attack if V > 5)",
    "fixed_mixed": "FixedMixed (40/35/25)",
    "equilibrium": "Equilibrium (Nash mixed strategy)",
    "expected_utility": "ExpectedUtility (fictitious play)",
    "depth_limited": "DepthLimited (equilibrium + depth d)",
}


@dataclass(frozen=True)
class Cell:
    resource: Fraction | None
    units: tuple[int, int]
    owner: int | None
    base: int | None

    @property
    def contested(self) -> bool:
        return self.resource is not None and self.units[0] > 0 and self.units[1] > 0


def board_cells(state: GameState, bases: tuple[tuple[int, int], tuple[int, int]]) -> list[list[Cell]]:
    grid = []
    for r in range(state.size):
        row = []
        for c in range(state.size):
            units = (sum(1 for u in state.units if u.pos == (r, c) and u.player == 0), sum(1 for u in state.units if u.pos == (r, c) and u.player == 1))
            base = 0 if (r, c) == bases[0] else 1 if (r, c) == bases[1] else None
            row.append(Cell(state.resources.get((r, c)), units, state.owner.get((r, c)), base))
        grid.append(row)
    return grid


def cell_text(cell: Cell) -> str:
    """Short plain-text label: 'B0', '$6', '2a', '1a1b$8' ..."""
    parts = []
    if cell.base is not None:
        parts.append(f"B{cell.base}")
    if cell.units[0]:
        parts.append(f"{cell.units[0]}a")
    if cell.units[1]:
        parts.append(f"{cell.units[1]}b")
    if cell.resource is not None:
        parts.append(f"${cell.resource}")
    return "".join(parts) or "."


def board_text(state: GameState, bases) -> str:
    grid = board_cells(state, bases)
    lines = ["     " + "".join(f"{c:^8}" for c in range(state.size))]
    for r, row in enumerate(grid):
        marks = []
        for cell in row:
            text = cell_text(cell)
            if cell.owner is not None and cell.base is None:
                text = f"{text}{'+' if cell.owner == 0 else '#'}"
            marks.append(f"{text:^8}")
        lines.append(f"  {r}  " + "".join(marks))
    lines.append("legend: a = player 0 unit, b = player 1 unit, $V = resource worth V, + / # = owned by player 0 / 1, B = base")
    return "\n".join(lines)


def structure_banner(analysis: GameAnalysis) -> str:
    """One sentence that says what kind of game this encounter is (configuration explorer)."""
    a = analysis
    if a.strict_elimination.solved:
        i, j = a.strict_elimination.remaining_rows[0], a.strict_elimination.remaining_cols[0]
        return (
            f"Solved by iterated strict dominance: ({a.game.row_labels[i]},{a.game.col_labels[j]}) is the unique equilibrium."
            + (" Here that is peaceful expansion." if (a.game.row_labels[i], a.game.col_labels[j]) == ("E", "E") else "")
        )
    if a.equilibria.degenerate:
        weak = [a.game.row_labels[d.dominated] for d in a.weak_dominated[0]]
        extra = f" {', '.join(weak)} is only WEAKLY dominated." if weak else ""
        return f"Degenerate game: the equilibria form a continuum ({len(a.equilibria.equilibria)} extreme points listed).{extra}"
    eqs = a.equilibria.equilibria
    if not a.pure_ne and len(eqs) == 1:
        tail = " Both players would be better off at a Pareto-dominating outcome, but it is not an equilibrium." if a.pareto_dominated_equilibria else ""
        return "Cyclic game: no pure equilibrium, a unique mixed equilibrium." + tail
    return f"{len(a.pure_ne)} pure and {len(eqs) - len(a.pure_ne)} mixed equilibria."


def battle_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per ply: the move, root value, nodes for both searches, % saved and times."""
    timing = summary.get("timing", {}).get("per_ply", [])
    rows = []
    for k, d in enumerate(summary.get("decisions", [])):
        mm = d.get("minimax_nodes")
        ab = d["alphabeta_nodes"]
        ab_s, mm_s = (timing[k] if k < len(timing) else (None, None))
        rows.append(
            {
                "ply": k + 1,
                "side": f"player {d['side']}",
                "move": d["move"],
                "root value": d["value"],
                "minimax nodes": mm if mm is not None else "-",
                "alpha-beta nodes": ab,
                "pruned": d["alphabeta_pruned"],
                "% nodes saved": f"{100 * (1 - ab / mm):.0f}%" if mm else "-",
                "minimax ms": f"{1000 * mm_s:.1f}" if mm_s else "-",
                "alpha-beta ms": f"{1000 * ab_s:.1f}" if ab_s is not None else "-",
            }
        )
    return rows


def battle_outcome_text(summary: dict[str, Any]) -> str:
    w = summary["winner"]
    who = "tie (cell stays neutral, both pay c)" if w is None else f"player {w} wins the cell"
    return f"Battle: depths {summary['depths']}, first mover player {summary['first_mover']}, {summary['plies']} plies, final HP {summary['final_hp']}: {who}"
