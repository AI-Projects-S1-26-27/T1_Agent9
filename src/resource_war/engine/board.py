"""Board geometry and point-symmetric resource placement.

Owner: Vaibhav Malik, Reviewer: Harshit Agrawal

Fairness by construction: the two bases are mirror images through the board
centre, and every resource cell is placed together with its mirror image,
carrying the same value. Neither player can be favoured by the map, so the
only systematic differences between players come from their agents.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from resource_war.config import GameConfig

Coord = tuple[int, int]
DIRECTIONS: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def mirror(cell: Coord, size: int) -> Coord:
    """Point reflection through the board centre: (r, c) -> (size-1-r, size-1-c)."""
    return size - 1 - cell[0], size - 1 - cell[1]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(cell: Coord, size: int) -> bool:
    return 0 <= cell[0] < size and 0 <= cell[1] < size


def candidate_cells(cfg: GameConfig) -> list[Coord]:
    """Cells eligible to hold the first member of a resource pair.

    Only the first half of the board in row-major order is used, so a cell and
    its mirror can never both be drawn. Bases (and cells whose mirror is a
    base) are excluded. ``band`` placement additionally restricts to cells
    near the anti-diagonal, where the two bases are (nearly) equidistant.
    """
    size = cfg.board_size
    bases = set(cfg.bases)
    cells = []
    for index in range(size * size // 2):
        cell = (index // size, index % size)
        if cell in bases or mirror(cell, size) in bases:
            continue
        if cfg.placement == "band" and abs(cell[0] + cell[1] - (size - 1)) > cfg.band_width:
            continue
        cells.append(cell)
    return cells


def place_resources(cfg: GameConfig, rng: np.random.Generator) -> dict[Coord, Fraction]:
    """Draw resource_count/2 cells and mirror them; each pair shares one value."""
    pairs = cfg.resource_count // 2
    cells = candidate_cells(cfg)
    if len(cells) < pairs:
        raise ValueError(f"only {len(cells)} candidate cells for {pairs} resource pairs; widen the band or reduce resources.count")
    chosen = rng.choice(len(cells), size=pairs, replace=False)
    resources: dict[Coord, Fraction] = {}
    for k in sorted(int(i) for i in chosen):
        value = cfg.resource_values[int(rng.integers(len(cfg.resource_values)))]
        cell = cells[k]
        resources[cell] = value
        resources[mirror(cell, cfg.board_size)] = value
    return dict(sorted(resources.items()))
