"""Search instrumentation: node counts, pruning counts and wall time.

Owner: Raunak Sarawgi, Reviewer: Harshit Agrawal

Counting conventions (DESIGN 4.6), identical for both searches so the
numbers are comparable:
- nodes: every call of the recursive search, root included;
- leaves: nodes scored by the terminal utility or the evaluation function;
- pruned: child moves skipped because of an alpha-beta cutoff (always 0
  for plain minimax);
- seconds: wall time of the whole decision (time.perf_counter).
Wall time depends on the machine; node counts do not, which is why H1 is
stated in nodes and time is reported alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from resource_war.battle.state import Move


@dataclass
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    pruned: int = 0
    seconds: float = 0.0


@dataclass
class SearchResult:
    algorithm: str
    depth: int
    move: Move
    value: int
    stats: SearchStats = field(default_factory=SearchStats)

    def summary(self) -> dict[str, Any]:
        """Deterministic part only (no wall time), safe for byte-identical game logs."""
        return {
            "algorithm": self.algorithm,
            "depth": self.depth,
            "move": self.move.describe(),
            "value": self.value,
            "nodes": self.stats.nodes,
            "leaves": self.stats.leaves,
            "pruned": self.stats.pruned,
        }
