"""Solver interface (design doc section 5).

``Solver.solve(SolveRequest) -> SolveResult``. A result always carries the
game it solved, the iterations, a convergence metric, the time taken and its
``strategy_source``. Requests are hashed so identical requests hit a cache.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from backend.agents.base import StrategySource


@dataclass(frozen=True)
class SolveRequest:
    game: str  # e.g. "pushfold"
    params: dict[str, Any]  # game definition: players, stack_bb, blinds, ...
    solver: dict[str, Any] = field(default_factory=dict)  # iterations, seed, ...

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class SolveResult:
    request: SolveRequest
    # infoset key -> {"actions": [...], "probs": [[p_a for a] per class], "ev": [[...]] | None}
    strategy: dict[str, dict[str, Any]]
    iterations: int
    # e.g. {"nash_conv_bb": 0.004, "method": "...", "per_player_bb": [...]}
    convergence: dict[str, Any]
    seconds: float
    strategy_source: StrategySource
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "request": asdict(self.request),
            "request_hash": self.request.key(),
            "strategy_source": str(self.strategy_source),
            "iterations": self.iterations,
            "convergence": self.convergence,
            "seconds": self.seconds,
            "notes": self.notes,
            "strategy": self.strategy,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> SolveResult:
        r = d["request"]
        return cls(
            request=SolveRequest(r["game"], r["params"], r["solver"]),
            strategy=d["strategy"],
            iterations=d["iterations"],
            convergence=d["convergence"],
            seconds=d["seconds"],
            strategy_source=StrategySource(d["strategy_source"]),
            notes=list(d.get("notes", [])),
        )


class Solver(Protocol):
    def solve(self, request: SolveRequest) -> SolveResult: ...
