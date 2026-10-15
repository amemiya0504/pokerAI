"""Agent registry and the boundaries agents run behind (inline or own process)."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.agents import CallingStationAgent, Decision, RandomAgent, RuleBasedAgent
from backend.engine.actions import Action, LegalActions
from backend.engine.codec import (
    action_from_dict,
    action_to_dict,
    legal_from_dict,
    legal_to_dict,
    view_from_dict,
    view_to_dict,
)
from backend.engine.hand import PlayerView

def _pushfold(seed: int, **kw):
    from backend.agents.pushfold import PushFoldAgent  # loads datasets lazily

    return PushFoldAgent(seed, **kw)


REGISTRY: dict[str, Callable[..., Any]] = {
    "random": RandomAgent,
    "station": lambda seed, **kw: CallingStationAgent(),
    "rule": RuleBasedAgent,
    "pushfold": _pushfold,
}


@dataclass(frozen=True)
class AgentSpec:
    """A named agent in a run: ``kind`` from REGISTRY plus constructor params."""

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self, seed: int):
        if self.kind not in REGISTRY:
            raise ValueError(f"unknown agent kind {self.kind!r} (known: {sorted(REGISTRY)})")
        return REGISTRY[self.kind](seed, **self.params)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "params": self.params}


def _decision_to_wire(d: Decision) -> dict[str, Any]:
    return {
        "action": action_to_dict(d.action),
        "strategy_source": str(d.strategy_source),
        "action_probs": d.action_probs,
        "reason": d.reason,
    }


class InlineAgent:
    """Runs the agent in the worker process. It receives the engine's
    ``PlayerView``: an immutable snapshot that shares nothing mutable with
    the engine, so the agent cannot change the game, but it is not walled
    off from process memory (use ProcessAgent for that)."""

    def __init__(self, agent) -> None:
        self._agent = agent

    def act(self, view: PlayerView, legal: LegalActions) -> Action:
        return self._agent.act(view, legal).action

    def close(self) -> None:
        pass


def _agent_process(conn, spec: dict[str, Any]) -> None:
    agents: dict[int, Any] = {}
    s = AgentSpec(spec["name"], spec["kind"], spec["params"])
    while True:
        msg = conn.recv()
        if msg is None:
            return
        if msg["op"] == "reset":
            agents[msg["id"]] = s.build(msg["seed"])
            conn.send(True)
        else:
            d = agents[msg["id"]].act(view_from_dict(msg["view"]), legal_from_dict(msg["legal"]))
            conn.send(_decision_to_wire(d))


class ProcessAgent:
    """Runs the agent in its own process; only JSON-able data crosses the pipe.
    Slower, but the agent cannot reach engine memory at all."""

    def __init__(self, spec: AgentSpec) -> None:
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_agent_process, args=(child, spec.to_dict()), daemon=True)
        self._proc.start()
        self._id = 0

    def reset(self, seed: int) -> None:
        self._id += 1
        self._conn.send({"op": "reset", "id": self._id, "seed": seed})
        self._conn.recv()

    def act(self, view: PlayerView, legal: LegalActions) -> Action:
        self._conn.send({"op": "act", "id": self._id, "view": view_to_dict(view),
                         "legal": legal_to_dict(legal)})
        return action_from_dict(self._conn.recv()["action"])

    def close(self) -> None:
        self._conn.send(None)
        self._proc.join(5)

