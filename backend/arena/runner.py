"""Arena: AI-vs-AI duplicate simulation across processes.

Each deal (one deck seed) is played once per cyclic rotation of agents over
the seats, so every agent holds every seat's cards in every position. Stacks
reset to ``stack_bb`` big blinds every hand and the button is always seat 0.

Every random source derives from the master seed by (deal, rotation, agent),
so results do not depend on the number of workers or how deals are chunked.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Literal

from backend.engine.actions import ActionType
from backend.engine.cards import Deck, cards_str, derive_seed
from backend.engine.hand import HandState

from .agents import AgentSpec, InlineAgent, ProcessAgent
from .stats import AgentStats, summarize

Boundary = Literal["inline", "process"]

_SHORT = {ActionType.FOLD: "f", ActionType.CHECK: "x", ActionType.CALL: "c",
          ActionType.BET: "b", ActionType.RAISE: "r"}


@dataclass(frozen=True)
class ArenaConfig:
    agents: tuple[AgentSpec, ...]
    deals: int
    seed: int
    stack_bb: int = 100
    small_blind: int = 50
    big_blind: int = 100
    boundary: Boundary = "inline"

    def __post_init__(self) -> None:
        if not 2 <= len(self.agents) <= 9:
            raise ValueError("an arena needs 2 to 9 agents")
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError("agent names must be unique")
        if self.deals <= 0:
            raise ValueError("deals must be positive")
        if self.boundary not in ("inline", "process"):
            raise ValueError(f"unknown boundary {self.boundary!r}")

    @property
    def hands(self) -> int:
        return self.deals * len(self.agents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": [a.to_dict() for a in self.agents],
            "deals": self.deals,
            "seed": self.seed,
            "stack_bb": self.stack_bb,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "boundary": self.boundary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArenaConfig:
        agents = tuple(AgentSpec(a["name"], a["kind"], dict(a.get("params") or {})) for a in d["agents"])
        return cls(agents=agents, **{k: v for k, v in d.items() if k != "agents"})


# (deal, rotation, deck_seed, seating "name,name,...", actions "f,c,r300", board)
HandRow = tuple[int, int, int, str, str, str]
# (deal, rotation, seat, agent, position, net_chips, showdown)
PlayerRow = tuple[int, int, int, str, int, int, bool]


def play_deals(cfg_dict: dict[str, Any], start: int, stop: int) -> tuple[list[HandRow], list[PlayerRow]]:
    """Play deals [start, stop) with every rotation. Runs inside workers."""
    cfg = ArenaConfig.from_dict(cfg_dict)
    n = len(cfg.agents)
    stack = cfg.stack_bb * cfg.big_blind
    procs = [ProcessAgent(a) for a in cfg.agents] if cfg.boundary == "process" else None
    hand_rows: list[HandRow] = []
    player_rows: list[PlayerRow] = []
    try:
        for deal in range(start, stop):
            deck_seed = derive_seed(cfg.seed, "deal", deal)
            for rot in range(n):
                seat_agent = [(seat + rot) % n for seat in range(n)]
                seats = {}
                for seat, ai in enumerate(seat_agent):
                    agent_seed = derive_seed(cfg.seed, "agent", deal, rot, ai)
                    if procs is not None:
                        procs[ai].reset(agent_seed)
                        seats[seat] = procs[ai]
                    else:
                        seats[seat] = InlineAgent(cfg.agents[ai].build(agent_seed))
                state = HandState({s: stack for s in range(n)}, 0, cfg.small_blind, cfg.big_blind,
                                  Deck(deck_seed))
                while not state.is_over:
                    seat = state.to_act
                    state.apply(seats[seat].act(state.observation(seat), state.legal_actions()))
                result = state.result()
                actions = ",".join(
                    _SHORT[a.type] + (str(a.amount) if a.amount else "") for _, a in state.actions
                )
                hand_rows.append((deal, rot, deck_seed,
                                  ",".join(cfg.agents[ai].name for ai in seat_agent),
                                  actions, cards_str(result.board)))
                for seat, ai in enumerate(seat_agent):
                    player_rows.append((deal, rot, seat, cfg.agents[ai].name, seat,
                                        result.net[seat], seat in result.showdown))
    finally:
        for p in procs or ():
            p.close()
    return hand_rows, player_rows


@dataclass
class ArenaResult:
    config: ArenaConfig
    hand_rows: list[HandRow]
    player_rows: list[PlayerRow]
    stats: list[AgentStats]
    seconds: float
    workers: int

    @property
    def hands_per_second(self) -> float:
        return len(self.hand_rows) / self.seconds if self.seconds else float("inf")


def agent_stats(cfg: ArenaConfig, player_rows: Iterable[PlayerRow]) -> list[AgentStats]:
    n = len(cfg.agents)
    blocks: dict[str, dict[int, float]] = {a.name: {} for a in cfg.agents}
    for deal, _rot, _seat, agent, _pos, net, _sd in player_rows:
        b = blocks[agent]
        b[deal] = b.get(deal, 0.0) + net / cfg.big_blind
    return [
        summarize(name, [total / n for _, total in sorted(by_deal.items())], len(by_deal) * n)
        for name, by_deal in blocks.items()
    ]


def run_arena(cfg: ArenaConfig, workers: int | None = None, chunk: int | None = None,
              progress: Callable[[int, int], None] | None = None) -> ArenaResult:
    workers = max(1, workers or os.cpu_count() or 1)
    chunk = chunk or max(1, min(256, -(-cfg.deals // (workers * 4))))
    ranges = [(s, min(s + chunk, cfg.deals)) for s in range(0, cfg.deals, chunk)]
    t0 = time.perf_counter()
    hand_rows: list[HandRow] = []
    player_rows: list[PlayerRow] = []
    done = 0
    if workers == 1:
        for start, stop in ranges:
            h, p = play_deals(cfg.to_dict(), start, stop)
            hand_rows += h
            player_rows += p
            done += stop - start
            if progress:
                progress(done, cfg.deals)
    else:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(play_deals, cfg.to_dict(), s, e) for s, e in ranges]
            for (start, stop), fut in zip(ranges, futures):
                h, p = fut.result()  # in submission order: output is deterministic
                hand_rows += h
                player_rows += p
                done += stop - start
                if progress:
                    progress(done, cfg.deals)
    seconds = time.perf_counter() - t0
    return ArenaResult(cfg, hand_rows, player_rows, agent_stats(cfg, player_rows), seconds, workers)
