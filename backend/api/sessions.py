"""Live Human-vs-AI games: one GameSession per game, shared by its websockets."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from backend.agents import CallingStationAgent, RandomAgent, RuleBasedAgent
from backend.db import Game
from backend.db.repository import players_by_id, record_statistics, save_hand
from backend.engine.actions import Action, ActionType, IllegalActionError
from backend.engine.cards import Deck, derive_seed
from backend.engine.hand import HandState
from backend.engine.records import hand_record
from backend.engine.table import Table

from .views import legal_json, view_json

log = logging.getLogger(__name__)

# Player statistics (M8) are updated off the game loop, one hand at a time:
# a single worker keeps the read-modify-write of the counts serial across all
# games, and a finished hand is broadcast without waiting for it.
_STATS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="player-stats")


def flush_statistics() -> None:
    """Wait until every queued statistics update is written (tests, shutdown)."""
    _STATS.submit(lambda: None).result()


def _record_statistics(sessions: sessionmaker, record: dict[str, Any]) -> None:
    try:
        with sessions() as db:
            record_statistics(db, record)
    except Exception:  # noqa: BLE001 - statistics must never break a game
        log.exception("updating player statistics failed for hand %s", record.get("hand_no"))


def _pushfold(seed: int):
    from backend.agents.pushfold import PushFoldAgent

    return PushFoldAgent(seed)


def _realtime(seed: int, opponent_models: dict | None = None):
    from backend.agents.web_realtime import WebRealtimeAgent

    # A8.3 (2026-10-27): defend capped ranges only vs raisers this session has seen iso / squeeze often
    return WebRealtimeAgent(seed, opponent_models=opponent_models, adaptive=True)


def _strategy_label(decision) -> str:
    source, reason = str(decision.strategy_source), decision.reason or ""
    return ("翻前表 · HEURISTIC" if reason.startswith("frozen preflop") else
            "push/fold · SOLVER APPROX" if reason.startswith("pushfold") else
            "HU 翻後 · SOLVER APPROX" if source == "solver_approx" else
            "HU 翻後 · NEURAL APPROX" if source == "neural_approx" else
            "HU 翻後 · HEURISTIC" if reason.startswith("HU HEURISTIC fallback") else
            "多人翻後 · HEURISTIC" if reason.startswith("HEURISTIC multiway") else
            source.upper())


AGENTS: dict[str, Callable[[int], Any]] = {
    "realtime": _realtime,
    "random": lambda seed: RandomAgent(seed, all_in_weight=0.05),
    "station": lambda seed: CallingStationAgent(),
    "rule": lambda seed: RuleBasedAgent(seed),
    "pushfold": _pushfold,
}


@dataclass(frozen=True)
class SeatInfo:
    seat: int
    player_id: int
    name: str
    kind: str  # human / ai
    agent: str | None
    start_stack: int


class GameSession:
    def __init__(self, game_id: int, game_seed: int, table: Table, seats: list[SeatInfo],
                 sessions: sessionmaker, ai_delay: float) -> None:
        self.game_id = game_id
        self.game_seed = game_seed
        self.table = table
        self.seats = {s.seat: s for s in seats}
        self.human_seat = next(s.seat for s in seats if s.kind == "human")
        self._sessions = sessions
        self.ai_delay = ai_delay
        self._lock = asyncio.Lock()
        self._listeners: set[asyncio.Queue] = set()
        self._decisions: list[dict[str, Any] | None] = []
        self._strategy_labels: dict[int, dict[str, str]] = {}
        self._agents: dict[int, Any] = {}
        self._driver: asyncio.Task | None = None
        # Opponent models (M8) of each real-time AI seat, kept across hands
        # (agents are rebuilt every hand).
        self._opponent_models: dict[int, dict] = {}
        self._replay_source: dict[str, Any] | None = None
        self._is_replay = False
        self._display_replay = False

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(cls, db: Session, game: Game, sessions: sessionmaker, ai_delay: float) -> GameSession:
        """Rebuild a live session from the database (also after a restart)."""
        names = players_by_id(db, [s["player_id"] for s in game.seating])
        table = Table(game.table.small_blind, game.table.big_blind, seed=game.seed,
                      max_seats=game.table.max_seats)
        seats = []
        stacks = {s["seat"]: s["start_stack"] for s in game.seating}
        if game.hands:
            last = game.hands[-1]
            for seat, stack in last.result["final_stacks"].items():
                stacks[int(seat)] = stack
            table.hand_no = last.hand_no
            table.button = last.button_seat
        for s in game.seating:
            p = names[s["player_id"]]
            agent = (p.agent_config or {}).get("agent")
            seats.append(SeatInfo(s["seat"], p.id, p.name, p.kind, agent, s["start_stack"]))
            table.sit(s["seat"], str(p.id), max(stacks[s["seat"]], 1))
            if stacks[s["seat"]] == 0:
                table.seats[s["seat"]].stack = 0
                table.seats[s["seat"]].sitting_out = True
        return cls(game.id, game.seed, table, seats, sessions, ai_delay)

    # ---------------------------------------------------------------- pub/sub

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners.discard(q)

    def _broadcast(self) -> None:
        snap = self.snapshot()
        for q in self._listeners:
            q.put_nowait(snap)

    # ---------------------------------------------------------------- views

    @property
    def status(self) -> str:
        if self.table.hand is not None:
            return "playing"
        return "hand_over" if self.table.last_hand is not None else "idle"

    def snapshot(self) -> dict[str, Any]:
        """The game as the human seat may see it."""
        state = self.table.hand or self.table.last_hand
        hand = None
        legal = None
        if state is not None:
            seat = self.human_seat if self.human_seat in state.seats else None
            revealed_cards = ({
                s: state.hole_cards(s)
                for s in state.seats
                if s != self.human_seat
            } if state.is_over else None)
            hand = view_json(state.observation(seat), revealed_cards=revealed_cards)
            if not state.is_over and state.to_act == self.human_seat:
                legal = legal_json(state.legal_actions())
        seats = []
        for seat, info in sorted(self.seats.items()):
            ts = self.table.seats[seat]
            seats.append({
                "seat": seat,
                "name": info.name,
                "kind": info.kind,
                "stack": ts.stack if ts else 0,
                "away": bool(ts and ts.away),
                "in_hand": state is not None and seat in state.seats,
                "strategy": self._strategy_labels.get(seat),
            })
        human = self.table.seats[self.human_seat]
        return {
            "type": "state",
            "game_id": self.game_id,
            "status": self.status,
            "hand_no": self.table.hand_no,
            "human_seat": self.human_seat,
            "small_blind": self.table.small_blind,
            "big_blind": self.table.big_blind,
            "ai_delay": self.ai_delay,
            "seats": seats,
            "hand": hand,
            "legal": legal,
            "can_rebuy": self.table.hand is None and human is not None
                         and human.stack < self.seats[self.human_seat].start_stack,
            "can_replay": self.table.hand is None and self._replay_source is not None,
            "is_replay": self._display_replay,
        }

    # ---------------------------------------------------------------- commands

    async def handle(self, msg: dict[str, Any]) -> str | None:
        """Apply one client command; returns an error message or None."""
        kind = msg.get("type")
        async with self._lock:
            if kind == "start_hand":
                err = self._start_hand()
            elif kind == "action":
                err = self._human_action(msg)
            elif kind == "replay_hand":
                err = self._replay_hand()
            elif kind == "rebuy":
                err = self._rebuy()
            elif kind == "set_delay":
                try:
                    self.ai_delay = min(max(float(msg.get("seconds", 0.6)), 0.0), 5.0)
                    err = None
                except (TypeError, ValueError):
                    err = "invalid delay"
            else:
                err = f"unknown command {kind!r}"
            if err is None:
                self._broadcast()
        if err is None:
            self._ensure_driver()
        return err

    def _start_hand(self) -> str | None:
        if self.table.hand is not None:
            return "a hand is already running"
        human = self.table.seats[self.human_seat]
        if human.stack == 0:
            return "out of chips: rebuy first"
        for seat, info in self.seats.items():
            ts = self.table.seats[seat]
            if info.kind == "ai" and ts.stack == 0:  # AI players rebuy automatically
                ts.stack = info.start_stack
                ts.sitting_out = False
        state = self.table.start_hand()
        hand_no = self.table.hand_no
        self._replay_source = {
            "start_stacks": state.starting_stacks(),
            "button": state.button,
            "deck_seed": state.deck_seed,
            "hand_no": hand_no,
            "opponent_models": copy.deepcopy(self._opponent_models),
        }
        self._agents = self._agents_for_hand(state, hand_no, self._opponent_models)
        self._is_replay = False
        self._display_replay = False
        self._decisions = []
        self._strategy_labels = {}
        return None

    def _agents_for_hand(self, state: HandState, hand_no: int, models: dict[int, dict]) -> dict[int, Any]:
        agents = {}
        for seat, info in self.seats.items():
            if info.kind != "ai" or seat not in state.seats:
                continue
            seed = derive_seed(self.game_seed, "agent", seat, hand_no)
            factory = AGENTS[info.agent]
            agents[seat] = factory(seed, models.setdefault(seat, {})) if info.agent == "realtime" else factory(seed)
        return agents

    def _replay_hand(self) -> str | None:
        source = self._replay_source
        if self.table.hand is not None:
            return "a hand is already running"
        if source is None:
            return "there is no hand to replay"
        state = HandState(source["start_stacks"], source["button"], self.table.small_blind,
                          self.table.big_blind, Deck(source["deck_seed"]))
        self.table.hand = state
        self._agents = self._agents_for_hand(
            state, source["hand_no"], copy.deepcopy(source["opponent_models"])
        )
        self._is_replay = True
        self._display_replay = True
        self._decisions = []
        self._strategy_labels = {}
        return None

    def _human_action(self, msg: dict[str, Any]) -> str | None:
        state = self.table.hand
        if state is None or state.to_act != self.human_seat:
            return "not your turn"
        try:
            action = Action(ActionType(msg.get("action")), int(msg.get("amount") or 0))
            state.apply(action)
        except (ValueError, IllegalActionError) as e:
            return str(e)
        self._decisions.append({"strategy_source": None, "reason": "human"})
        self._maybe_finish()
        return None

    def _rebuy(self) -> str | None:
        if self.table.hand is not None:
            return "cannot rebuy during a hand"
        info = self.seats[self.human_seat]
        ts = self.table.seats[self.human_seat]
        if ts.stack >= info.start_stack:
            return "stack is already full"
        self.table.add_chips(self.human_seat, info.start_stack - ts.stack)
        ts.sitting_out = False
        return None

    # ---------------------------------------------------------------- AI loop

    def _ensure_driver(self) -> None:
        if self._driver is None or self._driver.done():
            self._driver = asyncio.create_task(self._drive())

    async def _drive(self) -> None:
        """Play AI turns, pausing ai_delay before each, until the human must act."""
        while True:
            state = self.table.hand
            if state is None or state.to_act == self.human_seat:
                return
            await asyncio.sleep(self.ai_delay)
            async with self._lock:
                state = self.table.hand
                if state is None or state.to_act == self.human_seat:
                    return
                seat = state.to_act
                legal = state.legal_actions()
                # Off the event loop: a real-time solving AI may think for seconds.
                decision = await asyncio.to_thread(self._agents[seat].act, state.observation(seat), legal)
                label = _strategy_label(decision)
                # Reasons/probabilities can identify private cards; broadcast only the method.
                self._strategy_labels[seat] = {"label": label, "street": str(state.street)}
                state.apply(decision.action)
                self._decisions.append({
                    "strategy_source": str(decision.strategy_source),
                    "action_probs": decision.action_probs,
                    "reason": decision.reason,
                })
                self._maybe_finish()
                self._broadcast()

    def _maybe_finish(self) -> None:
        state = self.table.hand
        if state is None or not state.is_over:
            return
        if self._is_replay:
            self.table.last_hand, self.table.hand = state, None
            self._is_replay = False
            return
        player_ids = {s: self.seats[s].player_id for s in state.seats}
        record = hand_record(state, hand_no=self.table.hand_no, decisions=self._decisions,
                             player_ids=player_ids)
        with self._sessions() as db:
            save_hand(db, self.game_id, record)
        _STATS.submit(_record_statistics, self._sessions, record)
        for seat, agent in self._agents.items():
            if hasattr(agent, "hand_over"):
                agent.hand_over(state.observation(seat))
        self.table.finish_hand()
