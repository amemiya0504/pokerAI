"""AI decision API: the web AI's decision in any 2-9 player NLHE spot, and
hand recording that keeps per-player opponent models across hands.

A session holds the recorded hands of a group of players (by ``player_id``,
whatever seats they sit in). The opponent models (M8 statistics and A8.3's
iso / squeeze counts) are derived from those hands only: recording a hand
updates them, deleting one rebuilds them from the rest. Every decision uses
the models as they are at that moment, so recording hands as they finish
lets later decisions use the updated ranges.

A spot is (blinds, button, players with starting stacks, the AI's hole
cards, the board, the actions so far). Only cards somebody knows are given;
the engine replays the actions with the other cards filled in, which no
seat can see. Blinds only: no antes or straddles.

Decisions are serialised (one at a time) because heads-up postflop solves
on the GPU and parallel solves overflow its memory. Passing the same
``hand_id`` on every decision of a hand keeps the AI's in-hand memory
(its own range updates and safe re-solving), as at the web table; without
it every decision starts from a fresh agent.
"""

from __future__ import annotations

import json
import os
import random
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.engine.actions import Action, ActionType, IllegalActionError
from backend.engine.cards import Deck, cards_str, derive_seed, parse_cards
from backend.engine.hand import MAX_PLAYERS, HandState

from .views import legal_json

FORMAT = "poker-ai-session/1"
LIVE_HANDS = 64  # in-progress hands whose agents are kept (least recently used dropped first)
_SESSION_ID = re.compile(r"^[0-9a-f]{16}$")


# ----------------------------------------------------------------------------- requests


class PlayerIn(BaseModel):
    seat: int = Field(ge=0, lt=MAX_PLAYERS)
    player_id: str = Field(min_length=1, max_length=64)
    stack: int = Field(gt=0, description="stack at the start of the hand, before the blinds")


class ActionIn(BaseModel):
    seat: int | None = Field(None, description="checked against the seat to act when given")
    type: Literal["fold", "check", "call", "bet", "raise"]
    amount: int = Field(0, ge=0, description='for bet / raise: the "raise to" total on this street')


class HandIn(BaseModel):
    hand_id: str | None = Field(None, min_length=1, max_length=128)
    small_blind: int = Field(gt=0)
    big_blind: int = Field(gt=0)
    button: int = Field(ge=0, lt=MAX_PLAYERS)
    players: list[PlayerIn] = Field(min_length=2, max_length=MAX_PLAYERS)
    board: str = ""
    actions: list[ActionIn] = []

    @model_validator(mode="after")
    def _check(self) -> HandIn:
        if self.small_blind > self.big_blind:
            raise ValueError("small_blind must not exceed big_blind")
        seats = [p.seat for p in self.players]
        if len(set(seats)) != len(seats):
            raise ValueError("seats must be unique")
        ids = [p.player_id for p in self.players]
        if len(set(ids)) != len(ids):
            raise ValueError("player_id must be unique within a hand")
        if self.button not in seats:
            raise ValueError("button must be a seat in the hand")
        return self


class DecideIn(HandIn):
    hero_seat: int = Field(ge=0, lt=MAX_PLAYERS, description="the seat the AI plays; it must be its turn")
    hole_cards: str = Field(description='the AI\'s two cards, e.g. "AsKd"')


class RecordIn(HandIn):
    known_cards: dict[int, str] = Field(
        default_factory=dict,
        description="hole cards by seat that someone knows: the AI's own and those shown at showdown "
                    "(required for every seat that reached showdown)")


class CreateSession(BaseModel):
    name: str | None = Field(None, max_length=64)
    seconds: float = Field(10.0, ge=0.5, le=30.0, description="heads-up postflop solve budget per decision")
    mode: Literal["solve", "network"] = "solve"
    adaptive: bool = Field(True, description="A8.3: defend capped ranges vs players seen to iso / squeeze often")
    multiway_bluff: bool = False
    seed: int | None = Field(None, ge=0, lt=2**63)


# ----------------------------------------------------------------------------- replaying a spot


class _FixedDeck(Deck):
    """A deck in a given order (the known cards where they are dealt)."""

    def __init__(self, cards: list[int]) -> None:
        self.seed = 0
        self._cards = cards
        self._pos = 0


def _cards(text: str, what: str) -> tuple[int, ...]:
    try:
        return parse_cards(text)
    except ValueError as e:
        raise HTTPException(422, f"{what}: {e}") from None


def build_state(hand: HandIn, known: dict[int, tuple[int, ...]]) -> HandState:
    """Replay ``hand`` with the ``known`` hole cards by seat and its board;
    raises HTTPException(422) when the spot is not a legal hand."""
    stacks = {p.seat: p.stack for p in hand.players}
    board = _cards(hand.board, "board")
    if len(board) > 5:
        raise HTTPException(422, "board: at most 5 cards")
    for seat, cs in known.items():
        if seat not in stacks:
            raise HTTPException(422, f"known cards for seat {seat}, which is not in the hand")
        if len(cs) != 2:
            raise HTTPException(422, f"seat {seat}: hole cards must be exactly 2 cards")
    used = [c for cs in known.values() for c in cs] + list(board)
    if len(set(used)) != len(used):
        raise HTTPException(422, "the same card appears twice")

    # Where the engine deals each card: probe with an ordinary deck.
    probe_deck = Deck(0)
    order = list(probe_deck._cards)
    try:
        probe = HandState(stacks, hand.button, hand.small_blind, hand.big_blind, probe_deck)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    slots: list[int | None] = [None] * 52
    for seat in stacks:
        for pos, c in zip((order.index(c) for c in probe.hole_cards(seat)), known.get(seat, ())):
            slots[pos] = c
    n = 2 * len(stacks)
    for pos, c in zip((n + 1, n + 2, n + 3, n + 5, n + 7), board):  # a burn before each street
        slots[pos] = c
    rest = iter(c for c in range(52) if c not in set(used))
    state = HandState(stacks, hand.button, hand.small_blind, hand.big_blind,
                      _FixedDeck([c if c is not None else next(rest) for c in slots]))

    for i, a in enumerate(hand.actions):
        if state.is_over:
            raise HTTPException(422, f"action {i}: the hand is already over")
        if a.seat is not None and a.seat != state.to_act:
            raise HTTPException(422, f"action {i}: seat {state.to_act} is to act, not seat {a.seat}")
        try:
            state.apply(Action(ActionType(a.type), a.amount if a.type in ("bet", "raise") else 0))
        except IllegalActionError as e:
            raise HTTPException(422, f"action {i} (seat {state.to_act}): {e}") from None
    if len(state.board) != len(board):
        raise HTTPException(422, f"the actions reach {len(state.board)} board cards but {len(board)} were given")
    return state


# ----------------------------------------------------------------------------- sessions


@dataclass
class _LiveHand:
    agent: Any
    actions: list[tuple[int, Action]]  # the actions when it last decided
    decided: Action  # what it returned


@dataclass
class AISession:
    id: str
    config: dict[str, Any]
    created_at: float
    hands: list[dict[str, Any]] = field(default_factory=list)
    models: dict[str, Any] = field(default_factory=dict)  # player_id -> OpponentModel
    style: dict[str, list[int]] = field(default_factory=dict)  # player_id -> [iso chances, isos, sq chances, sqs]
    live: OrderedDict = field(default_factory=OrderedDict)  # (hand_id, hero_seat) -> _LiveHand
    lock: threading.RLock = field(default_factory=threading.RLock)

    def to_json(self) -> dict[str, Any]:
        return {"format": FORMAT, "id": self.id, "config": self.config, "created_at": self.created_at,
                "hands": self.hands}

    # ------------------------------------------------------------ models

    def observe(self, entry: dict[str, Any]) -> None:
        from backend.agents.web_realtime import preflop_style_chances
        from backend.opponent.model import OpponentModel
        from backend.opponent.stats import PublicHand

        hand = RecordIn.model_validate(entry["hand"])
        state = build_state(hand, {int(s): parse_cards(c) for s, c in hand.known_cards.items()})
        public = PublicHand.from_state(state)
        ids = {p.seat: p.player_id for p in hand.players}
        for seat, pid in ids.items():
            self.models.setdefault(pid, OpponentModel()).observe(public, seat)
        for seat, kind, took in preflop_style_chances(state.observation(None)):
            counts = self.style.setdefault(ids[seat], [0, 0, 0, 0])
            i = 0 if kind == "iso" else 2
            counts[i] += 1
            counts[i + 1] += took

    def rebuild(self) -> None:
        self.models, self.style = {}, {}
        for entry in self.hands:
            self.observe(entry)

    def opponent_models(self, hand: HandIn, hero_seat: int) -> dict:
        """The agent's ``opponent_models`` for this hand's seats (copies)."""
        from backend.agents.web_realtime import STYLE_KEY
        from backend.opponent.model import OpponentModel

        out: dict = {STYLE_KEY: {}}
        for p in hand.players:
            if p.seat == hero_seat:
                continue
            if p.player_id in self.models:
                out[p.seat] = OpponentModel.from_json(self.models[p.player_id].to_json())
            if p.player_id in self.style:
                out[STYLE_KEY][p.seat] = list(self.style[p.player_id])
        return out


class AIStore:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self._sessions: dict[str, AISession] = {}
        self._lock = threading.Lock()
        self.decide_lock = threading.Lock()  # one decision at a time: GPU solves

    def _path(self, sid: str) -> Path:
        return self.dir / f"{sid}.json"

    def create(self, config: dict[str, Any]) -> AISession:
        s = AISession(secrets.token_hex(8), config, time.time())
        with self._lock:
            self._sessions[s.id] = s
        self.save(s)
        return s

    def get(self, sid: str) -> AISession:
        if not _SESSION_ID.match(sid):
            raise HTTPException(404, "session not found")
        with self._lock:
            if sid not in self._sessions:
                path = self._path(sid)
                if not path.exists():
                    raise HTTPException(404, "session not found")
                d = json.loads(path.read_text())
                if d.get("format") != FORMAT:
                    raise HTTPException(500, f"unsupported session format {d.get('format')!r}")
                s = AISession(d["id"], d["config"], d["created_at"], d["hands"])
                s.rebuild()
                self._sessions[sid] = s
            return self._sessions[sid]

    def save(self, s: AISession) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self._path(s.id).with_suffix(".tmp")
        tmp.write_text(json.dumps(s.to_json()))
        tmp.replace(self._path(s.id))

    def list(self) -> list[str]:
        on_disk = {p.stem for p in self.dir.glob("*.json")} if self.dir.is_dir() else set()
        return sorted(on_disk | set(self._sessions))


# ----------------------------------------------------------------------------- views


def _summary(s: AISession) -> dict[str, Any]:
    return {"session_id": s.id, "name": s.config.get("name"), "config": s.config, "created_at": s.created_at,
            "hands": len(s.hands), "players": sorted(s.models)}


def _player(s: AISession, pid: str, full: bool) -> dict[str, Any]:
    from backend.opponent.stats import SUMMARY

    m = s.models[pid]
    stats = {}
    for g in (SUMMARY if not full else m.estimates()):
        e = m.estimate(g, level=0.9)
        stats[g] = {"n": e.opportunities, "mean": e.mean, "low": e.low, "high": e.high}
    iso = s.style.get(pid, [0, 0, 0, 0])
    return {"player_id": pid, "hands": m.hands, "stats": stats,
            "preflop_style": {"iso_chances": iso[0], "isos": iso[1], "squeeze_chances": iso[2], "squeezes": iso[3]}}


def _hand_view(entry: dict[str, Any]) -> dict[str, Any]:
    return {k: entry[k] for k in ("hand_no", "hand_id", "recorded_at", "hand", "result")}


def _actions(state: HandState) -> list[tuple[int, Action]]:
    return list(state.actions)


# ----------------------------------------------------------------------------- routes


def make_router(directory: Path | str | None = None) -> APIRouter:
    store = AIStore(Path(directory or os.environ.get("POKER_AI_DIR", "ai_sessions")))
    router = APIRouter(prefix="/api/ai", tags=["ai"])
    router.store = store  # type: ignore[attr-defined]  # tests

    @router.post("/sessions")
    def create_session(req: CreateSession) -> dict[str, Any]:
        config = req.model_dump()
        if config["seed"] is None:
            config["seed"] = random.SystemRandom().randrange(2**63)
        return _summary(store.create(config))

    @router.get("/sessions")
    def list_sessions() -> list[dict[str, Any]]:
        return [_summary(store.get(sid)) for sid in store.list()]

    @router.get("/sessions/{sid}")
    def get_session(sid: str) -> dict[str, Any]:
        return _summary(store.get(sid))

    @router.post("/sessions/{sid}/decide")
    def decide(sid: str, req: DecideIn) -> dict[str, Any]:
        from backend.agents.web_realtime import WebRealtimeAgent

        from .sessions import _strategy_label

        s = store.get(sid)
        if req.hero_seat not in {p.seat for p in req.players}:
            raise HTTPException(422, "hero_seat is not in the hand")
        state = build_state(req, {req.hero_seat: _cards(req.hole_cards, "hole_cards")})
        if state.is_over:
            raise HTTPException(409, "the hand is over: record it instead")
        if state.to_act != req.hero_seat:
            raise HTTPException(409, f"seat {state.to_act} is to act, not the AI's seat {req.hero_seat}")
        actions = _actions(state)
        key = (req.hand_id, req.hero_seat) if req.hand_id else None
        with store.decide_lock:
            with s.lock:
                live = s.live.get(key) if key else None
                memory = (live is not None and actions[:len(live.actions)] == live.actions
                          and len(actions) > len(live.actions)
                          and actions[len(live.actions)] == (req.hero_seat, live.decided))
                if memory:
                    agent = live.agent
                    s.live.move_to_end(key)
                else:
                    seed = derive_seed(s.config["seed"], "decide", req.hand_id or secrets.token_hex(8), req.hero_seat)
                    agent = WebRealtimeAgent(seed, seconds=s.config["seconds"], mode=s.config["mode"],
                                             opponent_models=s.opponent_models(req, req.hero_seat),
                                             multiway_bluff=s.config["multiway_bluff"],
                                             adaptive=s.config["adaptive"])
            legal = state.legal_actions()
            decision = agent.act(state.observation(req.hero_seat), legal)
            if key:
                with s.lock:
                    s.live[key] = _LiveHand(agent, actions, decision.action)
                    s.live.move_to_end(key)
                    while len(s.live) > LIVE_HANDS:
                        s.live.popitem(last=False)
        return {
            "action": {"type": str(decision.action.type), "amount": decision.action.amount},
            "action_probs": decision.action_probs,
            "strategy_source": str(decision.strategy_source),
            "label": _strategy_label(decision),
            "reason": decision.reason,
            "street": str(state.street),
            "in_hand_memory": memory,
            "legal": legal_json(legal),
            "opponents_modelled": {str(p.seat): s.models[p.player_id].hands for p in req.players
                                   if p.seat != req.hero_seat and p.player_id in s.models},
        }

    @router.post("/sessions/{sid}/hands")
    def record_hand(sid: str, req: RecordIn) -> dict[str, Any]:
        s = store.get(sid)
        known = {seat: _cards(c, f"known_cards[{seat}]") for seat, c in req.known_cards.items()}
        state = build_state(req, known)
        if not state.is_over:
            raise HTTPException(422, "the hand is not over")
        result = state.result()
        missing = sorted(set(result.showdown) - set(known))
        if missing:
            raise HTTPException(422, f"seats {missing} reached showdown: give their cards in known_cards")
        with s.lock:
            if req.hand_id and any(h["hand_id"] == req.hand_id for h in s.hands):
                raise HTTPException(409, f"hand_id {req.hand_id!r} is already recorded")
            entry = {
                "hand_no": (s.hands[-1]["hand_no"] + 1) if s.hands else 1,
                "hand_id": req.hand_id,
                "recorded_at": time.time(),
                "hand": req.model_dump(),
                "result": {"board": cards_str(result.board), "net": {str(k): v for k, v in result.net.items()},
                           "showdown": {str(k): {"cards": cards_str(c), "hand": result.hand_names[k]}
                                        for k, c in result.showdown.items()}},
            }
            s.observe(entry)
            s.hands.append(entry)
            for key in [k for k in s.live if k[0] == req.hand_id]:
                del s.live[key]
            store.save(s)
        return _hand_view(entry)

    @router.get("/sessions/{sid}/hands")
    def list_hands(sid: str) -> list[dict[str, Any]]:
        s = store.get(sid)
        return [{"hand_no": h["hand_no"], "hand_id": h["hand_id"], "recorded_at": h["recorded_at"],
                 "players": [p["player_id"] for p in h["hand"]["players"]], "board": h["result"]["board"],
                 "net": h["result"]["net"]} for h in s.hands]

    @router.get("/sessions/{sid}/hands/{hand_no}")
    def get_hand(sid: str, hand_no: int) -> dict[str, Any]:
        s = store.get(sid)
        for h in s.hands:
            if h["hand_no"] == hand_no:
                return _hand_view(h)
        raise HTTPException(404, "hand not found")

    @router.delete("/sessions/{sid}/hands/{hand_no}")
    def delete_hand(sid: str, hand_no: int) -> dict[str, Any]:
        """Remove a hand (e.g. recorded by mistake) and rebuild the models from the others."""
        s = store.get(sid)
        with s.lock:
            kept = [h for h in s.hands if h["hand_no"] != hand_no]
            if len(kept) == len(s.hands):
                raise HTTPException(404, "hand not found")
            s.hands = kept
            s.rebuild()
            store.save(s)
        return _summary(s)

    @router.get("/sessions/{sid}/players")
    def list_players(sid: str) -> list[dict[str, Any]]:
        s = store.get(sid)
        with s.lock:
            return [_player(s, pid, full=False) for pid in sorted(s.models)]

    @router.get("/sessions/{sid}/players/{pid}")
    def get_player(sid: str, pid: str) -> dict[str, Any]:
        s = store.get(sid)
        with s.lock:
            if pid not in s.models:
                raise HTTPException(404, "player not found")
            return _player(s, pid, full=True)

    return router
