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
seat can see. Blinds and equal per-player antes are supported; straddles are not.

Decisions are serialised (one at a time) because heads-up postflop solves
on the GPU and parallel solves overflow its memory. Passing the same
``hand_id`` on every decision of a hand keeps the AI's in-hand memory
(its own range updates and safe re-solving), as at the web table; without
it every decision starts from a fresh agent.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.agents.base import Decision, StrategySource
from backend.engine.actions import Action, ActionType, IllegalActionError
from backend.engine.cards import Deck, cards_str, derive_seed, parse_cards
from backend.engine.hand import MAX_PLAYERS, Event, EventType, HandState, PlayerView, SeatView, Street
from backend.engine.codec import legal_from_dict, view_from_dict

from .views import legal_json

logger = logging.getLogger(__name__)
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


class ForcedPostIn(BaseModel):
    seat: int = Field(ge=0, lt=MAX_PLAYERS, strict=True)
    live: int = Field(0, ge=0, strict=True)
    dead: int = Field(0, ge=0, strict=True)


class HandIn(BaseModel):
    hand_id: str | None = Field(None, min_length=1, max_length=128)
    small_blind: int = Field(gt=0)
    big_blind: int = Field(gt=0)
    ante: int = Field(0, ge=0)
    button: int = Field(ge=0, lt=MAX_PLAYERS)
    small_blind_seat: int | None = Field(None, ge=0, lt=MAX_PLAYERS, strict=True)
    big_blind_seat: int | None = Field(None, ge=0, lt=MAX_PLAYERS, strict=True)
    forced_posts: list[ForcedPostIn] = Field(default_factory=list)
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
        if self.big_blind_seat is None and self.small_blind_seat is not None:
            raise ValueError("big_blind_seat is required when small_blind_seat is specified")
        if self.big_blind_seat is None and self.button not in seats:
            raise ValueError("button must be a seat in the hand unless blind seats are explicit")
        if self.big_blind_seat is not None and (
            self.big_blind_seat not in seats
            or (self.small_blind_seat is not None and (
                self.small_blind_seat not in seats or self.small_blind_seat == self.big_blind_seat
            ))
        ):
            raise ValueError("blind seats must be distinct seats in the hand")
        forced_seats = [p.seat for p in self.forced_posts]
        if len(set(forced_seats)) != len(forced_seats) or not set(forced_seats) <= set(seats):
            raise ValueError("forced post seats must be unique seats in the hand")
        if any(p.live + p.dead == 0 for p in self.forced_posts):
            raise ValueError("forced post total must be positive")
        if any(p.live > self.big_blind for p in self.forced_posts):
            raise ValueError("live forced post must not exceed one big blind")
        if self.big_blind_seat is not None and set(forced_seats) & {
            s for s in (self.small_blind_seat, self.big_blind_seat) if s is not None
        }:
            raise ValueError("a blind seat cannot also make an extra forced post")
        return self


class DecideIn(HandIn):
    hero_seat: int = Field(ge=0, lt=MAX_PLAYERS, description="the seat the AI plays; it must be its turn")
    hole_cards: str = Field(description='the AI\'s two cards, e.g. "AsKd"')


class SnapshotDecideIn(BaseModel):
    hand_id: str = Field(min_length=1, max_length=128)
    view: dict[str, Any]
    legal: dict[str, Any]
    player_ids: dict[int, str] = Field(
        default_factory=dict,
        description="seat -> player_id: lets this decision use the session's opponent models")


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
    opponent_pool: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,32}$",
                                      description="opponent models shared by every session naming this pool, kept across restarts")


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

    blind_seats = ((hand.small_blind_seat, hand.big_blind_seat)
                   if hand.big_blind_seat is not None else None)
    forced_posts = {p.seat: (p.live, p.dead) for p in hand.forced_posts}

    # Where the engine deals each card: probe with an ordinary deck.
    probe_deck = Deck(0)
    order = list(probe_deck._cards)
    try:
        probe = HandState(stacks, hand.button, hand.small_blind, hand.big_blind, probe_deck,
                          ante=hand.ante, blind_seats=blind_seats, forced_posts=forced_posts)
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
                      _FixedDeck([c if c is not None else next(rest) for c in slots]),
                      ante=hand.ante, blind_seats=blind_seats, forced_posts=forced_posts)

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
    pool: OpponentPool | None = None

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
        if self.pool is None:
            self.models, self.style = {}, {}
        for entry in self.hands:
            self.observe(entry)

    def live_agent(self, key: tuple | None, actions: list[tuple[int, Action]], hero_seat: int) -> Any:
        """The agent from this hand's previous decision, if these actions continue from what it chose."""
        live = self.live.get(key) if key else None
        if (live is None or actions[:len(live.actions)] != live.actions or len(actions) <= len(live.actions)
                or actions[len(live.actions)] != (hero_seat, live.decided)):
            return None
        self.live.move_to_end(key)
        return live.agent

    def remember(self, key: tuple, agent: Any, actions: list[tuple[int, Action]], decided: Action) -> None:
        self.live[key] = _LiveHand(agent, actions, decided)
        self.live.move_to_end(key)
        while len(self.live) > LIVE_HANDS:
            self.live.popitem(last=False)

    def web_agent(self, seed: int, models: dict) -> Any:
        from backend.agents.web_realtime import WebRealtimeAgent

        return WebRealtimeAgent(seed, seconds=self.config["seconds"], mode=self.config["mode"],
                                opponent_models=models, multiway_bluff=self.config["multiway_bluff"],
                                adaptive=self.config["adaptive"])

    def opponent_models(self, players: dict[int, str], hero_seat: int) -> dict:
        """The agent's ``opponent_models`` for these seats (seat -> player_id; copies)."""
        from backend.agents.web_realtime import STYLE_KEY
        from backend.opponent.model import OpponentModel

        out: dict = {STYLE_KEY: {}}
        for seat, pid in players.items():
            if seat == hero_seat:
                continue
            if pid in self.models:
                out[seat] = OpponentModel.from_json(self.models[pid].to_json())
            if pid in self.style:
                out[STYLE_KEY][seat] = list(self.style[pid])
        return out


def _has_opening(view: PlayerView) -> bool:
    """Any snapshot with a rebuilt history shows the actions before the hero's first decision
    (``observe_opening`` cuts the rest), so every decision of a hand can record it."""
    return bool(view.history)


@dataclass
class OpponentPool:
    """Opponent models that outlive a session: every session naming this pool reads and feeds them."""

    path: Path
    models: dict[str, Any] = field(default_factory=dict)
    style: dict[str, list[int]] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def _add(self, view: PlayerView, player_ids: dict[int, str]) -> None:
        from backend.agents.web_realtime import observe_opening

        observe_opening(self.models, self.style, view, player_ids)

    def load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
                self._add(view_from_dict(entry["view"]), {int(s): p for s, p in entry["player_ids"].items()})
            except Exception:  # one damaged line must not cost the rest of the pool
                logger.exception("skipping an opponent observation in %s", self.path)
                continue
            self.seen.add(entry["hand_id"])

    def record(self, hand_id: str, raw_view: dict[str, Any], view: PlayerView, player_ids: dict[int, str]) -> None:
        with self.lock:
            if hand_id in self.seen:
                return
            try:
                self._add(view, player_ids)
                self.seen.add(hand_id)
                entry = {"hand_id": hand_id, "view": {**raw_view, "hole_cards": []}, "player_ids": player_ids}
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:  # the table's clock is running: the decision matters more than the statistics
                logger.exception("could not record the opponents of hand %s", hand_id)


class AIStore:
    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self._sessions: dict[str, AISession] = {}
        self._pools: dict[str, OpponentPool] = {}
        self._pools_lock = threading.Lock()
        self._lock = threading.Lock()
        self.decide_lock = threading.Lock()  # one decision at a time: GPU solves

    def _path(self, sid: str) -> Path:
        return self.dir / f"{sid}.json"

    def _attach_pool(self, s: AISession) -> None:
        name = s.config.get("opponent_pool")
        if not name:
            return
        with self._pools_lock:
            if name not in self._pools:
                self._pools[name] = OpponentPool(self.dir / "pools" / f"{name}.jsonl")
                self._pools[name].load()
            s.pool = self._pools[name]
        s.models, s.style, s.lock = s.pool.models, s.pool.style, s.pool.lock

    def create(self, config: dict[str, Any]) -> AISession:
        s = AISession(secrets.token_hex(8), config, time.time())
        self._attach_pool(s)
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
                self._attach_pool(s)
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


# ----------------------------------------------------------------------------- current snapshot validation


def _snapshot_int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f" in [{minimum}, {maximum}]" if maximum is not None else f" >= {minimum}"
        raise ValueError(f"{name} must be an integer{bounds}")
    return value


def _snapshot_card_list(value: Any, name: str, *, length: int | None = None) -> list[int]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        expected = f"exactly {length} cards" if length is not None else "a card list"
        raise ValueError(f"{name} must contain {expected}")
    return [_snapshot_int(card, name, maximum=51) for card in value]


def _validate_snapshot_history(history: Any, seats: set[int]) -> None:
    if not isinstance(history, list):
        raise ValueError("view.history must be a list")
    for row in history:
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError("each view.history row must have six codec fields")
        kind, street, seat, action, amount, cards = row
        if kind not in {e.value for e in EventType}:
            raise ValueError("view.history has an unknown event type")
        _snapshot_int(street, "history.street", maximum=3)
        if seat is not None and (_snapshot_int(seat, "history.seat", maximum=MAX_PLAYERS - 1) not in seats):
            raise ValueError("view.history names a seat that is not in the hand")
        if (kind == EventType.ACTION.value) != (action is not None):
            raise ValueError("view.history actions must be exactly the action events")
        if action is not None and (seat is None or action not in {a.value for a in ActionType}):
            raise ValueError("view.history has an invalid action")
        _snapshot_int(amount, "history.amount")
        _snapshot_card_list(cards, "history.cards")


def _history_problem(view) -> str | None:
    """Why the given actions do not reproduce this snapshot's board and chips, or None when they do."""
    from backend.agents.realtime import replay

    _, st, _ = replay(view)
    if st.street is not view.street or st.board != view.board:
        return "board or street differs from the snapshot"
    for s in view.seats:
        if st.bets[s.seat] != s.street_bet or st.behind_start[s.seat] - st.bets[s.seat] != s.stack:
            return f"seat {s.seat} chips differ from the snapshot"
    folded = {s.seat for s in view.seats if s.folded}
    if any(e.type is EventType.ACTION and e.action is ActionType.FOLD and e.seat not in folded for e in view.history):
        return "a seat that folded is still live in the snapshot"
    return None


def _history_actions(view) -> list[tuple[int, Action]]:
    return [(e.seat, Action(e.action, e.amount if e.action in (ActionType.BET, ActionType.RAISE) else 0))
            for e in view.history if e.type is EventType.ACTION]


def _validate_snapshot_view(v: dict[str, Any]) -> None:
    if not isinstance(v, dict):
        raise ValueError("view must be an object")
    required = {"seat", "hole_cards", "board", "street", "button", "small_blind", "big_blind",
                "pot", "current_bet", "to_act", "seats", "history", "is_over"}
    missing = required - v.keys()
    if missing:
        raise ValueError(f"view is missing fields: {', '.join(sorted(missing))}")
    seat = _snapshot_int(v["seat"], "view.seat", maximum=MAX_PLAYERS - 1)
    _snapshot_int(v["button"], "view.button", maximum=MAX_PLAYERS - 1)
    _snapshot_int(v["street"], "view.street", maximum=3)
    _snapshot_int(v["small_blind"], "view.small_blind", minimum=1)
    _snapshot_int(v["big_blind"], "view.big_blind", minimum=1)
    if v["small_blind"] > v["big_blind"]:
        raise ValueError("small_blind must not exceed big_blind")
    for name in ("ante", "pot", "current_bet"):
        _snapshot_int(v.get(name, 0), f"view.{name}")
    board = _snapshot_card_list(v["board"], "view.board")
    hole = _snapshot_card_list(v["hole_cards"], "view.hole_cards", length=2)
    expected_board = (0, 3, 4, 5)[v["street"]]
    if len(board) != expected_board:
        raise ValueError("board length does not match street")
    if len(set(board + hole)) != len(board) + len(hole):
        raise ValueError("a card appears more than once")
    if type(v["is_over"]) is not bool:
        raise ValueError("view.is_over must be boolean")
    raw_seats = v["seats"]
    if not isinstance(raw_seats, list) or not 2 <= len(raw_seats) <= MAX_PLAYERS:
        raise ValueError("view.seats must contain 2 to 9 seats")
    seen_seats: set[int] = set()
    all_cards = board + hole
    hero_row = None
    for row in raw_seats:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("each view.seats row must have seven codec fields")
        seat_id = _snapshot_int(row[0], "seat.seat", maximum=MAX_PLAYERS - 1)
        if seat_id in seen_seats:
            raise ValueError("view.seats contains a duplicate seat")
        seen_seats.add(seat_id)
        _snapshot_int(row[1], "seat.stack")
        street_bet = _snapshot_int(row[2], "seat.street_bet")
        total_bet = _snapshot_int(row[3], "seat.total_bet")
        if street_bet > total_bet:
            raise ValueError("seat.street_bet exceeds total_bet")
        if type(row[4]) is not bool or type(row[5]) is not bool:
            raise ValueError("seat.folded and seat.all_in must be boolean")
        if row[6] is not None:
            all_cards.extend(_snapshot_card_list(row[6], "seat.shown_cards", length=2))
        if seat_id == seat:
            hero_row = row
    if hero_row is None:
        raise ValueError("hero seat is missing from view.seats")
    if len(set(all_cards)) != len(all_cards):
        raise ValueError("a card appears more than once")
    _validate_snapshot_history(v["history"], seen_seats)
    _snapshot_int(v["to_act"], "view.to_act", maximum=MAX_PLAYERS - 1)
    sb, bb = v.get("sb_seat"), v.get("bb_seat")
    for name, blind_seat in (("sb_seat", sb), ("bb_seat", bb)):
        if blind_seat is not None:
            _snapshot_int(blind_seat, f"view.{name}", maximum=MAX_PLAYERS - 1)
            if blind_seat not in seen_seats:
                raise ValueError(f"{name} is not an occupied seat")
    if sb is not None and (bb is None or sb == bb):
        raise ValueError("blind seats must be distinct and include big_blind_seat")
    if v["button"] not in seen_seats and bb is None:
        raise ValueError("an empty button requires an explicit big_blind_seat")
    posts = v.get("forced_posts", {})
    if not isinstance(posts, dict):
        raise ValueError("view.forced_posts must be an object")
    normalized_posts: set[int] = set()
    for key, amounts in posts.items():
        if not isinstance(key, str) or not key.isdecimal():
            raise ValueError("forced_posts keys must be seat numbers")
        post_seat = _snapshot_int(int(key), "forced_posts.seat", maximum=MAX_PLAYERS - 1)
        if post_seat not in seen_seats or post_seat in normalized_posts:
            raise ValueError("forced_posts contains an invalid or duplicate seat")
        normalized_posts.add(post_seat)
        if post_seat in {seat for seat in (sb, bb) if seat is not None}:
            raise ValueError("a blind seat cannot also make an extra forced post")
        if not isinstance(amounts, list) or len(amounts) != 2:
            raise ValueError("each forced post must be [live, dead]")
        live = _snapshot_int(amounts[0], "forced_posts.live")
        dead = _snapshot_int(amounts[1], "forced_posts.dead")
        if live + dead == 0 or live > v["big_blind"]:
            raise ValueError("forced post must be positive and live cannot exceed one big blind")


def _validate_snapshot_legal(v: dict[str, Any], legal: dict[str, Any]) -> None:
    if not isinstance(legal, dict):
        raise ValueError("legal must be an object")
    required = {"seat", "can_fold", "can_check", "call_amount", "raise_type",
                "min_raise_to", "max_raise_to"}
    missing = required - legal.keys()
    if missing:
        raise ValueError(f"legal is missing fields: {', '.join(sorted(missing))}")
    hero = v["seat"]
    _snapshot_int(legal["seat"], "legal.seat", maximum=MAX_PLAYERS - 1)
    if legal["seat"] != hero or v["to_act"] != hero:
        raise ValueError("legal seat, view seat and to_act must match")
    if type(legal["can_fold"]) is not bool or type(legal["can_check"]) is not bool:
        raise ValueError("legal action flags must be boolean")
    call_amount = _snapshot_int(legal["call_amount"], "legal.call_amount")
    hero_row = next(row for row in v["seats"] if row[0] == hero)
    expected_call = min(max(v["current_bet"] - hero_row[2], 0), hero_row[1])
    if call_amount != expected_call:
        raise ValueError("call_amount does not match the current bet and hero stack")
    if legal["can_check"] != (call_amount == 0):
        raise ValueError("check availability does not match call_amount")
    raise_type = legal["raise_type"]
    if raise_type not in (None, "bet", "raise"):
        raise ValueError("raise_type must be bet, raise or null")
    low, high = legal["min_raise_to"], legal["max_raise_to"]
    if raise_type is None:
        if low is not None or high is not None:
            raise ValueError("raise bounds must be null when raising is unavailable")
    else:
        _snapshot_int(low, "legal.min_raise_to", minimum=1)
        _snapshot_int(high, "legal.max_raise_to", minimum=1)
        if low > high:
            raise ValueError("min_raise_to exceeds max_raise_to")
        if low <= v["current_bet"]:
            raise ValueError("min_raise_to must exceed the current bet")
        if (raise_type == "bet") != (v["current_bet"] == 0):
            raise ValueError("raise_type does not match the current bet")
        if high != hero_row[1] + hero_row[2]:
            raise ValueError("max_raise_to must equal hero stack plus street_bet")


SHORT_STACK_BB = 20
MAX_CHART_PLAYERS = 9


def _snapshot_history(view) -> tuple:
    """Preflop actions implied by the current street bets, since a snapshot carries no history.
    Each raise level's raiser is the first seat holding that bet after the previous raiser; others
    holding it called. A bet of exactly one big blind outside the BB is a limp."""
    from backend.agents.pushfold import preflop_order

    order = preflop_order(view)
    bets = {s.seat: s.street_bet for s in view.seats}
    events = [Event(EventType.ACTION, Street.PREFLOP, seat, ActionType.CALL)
              for seat in order if bets[seat] == view.big_blind and seat != view.bb_seat]
    start = 0
    for level in sorted({b for b in bets.values() if b > view.big_blind}):
        at = [seat for seat in order[start:] + order[:start] if bets[seat] == level]
        events.append(Event(EventType.ACTION, Street.PREFLOP, at[0], ActionType.RAISE, level))
        events += [Event(EventType.ACTION, Street.PREFLOP, seat, ActionType.CALL) for seat in at[1:]]
        start = order.index(at[0]) + 1
    return tuple(events)


def _with_empty_small_blind_chair(view):
    """The view with a folded, chipless small blind between button and big blind, so the chart for one
    more player gives every real seat its true position."""
    clockwise = sorted((s.seat for s in view.seats), key=lambda seat: (seat - view.button) % MAX_PLAYERS)
    new_seat = {seat: (0 if seat == view.button else i + 1) for i, seat in enumerate(clockwise)}
    seats = tuple(sorted((replace(s, seat=new_seat[s.seat]) for s in view.seats), key=lambda s: s.seat))
    return replace(
        view, seat=new_seat[view.seat], to_act=new_seat[view.to_act], button=0, sb_seat=1,
        bb_seat=new_seat[view.bb_seat], seats=(seats[0], SeatView(1, 0, 0, 0, True, False, None), *seats[1:]),
        history=tuple(replace(e, seat=new_seat[e.seat]) if e.seat is not None else e for e in view.history))


def _snapshot_preflop(view, legal, rng: random.Random, agent=None) -> tuple[Decision, str, list[str]]:
    """Frozen web preflop table (same as the web realtime agent) for a snapshot, with its real action
    history when it has one. Returns (decision, node looked up, approximations used)."""
    from backend.agents.pushfold import PushFoldAgent
    from backend.agents.web_preflop import effective_bb, lookup, tables
    from backend.solver.hands import class_index

    notes = []
    real_history = bool(view.history)
    seats = [s.seat for s in view.seats]
    if view.button not in seats and view.sb_seat in seats:
        # dead button: the seat right before the small blind acts last before the blinds, like a button
        view = replace(view, button=seats[seats.index(view.sb_seat) - 1])
        notes.append("dead_button_shifted")
    # shortcut: the chart has no node for a new player's forced blind, so it plays as a limp;
    # give forced posts their own nodes if they turn out to matter
    if view.forced_posts:
        notes.append("forced_post_as_limp")
        if real_history:
            limps = tuple(Event(EventType.ACTION, Street.PREFLOP, seat, ActionType.CALL)
                          for seat, (live, _) in view.forced_posts.items() if live == view.big_blind)
            view = replace(view, history=limps + view.history)
    view = replace(view, forced_posts={})
    if not real_history:
        view = replace(view, history=_snapshot_history(view))
        notes.append("history_inferred_from_bets")
    if view.sb_seat is None and view.button in seats and len(seats) < MAX_CHART_PLAYERS:
        view = _with_empty_small_blind_chair(view)
        notes.append("dead_small_blind_empty_chair")
        agent = None  # its opponent models are keyed by the snapshot's seat numbers
    # User's decision: under 20 BB effective, use the solved push/fold table (with this table's ante
    # when one is solved). Spots outside it (limps, non-all-in raises, under about 1.5 BB) stay on the chart.
    pushfold = PushFoldAgent(rng.getrandbits(63))
    found = pushfold._lookup(view) if effective_bb(view) < SHORT_STACK_BB else "deep"
    if not isinstance(found, str):
        game = found[0]["game"]
        if view.ante and not game.get("ante"):
            notes.append("pushfold_table_has_no_ante")
        node = f"pushfold/{game['players']}max/{game['stack_bb']:g}bb/ante{game.get('ante', 0):g}/{found[1]}"
        return pushfold.act(view, legal), node, notes
    data = agent.ante_table(view) if agent is not None else None
    if view.ante:
        notes.append("ante_table" if data else "chart_has_no_ante")
    defend = (real_history and agent is not None and agent.adaptive and "defend_groups" in tables()
              and agent._defend(view, legal))
    probs, table, key = lookup(view, legal, defend=defend, data=data)
    fold, passive, raising = (float(x) for x in probs[class_index(*view.hole_cards)])
    if legal.can_check:  # never fold when checking is free
        fold, passive = 0.0, passive + fold
    if legal.raise_type is None:  # raise size not readable on screen: continue passively instead
        passive, raising = passive + raising, 0.0
    q = rng.random()
    if q < fold:
        action = Action.fold()
    elif q < fold + passive:
        action = Action.check() if legal.can_check else Action.call()
    else:
        limpers = sum(s.street_bet == view.big_blind for s in view.seats) - 1
        to = (view.current_bet * table["raise_multiple"] if view.current_bet > view.big_blind else
              view.big_blind * (table["raise_bb"] + max(limpers, 0)))
        me = next(s for s in view.seats if s.seat == view.seat)
        if "jam_fraction" in table and to >= table["jam_fraction"] * (me.stack + me.street_bet):
            to = legal.max_raise_to
        action = Action(legal.raise_type, min(legal.max_raise_to, max(legal.min_raise_to, round(to))))
    decision = Decision(action, StrategySource.HEURISTIC,
                        action_probs={"fold": fold, "check" if legal.can_check else "call": passive, "raise": raising},
                        reason=(f"frozen preflop {(data or tables())['version']} {key} fold/passive/raise "
                                f"{fold:.2f}/{passive:.2f}/{raising:.2f}; HEURISTIC, not GTO"))
    return decision, key, notes


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

    @router.post("/sessions/{sid}/snapshot-decide")
    def snapshot_decide(sid: str, req: SnapshotDecideIn) -> dict[str, Any]:
        """Choose from a confirmed, current PlayerView. With its action history (``view.history``,
        checked against the snapshot's chips) postflop uses the web agent and preflop uses the real
        actions; without one, preflop infers them from the bets and postflop uses the rule agent."""
        from backend.agents.rule_based import RuleBasedAgent

        from .sessions import _strategy_label

        s = store.get(sid)
        v, l = req.view, req.legal
        try:
            _validate_snapshot_view(v)
            _validate_snapshot_legal(v, l)
            view = view_from_dict(v)
            legal = legal_from_dict(l)
            problem = _history_problem(view) if view.history else None
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise HTTPException(422, f"invalid current snapshot: {exc}") from None

        if view.is_over:
            raise HTTPException(409, "the hand is over")
        if view.seat is None or view.seat not in {seat.seat for seat in view.seats}:
            raise HTTPException(422, "hero seat is not in the snapshot")
        if view.to_act != view.seat or legal.seat != view.seat:
            raise HTTPException(409, "the snapshot does not show the hero to act")
        history = "provided" if view.history else "none"
        if problem:  # a live table still needs an answer: decide as if no history had been given
            view, history = replace(view, history=()), f"ignored: {problem}"

        decision, node, notes, memory, failure = None, None, [], False, None
        with s.lock:
            models = s.opponent_models(req.player_ids, view.seat)
            if s.pool is not None and req.player_ids and _has_opening(view):
                s.pool.record(req.hand_id, v, view, req.player_ids)
        if view.street is Street.PREFLOP:
            seed = derive_seed(s.config["seed"], "snapshot", req.hand_id, view.seat)
            try:
                decision, node, notes = _snapshot_preflop(view, legal, random.Random(seed), s.web_agent(seed, models))
            except (KeyError, ValueError) as exc:
                failure = f"chart does not cover this spot ({exc!r})"  # no button / blind, unknown node
        elif view.history:
            key, actions = (req.hand_id, view.seat), _history_actions(view)
            with store.decide_lock:
                with s.lock:
                    agent = s.live_agent(key, actions, view.seat)
                    memory = agent is not None
                    if agent is None:
                        agent = s.web_agent(derive_seed(s.config["seed"], "snapshot-postflop", req.hand_id, view.seat),
                                            models)
                try:
                    decision = agent.act(view, legal)
                except Exception as exc:  # the table's clock is running: fall back, and say so in the response
                    logger.exception("snapshot postflop agent failed")
                    failure = f"web postflop agent failed ({exc!r})"
                else:
                    with s.lock:
                        s.remember(key, agent, actions, decision.action)
        else:
            failure = "no action history"
        if decision is None:
            with store.decide_lock:
                seed = derive_seed(s.config["seed"], "snapshot-rule", req.hand_id, view.seat, str(view.street))
                rule = RuleBasedAgent(seed, board_aware=True).act(view, legal)
            decision = replace(rule, reason=f"{failure}; {rule.reason}")
            notes = [*notes, "rule_based_fallback"]
        if decision.action.type in (ActionType.BET, ActionType.RAISE) and legal.raise_type is None:
            # raise size not readable on screen: continue passively, as the preflop chart does
            decision = replace(decision, action=Action.check() if legal.can_check else Action.call())
            notes = [*notes, "raise_unavailable"]
        return {
            "action": {"type": str(decision.action.type), "amount": int(decision.action.amount)},
            "action_probs": decision.action_probs,
            "legal": legal_json(legal),
            "label": f"當前快照 · {_strategy_label(decision)}",
            "strategy_source": str(decision.strategy_source),
            "reason": decision.reason,
            "node": node,
            "history": history,
            "approximations": notes,
            "in_hand_memory": memory,
            "opponents_modelled": {str(seat): m.hands for seat, m in models.items() if isinstance(seat, int)},
        }

    @router.post("/validate")
    def validate(req: DecideIn) -> dict[str, Any]:
        """Replay a spot and return public state without running the AI."""
        if req.hero_seat not in {p.seat for p in req.players}:
            raise HTTPException(422, "hero_seat is not in the hand")
        state = build_state(req, {req.hero_seat: _cards(req.hole_cards, "hole_cards")})
        view = state.observation(req.hero_seat)
        return {
            "street": str(view.street),
            "board": cards_str(view.board),
            "to_act": view.to_act,
            "pot": view.pot,
            "current_bet": view.current_bet,
            "is_over": view.is_over,
            "sb_seat": view.sb_seat,
            "bb_seat": view.bb_seat,
            "forced_posts": {str(seat): {"live": live, "dead": dead}
                             for seat, (live, dead) in view.forced_posts.items()},
            "seats": {str(s.seat): {"stack": s.stack, "street_bet": s.street_bet,
                                    "total_bet": s.total_bet, "folded": s.folded,
                                    "all_in": s.all_in} for s in view.seats},
            "legal": legal_json(state.legal_actions()) if not view.is_over else None,
        }

    @router.post("/sessions/{sid}/decide")
    def decide(sid: str, req: DecideIn) -> dict[str, Any]:
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
                agent = s.live_agent(key, actions, req.hero_seat)
                memory = agent is not None
                if agent is None:
                    seed = derive_seed(s.config["seed"], "decide", req.hand_id or secrets.token_hex(8), req.hero_seat)
                    agent = s.web_agent(seed, s.opponent_models({p.seat: p.player_id for p in req.players},
                                                                req.hero_seat))
            legal = state.legal_actions()
            decision = agent.act(state.observation(req.hero_seat), legal)
            if key:
                with s.lock:
                    s.remember(key, agent, actions, decision.action)
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
