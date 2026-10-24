"""FastAPI app: REST for games and history, a websocket per live game.

    uvicorn --factory backend.api.app:create_app   # uses POKER_DB_URL, POKER_AI_DELAY
"""

from __future__ import annotations

import asyncio
import os
import random
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import func, select

from backend.db import Game, Hand, make_engine, make_sessionmaker, upgrade
from backend.db.repository import create_game, players_by_id, replay_hand
from backend.engine.cards import Deck, cards_str
from backend.engine.hand import HandState

from .sessions import GameSession
from .views import view_json

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


class CreateGame(BaseModel):
    name: str = Field("你", min_length=1, max_length=32)
    players: int = Field(6, ge=2, le=9)
    stack: int = Field(10_000, gt=0, le=10_000_000)
    small_blind: int = Field(50, gt=0)
    big_blind: int = Field(100, gt=0)
    opponents: Literal["random", "station", "rule", "pushfold", "realtime", "mixed"] = "mixed"
    seed: int | None = Field(None, ge=0, lt=2**63)

    @model_validator(mode="after")
    def _blinds(self) -> CreateGame:
        if self.small_blind > self.big_blind:
            raise ValueError("small_blind must not exceed big_blind")
        return self


def create_app(db_url: str | None = None, *, ai_delay: float | None = None,
               static_dir: Path | None = FRONTEND_DIST) -> FastAPI:
    db_url = db_url or os.environ.get("POKER_DB_URL", "sqlite:///poker.db")
    if ai_delay is None:
        ai_delay = float(os.environ.get("POKER_AI_DELAY", "0.6"))
    upgrade(db_url)
    sessions = make_sessionmaker(make_engine(db_url))
    live: dict[int, GameSession] = {}

    app = FastAPI(title="Poker AI")

    live_lock = threading.Lock()  # sync routes run in a threadpool

    def get_session(game_id: int) -> GameSession:
        with live_lock:
            if game_id not in live:
                with sessions() as db:
                    game = db.get(Game, game_id)
                    if game is None:
                        raise HTTPException(404, "game not found")
                    live[game_id] = GameSession.load(db, game, sessions, ai_delay)
            return live[game_id]

    app.state.get_session = get_session

    @app.post("/api/games")
    def new_game(req: CreateGame) -> dict[str, Any]:
        seed = req.seed if req.seed is not None else random.SystemRandom().randrange(2**63)
        players = [{"seat": 0, "name": req.name, "kind": "human", "stack": req.stack}]
        for seat in range(1, req.players):
            agent = req.opponents if req.opponents != "mixed" else ("station" if seat % 2 else "random")
            players.append({"seat": seat, "name": f"AI-{seat}", "kind": "ai",
                            "agent_config": {"agent": agent}, "stack": req.stack})
        with sessions() as db:
            game = create_game(db, max_seats=req.players, small_blind=req.small_blind,
                               big_blind=req.big_blind, seed=seed, mode="HvAI", players=players)
            return {"game_id": game.id, "seed": seed}

    @app.get("/api/games")
    def list_games() -> list[dict[str, Any]]:
        with sessions() as db:
            counts = dict(db.execute(select(Hand.game_id, func.count()).group_by(Hand.game_id)).all())
            games = db.scalars(select(Game).order_by(Game.id.desc()).limit(50)).all()
            return [{
                "game_id": g.id,
                "created_at": g.created_at.isoformat(),
                "players": len(g.seating),
                "small_blind": g.table.small_blind,
                "big_blind": g.table.big_blind,
                "hands": counts.get(g.id, 0),
            } for g in games]

    @app.get("/api/games/{game_id}/hands")
    def list_hands(game_id: int) -> list[dict[str, Any]]:
        session = get_session(game_id)
        human = str(session.human_seat)
        with sessions() as db:
            hands = db.scalars(select(Hand).where(Hand.game_id == game_id)
                               .order_by(Hand.hand_no.desc())).all()
            return [{
                "hand_id": h.id,
                "hand_no": h.hand_no,
                "board": h.board,
                "net": h.result["net"].get(human),
                "pot": sum(p["amount"] for p in h.result["pots"]),
                "showdown": bool(h.result["showdown"]),
                "created_at": h.created_at.isoformat(),
            } for h in hands]

    @app.get("/api/hands/{hand_id}")
    def hand_detail(hand_id: int) -> dict[str, Any]:
        """A finished hand step by step, as the human seat saw it."""
        with sessions() as db:
            hand = db.get(Hand, hand_id)
            if hand is None:
                raise HTTPException(404, "hand not found")
            final, record = replay_hand(hand)
            session = get_session(hand.game_id)
            names = players_by_id(db, [p["player_id"] for p in hand.start_stacks])
        seat = session.human_seat if session.human_seat in final.seats else None
        state = HandState({p["seat"]: p["start_stack"] for p in hand.start_stacks},
                          hand.button_seat, hand.small_blind, hand.big_blind, Deck(hand.deck_seed))
        def replay_view() -> dict[str, Any]:
            revealed_cards = ({
                s: state.hole_cards(s)
                for s in state.seats
                if s != seat
            } if state.is_over else None)
            return view_json(state.observation(seat), revealed_cards=revealed_cards)

        steps = [{"action": None, "view": replay_view()}]
        for s, action in final.actions:
            state.apply(action)
            steps.append({"action": {"seat": s, "type": str(action.type), "amount": action.amount},
                          "view": replay_view()})
        return {
            "hand_id": hand.id,
            "game_id": hand.game_id,
            "hand_no": hand.hand_no,
            "human_seat": seat,
            "button": hand.button_seat,
            "small_blind": hand.small_blind,
            "big_blind": hand.big_blind,
            "players": [{"seat": p["seat"], "name": names[p["player_id"]].name,
                         "start_stack": p["start_stack"]} for p in hand.start_stacks],
            "net": record["result"]["net"],
            "board": cards_str(final.board),
            "steps": steps,
        }

    @app.websocket("/ws/games/{game_id}")
    async def game_socket(ws: WebSocket, game_id: int) -> None:
        try:
            session = get_session(game_id)
        except HTTPException:
            await ws.close(code=4404)
            return
        await ws.accept()
        queue = session.subscribe()

        async def pump() -> None:
            while True:
                await ws.send_json(await queue.get())

        sender = asyncio.create_task(pump())
        try:
            await ws.send_json(session.snapshot())
            while True:
                msg = await ws.receive_json()
                err = await session.handle(msg if isinstance(msg, dict) else {})
                if err:
                    await ws.send_json({"type": "error", "message": err})
        except WebSocketDisconnect:
            pass
        finally:
            session.unsubscribe(queue)
            sender.cancel()
            with suppress(asyncio.CancelledError):
                await sender

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    return app

