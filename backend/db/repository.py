"""Persisting games and hands, and rebuilding hand records from the database."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.engine.records import FORMAT, hand_record, replay

from .models import ActionRow, Game, Hand, Player, PokerTable


def create_game(
    db: Session,
    *,
    max_seats: int,
    small_blind: int,
    big_blind: int,
    seed: int,
    mode: str,
    players: list[dict[str, Any]],
) -> Game:
    """``players``: [{seat, name, kind, agent_config, stack}]."""
    table = PokerTable(max_seats=max_seats, small_blind=small_blind, big_blind=big_blind)
    db.add(table)
    seating = []
    for p in players:
        row = Player(name=p["name"], kind=p["kind"], agent_config=p.get("agent_config"))
        db.add(row)
        db.flush()
        seating.append({"seat": p["seat"], "player_id": row.id, "start_stack": p["stack"]})
    game = Game(table=table, mode=mode, seed=seed, seating=seating)
    db.add(game)
    db.commit()
    return game


def save_hand(db: Session, game_id: int, record: dict[str, Any]) -> Hand:
    """Store a finished hand record (as produced by ``hand_record``)."""
    hand = Hand(
        game_id=game_id,
        hand_no=record["hand_no"],
        deck_seed=record["deck_seed"],
        button_seat=record["button"],
        small_blind=record["small_blind"],
        big_blind=record["big_blind"],
        start_stacks=[
            {"seat": p["seat"], "player_id": p["player_id"], "start_stack": p["start_stack"]}
            for p in record["players"]
        ],
        board=record["board"],
        result=record["result"],
    )
    hand.actions = [
        ActionRow(
            seq=a["seq"],
            seat=a["seat"],
            street=a["street"],
            type=a["type"],
            amount=a["amount"],
            strategy_source=a["strategy_source"],
            action_probs=a["action_probs"],
            reason=a["reason"],
        )
        for a in record["actions"]
    ]
    db.add(hand)
    db.commit()
    return hand


def players_by_id(db: Session, ids: list[int]) -> dict[int, Player]:
    rows = db.scalars(select(Player).where(Player.id.in_(ids))).all()
    return {p.id: p for p in rows}


def replay_hand(hand: Hand):
    """Rebuild the hand from seed and actions; returns (HandState, full record)."""
    minimal = {
        "format": FORMAT,
        "deck_seed": hand.deck_seed,
        "button": hand.button_seat,
        "small_blind": hand.small_blind,
        "big_blind": hand.big_blind,
        "players": hand.start_stacks,
        "actions": [{"seq": a.seq, "seat": a.seat, "type": a.type, "amount": a.amount}
                    for a in hand.actions],
    }
    state = replay(minimal)
    decisions = [{"strategy_source": a.strategy_source, "action_probs": a.action_probs,
                  "reason": a.reason} for a in hand.actions]
    player_ids = {p["seat"]: p["player_id"] for p in hand.start_stacks}
    record = hand_record(state, hand_no=hand.hand_no, decisions=decisions, player_ids=player_ids)
    if record["result"] != hand.result:
        raise ValueError(f"hand {hand.id}: stored result does not match replay")
    return state, record
