"""Persisting games and hands, and rebuilding hand records from the database."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.engine.records import FORMAT, hand_record, replay

from .models import ActionRow, Game, Hand, OpponentModelRow, Player, PlayerStatistic, PokerTable


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


def record_statistics(db: Session, record: dict[str, Any]) -> None:
    """Add a finished hand's public statistics to every seated player's
    ``player_statistics`` counts and refresh their ``opponent_models`` row
    (Milestone 8). Only public information is used (no hole cards except
    those shown at showdown)."""
    from backend.opponent.model import MODEL_VERSION, OpponentModel
    from backend.opponent.stats import GROUPS, PublicHand, group_street, observations

    hand = PublicHand.from_record(record)
    by_seat: dict[int, list] = {}
    for o in observations(hand):
        by_seat.setdefault(o.seat, []).append(o)
    for p in record["players"]:
        pid = p.get("player_id")
        if pid is None:
            continue
        rows = {(r.stat, r.position, r.street): r
                for r in db.scalars(select(PlayerStatistic).where(PlayerStatistic.player_id == pid))}
        for o in by_seat.get(p["seat"], []):
            street = group_street(o.group)
            for outcome in GROUPS[o.group]:
                key = (f"{o.group}:{outcome}", o.position, street)
                row = rows.get(key)
                if row is None:
                    row = rows[key] = PlayerStatistic(player_id=pid, stat=key[0], position=o.position, street=street,
                                                      n_opportunities=0, n_hits=0)
                    db.add(row)
                row.n_opportunities += 1
                row.n_hits += int(outcome == o.outcome)
        current = db.scalar(select(OpponentModelRow).where(OpponentModelRow.player_id == pid))
        hands = (current.sample_size if current else 0) + 1
        model = OpponentModel.from_statistic_rows(
            [{"stat": r.stat, "position": r.position, "n_hits": r.n_hits} for r in rows.values()], hands=hands)
        posterior = {"model": model.to_json(),
                     "estimates": {g: {"opportunities": e.opportunities, "mean": e.mean, "low": e.low,
                                       "high": e.high, "level": e.level}
                                   for g, e in model.estimates().items() if e.opportunities}}
        if current is None:
            db.add(OpponentModelRow(player_id=pid, model_version=MODEL_VERSION, posterior=posterior,
                                    sample_size=hands))
        else:
            current.model_version, current.posterior, current.sample_size = MODEL_VERSION, posterior, hands
    db.commit()


def opponent_model(db: Session, player_id: int):
    """The player's ``OpponentModel`` rebuilt from the raw counts (None if never seen)."""
    from backend.opponent.model import OpponentModel

    rows = db.scalars(select(PlayerStatistic).where(PlayerStatistic.player_id == player_id)).all()
    if not rows:
        return None
    current = db.scalar(select(OpponentModelRow).where(OpponentModelRow.player_id == player_id))
    return OpponentModel.from_statistic_rows(
        [{"stat": r.stat, "position": r.position, "n_hits": r.n_hits} for r in rows],
        hands=current.sample_size if current else 0)
