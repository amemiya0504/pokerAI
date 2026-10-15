"""Database models for Milestone 2 (design doc section 6).

A hand's source of truth is (deck seed, button, blinds, starting stacks,
action sequence); ``result`` and ``board`` are derived and kept for queries.
Tables for later milestones (statistics, solver results, models, ...) are
added with those milestones.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(8))  # human / ai
    agent_config: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PokerTable(Base):
    __tablename__ = "poker_tables"

    id: Mapped[int] = mapped_column(primary_key=True)
    max_seats: Mapped[int] = mapped_column(Integer)
    small_blind: Mapped[int] = mapped_column(Integer)
    big_blind: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Game(Base):
    __tablename__ = "games"

    id: Mapped[int] = mapped_column(primary_key=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("poker_tables.id"))
    mode: Mapped[str] = mapped_column(String(8))  # HvAI / AIvAI
    seed: Mapped[int] = mapped_column(BigInteger)
    # [{seat, player_id, start_stack}] at game creation
    seating: Mapped[list[Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    table: Mapped[PokerTable] = relationship()
    hands: Mapped[list[Hand]] = relationship(back_populates="game", order_by="Hand.hand_no")


class Hand(Base):
    __tablename__ = "hands"
    __table_args__ = (UniqueConstraint("game_id", "hand_no"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey("games.id"), index=True)
    hand_no: Mapped[int] = mapped_column(Integer)
    deck_seed: Mapped[int] = mapped_column(BigInteger)
    button_seat: Mapped[int] = mapped_column(Integer)
    small_blind: Mapped[int] = mapped_column(Integer)
    big_blind: Mapped[int] = mapped_column(Integer)
    # [{seat, player_id, start_stack}]
    start_stacks: Mapped[list[Any]] = mapped_column(JSON)
    board: Mapped[str] = mapped_column(String(10))
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    game: Mapped[Game] = relationship(back_populates="hands")
    actions: Mapped[list[ActionRow]] = relationship(
        back_populates="hand", order_by="ActionRow.seq", cascade="all, delete-orphan"
    )


class ActionRow(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    hand_id: Mapped[int] = mapped_column(ForeignKey("hands.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    seat: Mapped[int] = mapped_column(Integer)
    street: Mapped[str] = mapped_column(String(8))
    type: Mapped[str] = mapped_column(String(8))
    amount: Mapped[int] = mapped_column(Integer)
    strategy_source: Mapped[str | None] = mapped_column(String(24))
    action_probs: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    reason: Mapped[str | None] = mapped_column(String(256))

    hand: Mapped[Hand] = relationship(back_populates="actions")
