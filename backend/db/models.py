"""Database models (design doc section 6): Milestone 2 tables, Milestone 3
simulation tables, Milestone 4 strategy datasets / solver cache and
Milestone 7 model versions / training runs, Milestone 8 player statistics
and opponent models.

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


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    agents: Mapped[list[Any]] = mapped_column(JSON)  # [{name, kind, params}]
    seed: Mapped[int] = mapped_column(BigInteger)
    n_hands: Mapped[int] = mapped_column(Integer)
    git_commit: Mapped[str | None] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    seconds: Mapped[float] = mapped_column()
    data_path: Mapped[str | None] = mapped_column(String(512))  # Parquet directory
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    results: Mapped[list[BenchmarkResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("simulation_runs.id"), index=True)
    agent: Mapped[str] = mapped_column(String(64))
    hands: Mapped[int] = mapped_column(Integer)
    bb_per_100: Mapped[float] = mapped_column()
    stddev: Mapped[float | None] = mapped_column()
    ci_low: Mapped[float | None] = mapped_column()
    ci_high: Mapped[float | None] = mapped_column()

    run: Mapped[SimulationRun] = relationship(back_populates="results")


class GtoDataset(Base):
    """A strategy dataset and where it came from (design doc section 6)."""

    __tablename__ = "gto_datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(24))  # a StrategySource value
    license: Mapped[str] = mapped_column(String(256))
    row_count: Mapped[int] = mapped_column(Integer)
    path: Mapped[str] = mapped_column(String(512))
    details: Mapped[dict[str, Any]] = mapped_column(JSON)  # game, solver config, convergence
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SolverResultRow(Base):
    """Solver cache keyed by the request hash."""

    __tablename__ = "solver_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64), unique=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    iterations: Mapped[int] = mapped_column(Integer)
    exploitability: Mapped[float | None] = mapped_column()
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    source: Mapped[str] = mapped_column(String(24))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelVersion(Base):
    """A trained policy network (design doc section 6)."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    arch: Mapped[dict[str, Any]] = mapped_column(JSON)
    checkpoint_path: Mapped[str] = mapped_column(String(512))
    dataset_ids: Mapped[list[Any]] = mapped_column(JSON)  # gto_datasets ids or dataset names
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)  # test metrics, baselines, targets
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    runs: Mapped[list[TrainingRun]] = relationship(back_populates="model_version")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON)  # per-epoch history, device, versions
    resumed_from: Mapped[int | None] = mapped_column(Integer)  # epoch resumed from, if any
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    model_version: Mapped[ModelVersion] = relationship(back_populates="runs")


class PlayerStatistic(Base):
    """Raw counts behind an opponent model (design doc section 6): one row per
    statistic outcome and position, keeping the denominator."""

    __tablename__ = "player_statistics"
    __table_args__ = (UniqueConstraint("player_id", "stat", "position", "street"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), index=True)
    stat: Mapped[str] = mapped_column(String(48))  # "<group>:<outcome>", see backend.opponent.stats
    position: Mapped[str] = mapped_column(String(8))
    street: Mapped[str] = mapped_column(String(8))
    n_opportunities: Mapped[int] = mapped_column(Integer)
    n_hits: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class OpponentModelRow(Base):
    """A player's current posterior (rebuilt from ``player_statistics``)."""

    __tablename__ = "opponent_models"

    id: Mapped[int] = mapped_column(primary_key=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), unique=True)
    model_version: Mapped[str] = mapped_column(String(32))
    posterior: Mapped[dict[str, Any]] = mapped_column(JSON)
    sample_size: Mapped[int] = mapped_column(Integer)  # hands observed
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)
