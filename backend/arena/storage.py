"""Writing arena runs: per-hand rows to Parquet, the summary to JSON and the DB."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .runner import ArenaResult

REPO = Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    """HEAD commit, suffixed with -dirty when the work tree has changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


def _finite(x: float) -> float | None:
    return x if math.isfinite(x) else None


def summary(result: ArenaResult, commit: str | None) -> dict[str, Any]:
    return {
        "config": result.config.to_dict(),
        "git_commit": commit,
        "hands": len(result.hand_rows),
        "seconds": result.seconds,
        "workers": result.workers,
        "hands_per_second": result.hands_per_second,
        "stats": [{k: (_finite(v) if isinstance(v, float) else v) for k, v in asdict(s).items()}
                  for s in result.stats],
    }


def write_parquet(result: ArenaResult, out_dir: Path, commit: str | None) -> Path:
    """hands.parquet (one row per hand: enough to replay it), players.parquet
    (one row per player per hand) and summary.json."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    deal, rot, seed, seating, actions, board = zip(*result.hand_rows) if result.hand_rows else ([],) * 6
    pq.write_table(pa.table({
        "deal": pa.array(deal, pa.int32()),
        "rotation": pa.array(rot, pa.int8()),
        "deck_seed": pa.array(seed, pa.int64()),
        "seating": pa.array(seating, pa.string()),
        "actions": pa.array(actions, pa.string()),
        "board": pa.array(board, pa.string()),
    }), out_dir / "hands.parquet")
    cols = list(zip(*result.player_rows)) if result.player_rows else [[]] * 7
    pq.write_table(pa.table({
        "deal": pa.array(cols[0], pa.int32()),
        "rotation": pa.array(cols[1], pa.int8()),
        "seat": pa.array(cols[2], pa.int8()),
        "agent": pa.array(cols[3], pa.string()),
        "position": pa.array(cols[4], pa.int8()),
        "net_chips": pa.array(cols[5], pa.int64()),
        "showdown": pa.array(cols[6], pa.bool_()),
    }), out_dir / "players.parquet")
    (out_dir / "summary.json").write_text(
        json.dumps(summary(result, commit), ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def save_run(db_url: str, result: ArenaResult, commit: str | None, data_path: Path | None) -> int:
    from backend.db import BenchmarkResult, SimulationRun, make_engine, make_sessionmaker, upgrade

    upgrade(db_url)
    with make_sessionmaker(make_engine(db_url))() as db:
        run = SimulationRun(
            agents=[a.to_dict() for a in result.config.agents],
            seed=result.config.seed,
            n_hands=len(result.hand_rows),
            git_commit=commit,
            config=result.config.to_dict(),
            seconds=result.seconds,
            data_path=str(data_path) if data_path else None,
        )
        run.results = [
            BenchmarkResult(agent=s.agent, hands=s.hands, bb_per_100=s.bb_per_100,
                            stddev=_finite(s.stddev), ci_low=_finite(s.ci_low),
                            ci_high=_finite(s.ci_high))
            for s in result.stats
        ]
        db.add(run)
        db.commit()
        return run.id
