"""Milestone 9 acceptance: the exploit engine wins more than the GTO baseline
against fixed opponents and does not lose significantly to GTO play
(design doc section 8).

Agents (heads-up, 100 BB, real-time solving, same time budget):

* GTO baseline: ``RealtimeAgent(use_opponent_model=False, exploit=False)``;
* exploit: ``RealtimeAgent`` with an opponent model built from that
  opponent's public history (``HISTORY_HANDS`` hands against the rule-based
  agent): range tracking from the model (M8) and the data-biased response
  in its solves (M9), relative to the GTO baseline's own statistics
  (``gto.json``, ``GTO_HISTORY_HANDS`` hands, other seeds than E2's model).

Gates, fixed before the first run:

* E1 fixed opponents: for each opponent in ``POOL``, the paired difference
  exploit - baseline in bb/100 (same deals, seats and opponent random seeds;
  95% CI clustered by deal). Pass: CI > 0 for at least 4 of the 6 opponents,
  and CI < 0 for none.
* E2 against GTO: the exploit agent, with a model of the baseline built from
  the baseline's own history, against the baseline (duplicate): its bb/100 is
  not significantly negative (95% CI upper bound >= 0).

Preflop is the same rule-based logic for every real-time agent (no solved
deep-stack preflop), so all differences come from postflop play.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from backend.arena.agents import REGISTRY, AgentSpec
from backend.arena.runner import ArenaConfig, run_arena

from .exploit import history_model

POOL = ("style-station", "style-maniac", "style-nit", "rule", "net-linear", "net-mlp")
HISTORY_HANDS = 2000
GTO_HISTORY_HANDS = 300
SECONDS = 2.0
DEALS = 200


def build_models(out_dir: Path, opponents=POOL, hands: int = HISTORY_HANDS, seed: int = 9) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in opponents:
        path = out_dir / f"{name}.json"
        if not path.exists():
            model = history_model(lambda s, n=name: REGISTRY[n](s), hands, seed)
            path.write_text(json.dumps(model.to_json()))
        paths[name] = path
    return paths


def build_gto_model(out_dir: Path, hands: int = GTO_HISTORY_HANDS, seconds: float = SECONDS, seed: int = 9,
                    name: str = "gto") -> Path:
    """The GTO baseline's statistics. "gto" (seed 9) is the reference that
    exploitation is relative to; "gto-opponent" (seed 10, other hands) is the
    model the exploit agent has of the baseline in E2."""
    path = out_dir / f"{name}.json"
    if not path.exists():
        from backend.agents.realtime import RealtimeAgent

        model = history_model(lambda s: RealtimeAgent(s, seconds=seconds, use_opponent_model=False, exploit=False),
                              hands, seed)
        path.write_text(json.dumps(model.to_json()))
    return path


def gto_spec(seconds: float = SECONDS) -> AgentSpec:
    return AgentSpec("gto", "realtime", {"seconds": seconds, "use_opponent_model": False, "exploit": False})


def exploit_spec(model_path: Path, reference_path: Path, seconds: float = SECONDS) -> AgentSpec:
    return AgentSpec("exploit", "realtime", {"seconds": seconds, "opponent_model_path": str(model_path),
                                             "reference_model_path": str(reference_path)})


def _nets(result, hero: str) -> dict[tuple[int, int], float]:
    return {(r[0], r[1]): r[5] for r in result.player_rows if r[3] == hero}


def paired(a: AgentSpec, b: AgentSpec, opponent: AgentSpec, deals: int, seed: int, workers: int,
           big_blind: int = 100) -> dict[str, Any]:
    """bb/100 of ``a`` minus ``b`` against the same opponent on the same deals."""
    ra = run_arena(ArenaConfig((a, opponent), deals, seed), workers)
    rb = run_arena(ArenaConfig((b, opponent), deals, seed), workers)
    na, nb = _nets(ra, a.name), _nets(rb, b.name)
    per_deal: dict[int, float] = {}
    for key, x in na.items():
        per_deal[key[0]] = per_deal.get(key[0], 0.0) + x - nb[key]
    hands = len(na)
    d = list(per_deal.values())
    scale = 100 / big_blind / (hands / len(d))  # per-deal sums -> bb/100
    mean = statistics.fmean(d) * scale
    se = statistics.stdev(d) / math.sqrt(len(d)) * scale if len(d) > 1 else float("nan")
    single = {s.agent: s.bb_per_100 for s in ra.stats} | {f"{s.agent} (b)": s.bb_per_100 for s in rb.stats}
    return {"opponent": opponent.name, "deals": len(d), "hands": hands, "diff_bb_per_100": mean,
            "ci": [mean - 1.96 * se, mean + 1.96 * se], "single": single,
            "seconds": ra.seconds + rb.seconds}


def head_to_head(a: AgentSpec, b: AgentSpec, deals: int, seed: int, workers: int) -> dict[str, Any]:
    r = run_arena(ArenaConfig((a, b), deals, seed), workers)
    s = next(x for x in r.stats if x.agent == a.name)
    return {"hands": len(r.hand_rows), "bb_per_100": s.bb_per_100, "ci": [s.ci_low, s.ci_high],
            "seconds": r.seconds}


def run(out_dir: Path = Path("out/m9"), deals: int = DEALS, seconds: float = SECONDS, workers: int = 3,
        seed: int = 19, log=print) -> dict[str, Any]:
    models = build_models(out_dir / "models")
    reference = build_gto_model(out_dir / "models", seconds=seconds)
    rows = []
    for name in POOL:
        r = paired(exploit_spec(models[name], reference, seconds), gto_spec(seconds), AgentSpec(name, name), deals,
                   seed, workers)
        rows.append(r)
        log(f"{name}: exploit - GTO {r['diff_bb_per_100']:+.1f} bb/100 [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] "
            f"({r['hands']} hands each, {r['seconds'] / 60:.0f} min)")
    gto_model = build_gto_model(out_dir / "models", seconds=seconds, seed=10, name="gto-opponent")
    h2h = head_to_head(exploit_spec(gto_model, reference, seconds), gto_spec(seconds), deals, seed, workers)
    log(f"exploit vs GTO: {h2h['bb_per_100']:+.1f} bb/100 [{h2h['ci'][0]:+.1f}, {h2h['ci'][1]:+.1f}]")
    positive = sum(r["ci"][0] > 0 for r in rows)
    negative = sum(r["ci"][1] < 0 for r in rows)
    e1 = positive >= 4 and negative == 0
    e2 = h2h["ci"][1] >= 0
    report = {"E1_fixed_opponents": e1, "E2_vs_gto": e2, "passed": e1 and e2, "significant_positive": positive,
              "significant_negative": negative, "fixed_opponents": rows, "vs_gto": h2h,
              "config": {"deals": deals, "seconds": seconds, "seed": seed, "pool": POOL,
                         "history_hands": HISTORY_HANDS, "gto_history_hands": GTO_HISTORY_HANDS}}
    (out_dir / "acceptance.json").write_text(json.dumps(report, indent=1, default=float))
    return report
