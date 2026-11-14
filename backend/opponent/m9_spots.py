"""M9 evidence without arena noise: on the first test spots (turn and river,
the policy network's test split) and for each fixed opponent that can be
written as a tree policy (style agents, policy networks), the exact gain of
the exploiting strategy over the equilibrium against that opponent, and its
exact cost against the equilibrium. The opponent model is the same
history-built model the arena acceptance uses.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from backend.agents.style import STYLES
from backend.arena.agents import NETWORK_OPPONENTS
from backend.dataset.spots import make_spot
from backend.policy.data import SPLIT_EVERY

from .exploit import evaluate_spot, style_policy
from .m9 import build_models
from .model import OpponentModel


def _opponents() -> dict:
    out = {f"style-{s}": ("style", s) for s in STYLES}
    out |= {f"net-{n}": ("net", p) for n, p in NETWORK_OPPONENTS.items() if p.exists()}
    return out


def run(spots: int = 30, out: Path = Path("out/m9/spots.json"), models_dir: Path = Path("out/m9/models"),
        log=print, relative: bool = True) -> dict:
    """``relative``: tendencies relative to the GTO baseline's statistics
    (``models_dir / "gto.json"``, the agent's method); False: absolute
    frequencies (the first method, kept for the record)."""
    from backend.policy.exploit import runner_policy
    from backend.policy.infer import PolicyRunner

    opponents = _opponents()
    paths = build_models(models_dir, tuple(opponents))
    ref_path = models_dir / "gto.json"
    reference = OpponentModel.from_json(json.loads(ref_path.read_text())) if relative else None
    chosen = []
    for idx in range(0, 10_000, SPLIT_EVERY):
        chosen.append(make_spot(1, idx, {"river": 0.7, "turn": 0.3}))
        if len(chosen) >= spots:
            break
    summary, rows = {}, []
    for name, (kind, arg) in opponents.items():
        model = OpponentModel.from_json(json.loads(paths[name].read_text()))
        if kind == "net":
            runner = PolicyRunner(arg, device="cuda", cache_boards=128)
            policy = lambda street, r=runner: runner_policy(r)  # noqa: E731
        else:
            policy = lambda street, a=arg: style_policy(STYLES[a], street)  # noqa: E731
        res = []
        for spot in chosen:
            res += evaluate_spot(spot, policy(spot.street), model, reference=reference)
        gain = [r.gain_pct_pot for r in res]
        cost = [r.cost_pct_pot for r in res]
        summary[name] = {"results": len(res), "gain_mean": statistics.fmean(gain),
                         "gain_se": statistics.stdev(gain) / len(gain) ** 0.5,
                         "cost_mean": statistics.fmean(cost), "cost_se": statistics.stdev(cost) / len(cost) ** 0.5,
                         "gain_share_positive": sum(g > 0 for g in gain) / len(gain),
                         "eq_vs_opponent_mean": statistics.fmean(r.eq_vs_opponent_pct_pot for r in res)}
        rows += [{"opponent": name, **asdict(r)} for r in res]
        s = summary[name]
        log(f"{name}: gain {s['gain_mean']:+.2f} ± {1.96 * s['gain_se']:.2f}% pot, cost {s['cost_mean']:+.2f} ± "
            f"{1.96 * s['cost_se']:.2f}% pot ({s['gain_share_positive']:.0%} of spot-seats gain)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows, "spots": spots, "relative": relative,
                              "method": "relative-with-evidence" if relative else "absolute"}, indent=1,
                              default=float))
    return summary
