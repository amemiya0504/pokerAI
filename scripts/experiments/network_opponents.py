"""Policy networks of different architectures as opponents: how each one
plays. Collects, per architecture:

* imitation of the solver on the test spots (KL, EV regret; training report);
* exact exploitability of its strategy on the first 40 test spots
  (``python -m backend.policy exploit <model> --spots 40``);
* CPU latency of one decision (features included);
* duplicate arena results (bb/100, 95% CI) against the rule-based agent and
  against the M7 MLP (``net-mlp``). Preflop and flop are the same rule-based
  logic for every network agent, so the differences come from the turn and river.

Run from the repo root after training (README, "對手池"):

    .venv/bin/python scripts/experiments/network_opponents.py --deals 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from backend.arena.agents import NETWORK_OPPONENTS, AgentSpec
from backend.arena.runner import ArenaConfig, run_arena


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_SOURCE_DIRS = ("backend/policy", "backend/solver", "backend/dataset", "backend/engine")


def exploit(name: str, model: Path, spots: int) -> dict:
    fingerprint = hashlib.sha256(model.read_bytes() + f":{spots}".encode())
    sources = [Path(__file__).resolve()]
    sources.extend(
        path for directory in CACHE_SOURCE_DIRS
        for path in sorted((PROJECT_ROOT / directory).rglob("*.py"))
    )
    for source in sources:
        fingerprint.update(source.as_posix().encode())
        fingerprint.update(source.read_bytes())
    cache_key = fingerprint.hexdigest()
    out = Path("out/policy") / f"arch-{name}" / f"exploitability-{cache_key}.json"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-W", "ignore", "-m", "backend.policy", "exploit", str(model),
                        "--spots", str(spots), "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())["summary"]


def latency(model: Path) -> dict:
    from backend.policy.__main__ import _sample_spots
    from backend.policy.infer import PolicyRunner, benchmark

    runner = PolicyRunner(model, device="cpu")
    spots = _sample_spots(50)
    runner.predict(**spots[0])
    return benchmark(runner, spots)


def arena(a: str, b: str, deals: int, seed: int, workers: int = 1) -> dict:
    r = run_arena(ArenaConfig((AgentSpec(a, a), AgentSpec(b, b)), deals, seed), workers=workers,
                  progress=lambda done, total: print(f"{a} vs {b}: {done}/{total} deals", flush=True))
    s = next(x for x in r.stats if x.agent == a)
    return {"opponent": b, "hands": len(r.hand_rows), "bb_per_100": s.bb_per_100, "ci": [s.ci_low, s.ci_high]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--spots", type=int, default=40)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="out/policy/network_opponents.json")
    args = ap.parse_args()
    rows = {}
    for name, model in NETWORK_OPPONENTS.items():
        if not model.exists():
            print(f"{name}: no model at {model}, skipped", flush=True)
            continue
        rep = json.loads((model.parent / "report.json").read_text())
        row = {"model": str(model), "model_config": rep.get("model_config", {"arch": "mlp", **{
            k: rep["config"][k] for k in ("hidden", "layers", "range_dim", "dropout")}}),
               "parameters": rep.get("parameters"), "train_seconds": rep["seconds"], "test": rep["test"],
               "exploitability": exploit(name, model, args.spots), "latency_cpu": latency(model)}
        row["arena"] = [arena(f"net-{name}", "rule", args.deals, args.seed, args.workers)]
        if name != "mlp":
            row["arena"].append(arena(f"net-{name}", "net-mlp", args.deals, args.seed, args.workers))
        rows[name] = row
        Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
        e = row["exploitability"]
        print(f"{name}: KL {row['test']['kl_nats']:.3f}, EV regret {row['test']['ev_regret_pct_pot']:.2f}% | "
              f"exploitability river {e['river']['policy_pct_pot']['median']:.1f}% turn "
              f"{e['turn']['policy_pct_pot']['median']:.1f}% | CPU p50 {row['latency_cpu']['p50_ms']:.0f} ms | "
              + " ".join(f"vs {a['opponent']} {a['bb_per_100']:+.1f} [{a['ci'][0]:+.1f}, {a['ci'][1]:+.1f}]"
                         for a in row["arena"]), flush=True)
    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
