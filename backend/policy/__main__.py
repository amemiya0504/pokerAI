"""Prepare data, train, and benchmark the GTO policy network.

    python -m backend.policy prepare out/datasets/gto-postflop-hu-v1 out/datasets/gto-postflop-hu-v2 \\
        --out out/policy/v1v2.npz
    python -m backend.policy train out/policy/v1v2.npz out/policy/run1 --db sqlite:///poker.db
    python -m backend.policy benchmark out/policy/run1/model.pt
    python -m backend.policy exploit out/policy/run1/model.pt --spots 100

``train`` resumes from ``<run>/checkpoint.pt`` when it exists (same config).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def register(db_url: str, run_dir: Path, report: dict) -> int:
    from backend.db import ModelVersion, TrainingRun, make_engine, make_sessionmaker, upgrade

    meta_file = Path(report["config"]["features"]).with_suffix(".json")
    datasets = json.loads(meta_file.read_text())["datasets"] if meta_file.exists() else []
    upgrade(db_url)
    with make_sessionmaker(make_engine(db_url))() as db:
        mv = ModelVersion(
            arch={"type": "PolicyNet", **{k: report["config"][k] for k in ("hidden", "layers", "range_dim", "dropout")}},
            checkpoint_path=str(run_dir / "model.pt"),
            dataset_ids=[f"{d['dataset']}/{d['version']}" for d in datasets],
            metrics={k: report[k] for k in ("test", "baselines", "targets", "meets_targets", "best_epoch")},
        )
        mv.runs = [TrainingRun(config=report["config"],
                               metrics={k: report[k] for k in ("history", "device", "torch", "cuda", "gpu", "seconds")},
                               resumed_from=report["resumed_from"])]
        db.add(mv)
        db.commit()
        return mv.id


def _sample_spots(n: int, seed: int = 0) -> list[dict]:
    """Random river/turn decision inputs for the latency benchmark."""
    from backend.dataset.spots import make_spot

    out = []
    for i in range(n):
        s = make_spot(seed, i, {"river": 0.5, "turn": 0.5})
        r0, r1 = s.ranges()
        live = [int(j) for j in (r0 > 0).nonzero()[0]]
        out.append({"board": s.board, "street": s.street, "pot": s.pot, "stack": s.stack, "inv": (0.0, 0.0),
                    "player": 0, "actions": ["x", "b50" if s.street == "turn" else "b75", "a"],
                    "reach_oop": r0, "reach_ip": r1, "combos": live[:1]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backend.policy", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("features")
    t.add_argument("run", type=Path)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch-size", type=int, default=8192)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--range-dim", type=int, default=64)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device")
    t.add_argument("--db", help="register the model version and training run here")
    e = sub.add_parser("exploit", help="exact exploitability of the network's policy on test spots")
    e.add_argument("model", type=Path)
    e.add_argument("--seed", type=int, default=1, help="dataset seed the spots come from")
    e.add_argument("--max-index", type=int, default=1000, help="spot indices below this")
    e.add_argument("--spots", type=int, default=100, help="how many test spots to measure")
    e.add_argument("--streets", default="river,turn")
    e.add_argument("--out", type=Path, help="write per-spot results and the summary as JSON")
    b = sub.add_parser("benchmark")
    b.add_argument("model", type=Path)
    b.add_argument("--decisions", type=int, default=50)
    args = ap.parse_args(argv)

    if args.cmd == "prepare":
        from .data import prepare

        print(json.dumps(prepare(args.datasets, args.out), indent=2))
        return 0
    if args.cmd == "train":
        from .train import TrainConfig, train

        cfg = TrainConfig(args.features, args.epochs, args.batch_size, args.lr, args.weight_decay,
                          args.hidden, args.layers, args.range_dim, args.dropout, args.seed)
        report = train(cfg, args.run, args.device)
        print(json.dumps({k: report[k] for k in ("test", "baselines", "targets", "meets_targets", "best_epoch")},
                         indent=2, default=float))
        if args.db:
            print(f"registered model version {register(args.db, args.run, report)} in {args.db}")
        return 0
    if args.cmd == "exploit":
        return _exploit(args)
    from .infer import PolicyRunner, benchmark

    runner = PolicyRunner(args.model, device="cpu")
    spots = _sample_spots(args.decisions)
    runner.predict(**spots[0])  # warm-up
    print(json.dumps(benchmark(runner, spots), indent=2))
    return 0


def _exploit(args) -> int:
    from dataclasses import asdict

    from backend.dataset.spots import make_spot

    from .data import SPLIT_EVERY
    from .exploit import evaluate_spot, runner_policy, summarize
    from .infer import PolicyRunner

    import torch

    # Offline evaluation, so the network may run on the GPU (the CPU latency
    # target is for single game decisions; see ``benchmark``).
    runner = PolicyRunner(args.model, device="cuda" if torch.cuda.is_available() else "cpu", cache_boards=128)
    policy = runner_policy(runner)
    wanted = set(args.streets.split(","))
    results = []
    # Test split only: spot index % 10 == 0 (the model never trained or selected on these).
    for idx in range(0, args.max_index, SPLIT_EVERY):
        spot = make_spot(args.seed, idx, {"river": 0.7, "turn": 0.3})
        if spot.street not in wanted:
            continue
        results.append(evaluate_spot(spot, policy))
        r = results[-1]
        print(f"spot {idx} {r.street}: solver {r.solver_pct_pot:.2f}% pot, network {r.policy_pct_pot:.2f}% pot",
              file=sys.stderr, flush=True)
        if len(results) >= args.spots:
            break
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "spots": [asdict(r) for r in results]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
